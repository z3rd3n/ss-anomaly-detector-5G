import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
from tqdm import tqdm
import os
import json
from collections import Counter

class AnomalySequenceDataset(Dataset):
    def __init__(
        self, 
        parquet_path,
        all_features=['SFN', 'Slot', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI'],
        all_feature_dims=[1024, 31, 16, 33, 2, 9, 2],
        seq_len=50,
        sample_files=None,
        sample_fraction=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        is_training=True
    ):
        self.parquet_path = parquet_path
        self.all_features = all_features
        self.all_feature_dims = all_feature_dims
        self.seq_len = seq_len
        self.device = device
        self.is_training = is_training
        
        # Anomaly mapping for faster label conversion
        self.ANOMALY_MAPPING = {
            "unnecessary_retx": 1,
            "missing_retx": 2,
            "new_data_no_retx": 3,
            "max_retx_achieved": 4,
        }
        
        # Get file IDs to process
        self.file_ids = self.getfile_ids(sample_files, sample_fraction)
        
        # Prepare sequences
        self.sequences = []
        self.labels = []
        self.timestamps = []
        
        # Anomaly counters
        self.anomaly_counts = Counter()
        self.normal_count = 0
        
        # Create sequences
        self._create_sequences()
    
    def getfile_ids(self, sample_files, sample_fraction):
        """Get the file IDs to process using PyArrow directly"""
        # Read all file IDs from the parquet file
        table = pq.read_table(self.parquet_path, columns=['file_id'])
        unique_file_ids = np.unique(table['file_id'].to_numpy())

        # Sample files if needed
        if sample_files is not None:
            return sample_files

        # Randomly sample a fraction of files for training
        if sample_fraction < 1.0:
            np.random.seed(42)  # For reproducibility
            sampled_files = np.random.choice(
                unique_file_ids, 
                size=int(len(unique_file_ids) * sample_fraction),
                replace=False
            )
            return sampled_files

        return unique_file_ids
    
    def insight_to_label(self, insights):
        """Convert batch of insights to labels using torch operations"""
        # Convert insights to a list if it's a numpy array
        if isinstance(insights, np.ndarray):
            insights = insights.tolist()
        
        labels = []
        for insight in insights:
            if not insight or isinstance(insight, float):  # Check for None, empty string, or NaN
                labels.append(0)
                continue
                
            # Parse anomalies from the insight string
            anomalies = [a.strip() for a in insight.split(",") if a.strip()]
            if not anomalies:
                labels.append(0)
                continue
                
            # Special case: remove max_retx_achieved if other anomalies exist
            if len(anomalies) > 1 and "max_retx_achieved" in anomalies:
                anomalies = [a for a in anomalies if a != "max_retx_achieved"]

            # Get valid anomaly labels
            valid_labels = [self.ANOMALY_MAPPING.get(a) for a in anomalies if a in self.ANOMALY_MAPPING]

            # Return 0 if no valid anomalies
            if not valid_labels:
                labels.append(0)
            else:
                # Return the smallest valid label
                labels.append(min(valid_labels))
                
        return torch.tensor(labels, dtype=torch.long)
    
    def _create_sequences(self):
        """Create sequences from all files using PyArrow and torch operations"""
        print(f"Creating sequences from {len(self.file_ids)} files...")
        
        for file_id in tqdm(self.file_ids, desc="Processing files"):
            # Read data for this file_id using PyArrow
            table = pq.read_table(
                self.parquet_path, 
                filters=[('file_id', '==', file_id)]
            )
            
            # Convert to numpy arrays for faster processing
            all_columns = table.column_names
            data_dict = {col: table[col].to_numpy() for col in all_columns}
            
            # Create tensors for all features (treating all as categorical)
            feature_data = torch.tensor(
                np.column_stack([data_dict[col] for col in self.all_features]), 
                dtype=torch.long
            )
            
            # Convert insights to labels
            labels = self.insight_to_label(data_dict['insight'])
            
            # Count normal timesteps
            self.normal_count += (labels == 0).sum().item()
            
            # Count anomalies by type
            for label in range(1, 5):
                self.anomaly_counts[label] += (labels == label).sum().item()
            
            # Check if we're creating sequences for training or evaluation
            if self.is_training:
                self._process_file_training(
                    feature_data,
                    labels, 
                    data_dict['timestamp_str']
                )
            else:
                self._process_file_evaluation(
                    feature_data,
                    labels, 
                    data_dict['timestamp_str']
                )
    
    def _process_file_training(self, feature_data, labels, timestamps):
        """Process a single file to create sequences for training with anomaly-centered approach"""
        # Find anomaly indices (label != 0)
        anomaly_indices = torch.nonzero(labels != 0, as_tuple=True)[0]
        
        # Process sequences centered around anomalies
        for idx in anomaly_indices:
            idx = idx.item()
            label = labels[idx].item()
            
            # Get sequence start index (considering previous seq_len-1 timesteps)
            start_idx = max(0, idx - (self.seq_len - 1))
            
            # Extract sequence - this includes the anomaly point
            feat_seq = feature_data[start_idx:(idx + 1)]
            seq_labels = labels[start_idx:(idx + 1)]
            
            # Extract sequence timestamps - one for each timestep
            seq_timestamps = timestamps[start_idx:(idx + 1)].tolist()
            
            # If sequence is shorter than seq_len, pad with zeros
            if len(feat_seq) < self.seq_len:
                # Create padding tensors
                pad_length = self.seq_len - len(feat_seq)
                
                feat_padding = torch.zeros(pad_length, len(self.all_features), dtype=torch.long)
                label_padding = torch.zeros(pad_length, dtype=torch.long)
                
                # Pad timestamps with empty strings
                timestamp_padding = [""] * pad_length
                
                # Combine padding with actual sequence
                feat_seq = torch.cat([feat_padding, feat_seq], dim=0)
                seq_labels = torch.cat([label_padding, seq_labels], dim=0)
                seq_timestamps = timestamp_padding + seq_timestamps
            
            # Store sequence, label and timestamp
            self.sequences.append(feat_seq)
            self.labels.append(seq_labels)
            self.timestamps.append(seq_timestamps)
        
        # Add sequences with only normal labels
        # Only create normal sequences if there are enough normal timesteps
        if len(labels) >= self.seq_len:
            self._add_normal_sequences(
                feature_data, labels, timestamps
            )

    def _add_normal_sequences(self, feature_data, labels, timestamps):
        """Add sequences that only consist of normal labels"""
        # Find continuous chunks of normal (label=0) points
        normal_mask = labels == 0
        
        # Convert to numpy for easier processing of continuous chunks
        normal_indices = torch.nonzero(normal_mask, as_tuple=True)[0].numpy()
        
        # Find continuous chunks
        if len(normal_indices) > 0:
            # Split into continuous chunks
            chunks = np.split(normal_indices, np.where(np.diff(normal_indices) != 1)[0] + 1)
            
            # Filter chunks that are long enough
            valid_chunks = [chunk for chunk in chunks if len(chunk) >= self.seq_len]
            
            # Determine how many normal sequences to add based on the anomaly distribution
            # Based on your statistics, normal sequences should be about 15% of anomaly sequences
            # to maintain a reasonable balance
            anomaly_count = len(self.sequences)
            target_normal_count = min(int(anomaly_count * 0.15), len(valid_chunks))
            
            if target_normal_count > 0 and valid_chunks:
                # Randomly select chunks to sample from
                np.random.seed(42)  # For reproducibility
                selected_chunks = np.random.choice(
                    len(valid_chunks), 
                    size=min(target_normal_count, len(valid_chunks)),
                    replace=False
                )
                
                for chunk_idx in selected_chunks:
                    chunk = valid_chunks[chunk_idx]
                    
                    # Select a random starting point that allows a full sequence
                    if len(chunk) > self.seq_len:
                        start_pos = np.random.randint(0, len(chunk) - self.seq_len + 1)
                        chunk = chunk[start_pos:start_pos + self.seq_len]
                    
                    # Extract sequence
                    feat_seq = feature_data[chunk]
                    seq_labels = labels[chunk]
                    seq_timestamps = [timestamps[i] for i in chunk]
                    
                    # Store sequence
                    self.sequences.append(feat_seq)
                    self.labels.append(seq_labels)
                    self.timestamps.append(seq_timestamps)

    def _process_file_evaluation(self, feature_data, labels, timestamps):
        """Process a single file to create non-overlapping sequences for evaluation"""
        # For evaluation, we create non-overlapping sequences covering the entire dataset
        total_length = len(feature_data)
        
        # Process full sequences
        for start_idx in range(0, total_length, self.seq_len):
            end_idx = min(start_idx + self.seq_len, total_length)
            
            # Skip if the sequence is too short
            if end_idx - start_idx < self.seq_len:
                # Create a sequence that includes the remaining data
                start_idx = max(0, total_length - self.seq_len)
                end_idx = total_length
                
                # Skip if we've already processed this range
                if start_idx < (total_length - self.seq_len):
                    continue
            
            # Extract sequence
            feat_seq = feature_data[start_idx:end_idx]
            seq_labels = labels[start_idx:end_idx]
            seq_timestamps = timestamps[start_idx:end_idx].tolist()
            
            # If sequence is shorter than seq_len, pad with zeros
            if len(feat_seq) < self.seq_len:
                # Create padding tensors
                pad_length = self.seq_len - len(feat_seq)
                
                feat_padding = torch.zeros(pad_length, len(self.all_features), dtype=torch.long)
                label_padding = torch.zeros(pad_length, dtype=torch.long)
                
                # Pad timestamps with empty strings
                timestamp_padding = [""] * pad_length
                
                # Combine padding with actual sequence
                feat_seq = torch.cat([feat_seq, feat_padding], dim=0)  # Padding at the end for evaluation
                seq_labels = torch.cat([seq_labels, label_padding], dim=0)
                seq_timestamps = seq_timestamps + timestamp_padding
            
            # Store sequence, label and timestamp
            self.sequences.append(feat_seq)
            self.labels.append(seq_labels)
            self.timestamps.append(seq_timestamps)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        feature_data = self.sequences[idx]
        labels = self.labels[idx]
        timestamps = self.timestamps[idx]  # Now a list of timestamps for each timestep
        
        return {
            'feature_data': feature_data,  # Shape: [seq_len, num_features]
            'label': labels,  # Shape: [seq_len]
            'timestamp': timestamps  # List of length seq_len
        }
    
    def report_statistics(self):
        """Report dataset statistics"""
        print("\nDataset Statistics:")
        print(f"Total sequences: {len(self.sequences)}")
        
        if self.is_training:
            # For training, report anomaly-specific statistics
            anomaly_sequences = sum(1 for seq_labels in self.labels if torch.any(seq_labels != 0))
            normal_sequences = len(self.sequences) - anomaly_sequences
            
            print(f"Anomaly sequences: {anomaly_sequences}")
            print(f"Normal-only sequences: {normal_sequences}")
        
        print(f"Normal timesteps: {self.normal_count}")
        
        # Count the number of instances with each label in the sequences
        zero_label_count = sum((seq_labels == 0).sum().item() for seq_labels in self.labels)
        one_label_count = sum((seq_labels == 1).sum().item() for seq_labels in self.labels)
        two_label_count = sum((seq_labels == 2).sum().item() for seq_labels in self.labels)
        three_label_count = sum((seq_labels == 3).sum().item() for seq_labels in self.labels)
        four_label_count = sum((seq_labels == 4).sum().item() for seq_labels in self.labels)
        
        total_elements = sum(len(seq_labels) for seq_labels in self.labels)
        
        print(f"Instances with label 0 in sequences: {zero_label_count} ({zero_label_count / total_elements:.2%})")
        print(f"Instances with label 1 in sequences: {one_label_count} ({one_label_count / total_elements:.2%})")
        print(f"Instances with label 2 in sequences: {two_label_count} ({two_label_count / total_elements:.2%})")
        print(f"Instances with label 3 in sequences: {three_label_count} ({three_label_count / total_elements:.2%})")
        print(f"Instances with label 4 in sequences: {four_label_count} ({four_label_count / total_elements:.2%})")
        
        print("Anomaly counts:")
        anomaly_names = {
            1: "unnecessary_retx",
            2: "missing_retx",
            3: "new_data_no_retx",
            4: "max_retx_achieved"
        }
        
        for label, count in sorted(self.anomaly_counts.items()):
            print(f"  - {anomaly_names[label]} (label {label}): {count}")
        
        # Calculate class weights
        class_counts = torch.tensor([
            zero_label_count,
            one_label_count,
            two_label_count,
            three_label_count,
            four_label_count
        ], dtype=torch.float)
        
        total_samples = class_counts.sum()
        num_classes = len(class_counts)
        
        class_weights = total_samples / (class_counts * num_classes)
        class_weights[class_counts == 0] = 0.0  # Handle division by zero
        
        print("Class weights calculated:")
        for i in range(num_classes):
            if i == 0:
                class_name = "normal"
            else:
                class_name = anomaly_names[i]
            print(f"  - Class {i} ({class_name}): count={class_counts[i]}, weight={class_weights[i]:.4f}")
        
        self.class_weights = class_weights
        

def custom_collate_fn(batch):
    """
    Collate function that stacks batch items without moving to device
    Device transfer should happen in the training loop after pinning
    """
    feature_data = torch.stack([item['feature_data'] for item in batch])
    # Shape: [batch_size, seq_len, num_features]
    
    labels = torch.stack([item['label'] for item in batch])
    # Shape: [batch_size, seq_len]
    
    # Create a list of lists for timestamps
    timestamps = [item['timestamp'] for item in batch]
    # Shape: [batch_size, seq_len] as a list of lists of strings
    
    return {
        'feature_data': feature_data,
        'label': labels,
        'timestamp': timestamps
    }

def create_data_loaders(
    train_parquet_path='unscaled_pdsch_val.parquet', 
    test_parquet_path='unscaled_pdsch_val_min.parquet',
    all_features=['SFN', 'Slot', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI'],
    all_feature_dims=[1024, 31, 16, 33, 2, 9, 2],
    seq_len=50,
    batch_size=64,
    num_workers=min(os.cpu_count(), 4),
    sample_fraction=1.0,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # Create training dataset
    print("Creating training dataset...")
    train_dataset = AnomalySequenceDataset(
        parquet_path=train_parquet_path,
        all_features=all_features,
        all_feature_dims=all_feature_dims,
        seq_len=seq_len,
        sample_fraction=sample_fraction,
        device=device,
        is_training=True  # Training mode
    )
    
    # Create test dataset
    print("Creating test dataset...")
    test_dataset = AnomalySequenceDataset(
        parquet_path=test_parquet_path,
        all_features=all_features,
        all_feature_dims=all_feature_dims,
        seq_len=seq_len,
        device=device,
        is_training=False  # Evaluation mode
    )
    
    # Report dataset statistics
    print("\nTraining dataset:")
    train_dataset.report_statistics()
    
    print("\nTest dataset:")
    test_dataset.report_statistics()
    
    # Create data loaders
    print(f"\nUsing device: {device}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for training
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=(device == 'cuda')
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for evaluation
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=(device == 'cuda')
    )
    
    return train_loader, test_loader, None


if __name__ == "__main__":
    # Configuration
    config = {
        'train_parquet_path': 'unscaled_pdsch_val.parquet',
        'test_parquet_path': 'unscaled_pdsch_val_min.parquet',
        'all_features': ['SFN', 'Slot', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI'],
        'all_feature_dims': [1024, 31, 16, 33, 2, 9, 2],
        'seq_len': 20,
        'batch_size': 64,
        'num_workers': min(os.cpu_count(), 4),
        'sample_fraction': 1.0
    }
    
    # Create data loaders
    train_loader, test_loader, _ = create_data_loaders(
        train_parquet_path=config['train_parquet_path'],
        test_parquet_path=config['test_parquet_path'],
        all_features=config['all_features'],
        all_feature_dims=config['all_feature_dims'],
        seq_len=config['seq_len'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sample_fraction=config['sample_fraction']
    )
    
    print(f"\nCreated data loaders:")
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Display a sample batch
    for batch in train_loader:
        print("\nSample batch:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"{k}: type={type(v)}, length={len(v)}")
        break