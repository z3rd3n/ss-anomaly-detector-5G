# dataset.py
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

EPS = 1e-8  # small constant

def custom_collate_fn(batch):
    """
    Merges a list of samples into a batch.
    """
    try:
        features = torch.stack([item['features'] for item in batch])
        timestamps = [item['timestamps'] for item in batch]
        labels = torch.stack([item['labels'] for item in batch]) if 'labels' in batch[0] else None
        out = {'features': features, 'timestamps': timestamps}
        if labels is not None:
            out['labels'] = labels
        return out
    except Exception as e:
        logging.error(f"Error in collate_fn: {e}")
        return {}

class ParquetSequenceDataset(Dataset):
    def __init__(self,
                 parquet_path: str,
                 feature_columns: list,
                 seq_len: int,
                 stride: int = None,
                 ratio: float = 1.0,
                 seed: int = 42,
                 skip_anomalies: bool = False,
                 normalization_stats: dict = None,
                 pre_normalize: bool = True):
        """
        Loads sequences from a parquet file.
        
        Args:
            parquet_path: Path to the parquet file
            feature_columns: List of column names to use as features
            seq_len: Length of each sequence
            stride: Stride between sequences (default: equal to seq_len)
            ratio: Portion of data to use (1.0 = all data)
            seed: Random seed for reproducibility
            skip_anomalies: If True, skip sequences containing anomalies
            normalization_stats: Dictionary with 'means' and 'variances' for normalization
            pre_normalize: If True, normalize features at load time
        """
        super().__init__()
        self.parquet_path = parquet_path
        self.feature_columns = feature_columns
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.skip_anomalies = skip_anomalies
        self.normalization_stats = normalization_stats
        self.pre_normalize = pre_normalize

        # Define mapping for anomaly labels
        self.anomaly_mapping = {
            "unnecessary_retx": 1,
            "missing_retx": 2,
            "new_data_no_retx": 3,
            "max_retx_achieved": 4,
        }

        # Load data and prepare sequences
        self._load_data(ratio, seed)
        self._prepare_sequences()
        self._report_class_counts()

    def _load_data(self, ratio, seed):
        """Load data from the parquet file"""
        logging.info(f"Loading data from {self.parquet_path}")
        
        # Sample files if ratio < 1.0
        if ratio < 1.0:
            df_ids = pd.read_parquet(self.parquet_path, columns=['file_id'])
            unique_file_ids = df_ids['file_id'].unique()
            
            np.random.seed(seed)
            num_files = max(1, int(len(unique_file_ids) * ratio))
            sampled_files = np.random.choice(unique_file_ids, num_files, replace=False)
            
            # Load only sampled files
            self.df = pd.read_parquet(
                self.parquet_path,
                filters=[('file_id', 'in', sampled_files.tolist())]
            )
        else:
            # Load all data
            self.df = pd.read_parquet(self.parquet_path)
        
        # Ensure required columns exist
        if 'insight' not in self.df.columns:
            self.df['insight'] = ""
        
        if 'timestamp_str' not in self.df.columns:
            self.df['timestamp_str'] = ["" for _ in range(len(self.df))]
        
        # Extract and potentially normalize features
        features = self.df[self.feature_columns].values
        
        # Normalize features if requested
        if self.normalization_stats and self.pre_normalize:
            means = self.normalization_stats.get('means')
            variances = self.normalization_stats.get('variances')
            std = np.sqrt(np.array(variances) + EPS)
            features = (features - np.array(means)) / std
        
        # Convert to torch tensor
        self.features = torch.tensor(features, dtype=torch.float32)
        
        # Process labels
        self.labels = torch.tensor([
            self.insight_to_label(insight) for insight in self.df['insight']
        ], dtype=torch.long)
        
        # Store timestamps
        self.timestamps = self.df['timestamp_str'].tolist()
        
        logging.info(f"Loaded {len(self.df)} rows from the dataset")

    def _prepare_sequences(self):
        """Pre-compute all valid sequence indices"""
        self.sequences = []
        
        # Group by file_id to ensure sequences come from the same file
        for _, group_indices in self.df.groupby('file_id').groups.items():
            indices = list(group_indices)
            num_rows = len(indices)
            
            if num_rows < self.seq_len:
                continue
            
            # Generate valid sequence indices
            for start_idx in range(0, num_rows - self.seq_len + 1, self.stride):
                seq_indices = indices[start_idx:start_idx + self.seq_len]
                
                # Skip sequences with anomalies if requested
                if self.skip_anomalies:
                    seq_labels = self.labels[seq_indices]
                    if (seq_labels != 0).any():
                        continue
                        
                self.sequences.append(seq_indices)
                
        logging.info(f"Generated {len(self.sequences)} sequences")

    def insight_to_label(self, insight):
        """Convert insight string to integer label"""
        if pd.isna(insight) or insight.strip() == "":
            return 0

        # Parse anomalies from the insight string
        anomalies = [a.strip() for a in insight.split(",") if a.strip()]
        if not anomalies:
            return 0

        # Special case: remove max_retx_achieved if other anomalies exist
        if len(anomalies) > 1 and "max_retx_achieved" in anomalies:
            anomalies = [a for a in anomalies if a != "max_retx_achieved"]
        
        # Get valid anomaly labels
        valid_labels = [self.anomaly_mapping.get(a) for a in anomalies 
                        if a in self.anomaly_mapping]
        
        # Return 0 if no valid anomalies
        if not valid_labels:
            return 0
            
        # Return the smallest valid label
        return min(valid_labels)
    
    def _report_class_counts(self):
        """Report class distribution in the dataset"""
        unique_labels, counts = torch.unique(self.labels, return_counts=True)
        
        logging.info("Class distribution in the dataset:")
        reverse_mapping = {v: k for k, v in self.anomaly_mapping.items()}
        reverse_mapping[0] = "normal"
        
        # Store counts for future reference
        self.counts = {label.item(): count.item() 
                      for label, count in zip(unique_labels, counts)}
        
        # Log the counts
        for label, count in self.counts.items():
            label_name = reverse_mapping.get(label, "unknown")
            logging.info(f"  {label} ({label_name}): {count}")

    def _normalize(self, features):
        """Normalize features on-the-fly if not pre-normalized"""
        if not self.normalization_stats or self.pre_normalize:
            return features

        means = self.normalization_stats.get('means')
        variances = self.normalization_stats.get('variances')
        
        if not torch.is_tensor(means):
            means = torch.tensor(means, dtype=torch.float32, device=features.device)
        if not torch.is_tensor(variances):
            variances = torch.tensor(variances, dtype=torch.float32, device=features.device)

        std = torch.sqrt(variances + EPS)
        return (features.float() - means) / std

    def __getitem__(self, index):
        """Get a sequence by index"""
        seq_indices = self.sequences[index]
        
        # Get features, labels, and timestamps for this sequence
        seq_features = self.features[seq_indices]
        seq_labels = self.labels[seq_indices]
        seq_timestamps = [self.timestamps[i] for i in seq_indices]
        
        # Normalize if needed and not already done
        if self.normalization_stats and not self.pre_normalize:
            seq_features = self._normalize(seq_features)
        
        return {
            'features': seq_features,
            'timestamps': seq_timestamps,
            'labels': seq_labels
        }

    def __len__(self):
        """Return the number of sequences"""
        return len(self.sequences)