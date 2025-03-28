import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import math
import pyarrow.parquet as pq
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, f1_score
import logging

# Setup logger
def setup_logger(log_file='lstm_ae_anomaly_detector.log'):
    logger = logging.getLogger('lstm_ae_detector')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create logger
logger = setup_logger()

# EmbeddingLayer - reused from your implementation to maintain consistency
class EmbeddingLayer(nn.Module):
    """
    Embedding layer for all features with learned positional encoding.
    """
    def __init__(self, feature_dims, embedding_dim=8, position_encoding=True):
        super(EmbeddingLayer, self).__init__()
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(dim, embedding_dim) for dim in feature_dims
        ])
        self.position_encoding = position_encoding
        self.embedding_dim = embedding_dim
        
    def forward(self, x, seq_len):
        # x shape: [batch_size, seq_len, num_features]
        batch_size, seq_len, num_features = x.size()
        
        # Apply embedding for each feature
        embeddings = []
        for i in range(num_features):
            feature_embedding = self.embedding_layers[i](x[:, :, i])  # [batch_size, seq_len, embedding_dim]
            embeddings.append(feature_embedding)
        
        # Concatenate all embeddings
        embeddings = torch.cat(embeddings, dim=2)  # [batch_size, seq_len, num_features * embedding_dim]
        
        # Add positional encoding if enabled
        if self.position_encoding:
            pos_enc = self.get_positional_encoding(seq_len, embeddings.size(2), x.device)
            embeddings = embeddings + pos_enc
        
        return embeddings
    
    def get_positional_encoding(self, seq_len, dim, device):
        pe = torch.zeros(seq_len, dim, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :(dim//2)]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
            
        return pe.unsqueeze(0)  # [1, seq_len, dim]

# LSTM Autoencoder for Anomaly Detection
class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for anomaly detection in time series data.
    Uses an encoder-decoder architecture to learn normal patterns and detect anomalies
    through reconstruction error.
    """
    def __init__(
        self,
        feature_dims,
        embedding_dim=8,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        position_encoding=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(LSTMAutoencoder, self).__init__()
        
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.device = device
        
        # Feature names for reference
        self.features = ['SFN', 'Slot', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI']
        
        # Embedding layer for all features
        self.feature_embedding = EmbeddingLayer(
            feature_dims, 
            embedding_dim=embedding_dim,
            position_encoding=position_encoding
        )
        
        # Combined input dimension after embedding all features
        self.combined_feature_dim = len(feature_dims) * embedding_dim
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=self.combined_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=hidden_dim,  # Input is the hidden representation from encoder
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Reconstruction layer to map decoder outputs back to embedding space
        self.reconstruction_layer = nn.Linear(hidden_dim, self.combined_feature_dim)
        
    def forward(self, feature_data):
        # feature_data: [batch_size, seq_len, num_features]
        batch_size, seq_len, _ = feature_data.size()
        
        # Embed all features
        embedded_features = self.feature_embedding(feature_data, seq_len)
        # embedded_features: [batch_size, seq_len, num_features * embedding_dim]
        
        # Encode the sequence
        encoder_outputs, (h_n, c_n) = self.encoder(embedded_features)
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        # h_n, c_n: [num_layers, batch_size, hidden_dim]
        
        # Use the final hidden state for each timestep as input to decoder
        # Create a tensor filled with the last encoder output
        decoder_input = encoder_outputs[:, -1:].repeat(1, seq_len, 1)
        # decoder_input: [batch_size, seq_len, hidden_dim]
        
        # Decode the sequence
        decoder_outputs, _ = self.decoder(decoder_input, (h_n, c_n))
        # decoder_outputs: [batch_size, seq_len, hidden_dim]
        
        # Reconstruct the embedded features
        reconstructed_features = self.reconstruction_layer(decoder_outputs)
        # reconstructed_features: [batch_size, seq_len, num_features * embedding_dim]
        
        # Calculate reconstruction error (MSE)
        reconstruction_error = torch.pow(embedded_features - reconstructed_features, 2)
        error_per_timestep = torch.mean(reconstruction_error, dim=2)  # Mean across feature dimension
        # error_per_timestep: [batch_size, seq_len]
        
        # Overall error per sequence
        overall_error = torch.mean(error_per_timestep, dim=1)  # Mean across time dimension
        # overall_error: [batch_size]
        
        return {
            'embedded_features': embedded_features,
            'encoder_outputs': encoder_outputs,
            'decoder_outputs': decoder_outputs,
            'reconstructed_features': reconstructed_features,
            'error_per_timestep': error_per_timestep,
            'overall_error': overall_error
        }
    
    def count_parameters(self):
        """Count the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Normal Sequence Dataset - only extracts normal sequences for training
class NormalSequenceDataset(Dataset):
    """
    Dataset for LSTM Autoencoder that creates sequences of normal events for training
    and all events for testing.
    """
    def __init__(
        self, 
        parquet_path,
        all_features=['SFN', 'Slot', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI'],
        all_feature_dims=[1024, 31, 16, 33, 2, 9, 2],
        seq_len=50,
        sample_fraction=1.0,
        training_mode=True,  # True for training (only normal), False for testing (all)
        overlap_ratio=0.5,   # For training sequences
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.parquet_path = parquet_path
        self.all_features = all_features
        self.all_feature_dims = all_feature_dims
        self.seq_len = seq_len
        self.device = device
        self.training_mode = training_mode
        self.overlap_ratio = overlap_ratio
        
        # Anomaly mapping
        self.ANOMALY_MAPPING = {
            "unnecessary_retx": 1,
            "missing_retx": 2,
            "new_data_no_retx": 3,
            "max_retx_achieved": 4,
        }
        
        # Get file IDs
        self.file_ids = self.get_file_ids(sample_fraction)
        
        # Prepare storage
        self.sequences = []
        self.labels = []
        self.timestamps = []
        
        # Count statistics
        self.normal_count = 0
        self.anomaly_counts = {i: 0 for i in range(1, 5)}
        
        # Create sequences
        self._create_sequences()
        
    def get_file_ids(self, sample_fraction):
        """Get file IDs to process, with optional sampling"""
        table = pq.read_table(self.parquet_path, columns=['file_id'])
        unique_file_ids = np.unique(table['file_id'].to_numpy())
        
        # Randomly sample if needed
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
        """Convert insights to numeric labels"""
        if isinstance(insights, np.ndarray):
            insights = insights.tolist()
        
        labels = []
        for insight in insights:
            if not insight or isinstance(insight, float):  # None, empty, or NaN
                labels.append(0)
                continue
                
            # Parse anomalies
            anomalies = [a.strip() for a in insight.split(",") if a.strip()]
            if not anomalies:
                labels.append(0)
                continue
                
            # Special case: remove max_retx_achieved if other anomalies exist
            if len(anomalies) > 1 and "max_retx_achieved" in anomalies:
                anomalies = [a for a in anomalies if a != "max_retx_achieved"]

            # Get valid anomaly labels
            valid_labels = [self.ANOMALY_MAPPING.get(a) for a in anomalies if a in self.ANOMALY_MAPPING]

            # Return 0 if no valid anomalies, else the smallest valid label
            labels.append(0 if not valid_labels else min(valid_labels))
                
        return torch.tensor(labels, dtype=torch.long)
    
    def _create_sequences(self):
        """Create sequences from files"""
        logger.info(f"Creating sequences from {len(self.file_ids)} files...")
        
        for file_id in tqdm(self.file_ids, desc="Processing files"):
            # Read data for this file
            table = pq.read_table(
                self.parquet_path, 
                filters=[('file_id', '==', file_id)]
            )
            
            # Convert to numpy arrays
            all_columns = table.column_names
            data_dict = {col: table[col].to_numpy() for col in all_columns}
            
            # Create feature tensor
            feature_data = torch.tensor(
                np.column_stack([data_dict[col] for col in self.all_features]), 
                dtype=torch.long
            )
            
            # Convert insights to labels
            labels = self.insight_to_label(data_dict['insight'])
            
            # Update statistics
            self.normal_count += (labels == 0).sum().item()
            for label in range(1, 5):
                self.anomaly_counts[label] += (labels == label).sum().item()
            
            # Process sequences based on mode
            if self.training_mode:
                self._process_normal_sequences(feature_data, labels, data_dict['timestamp_str'])
            else:
                self._process_test_sequences(feature_data, labels, data_dict['timestamp_str'])
    
    def _process_normal_sequences(self, feature_data, labels, timestamps):
        """Process sequences for training - only using normal data"""
        # Find continuous chunks of normal data
        normal_mask = labels == 0
        normal_indices = torch.nonzero(normal_mask, as_tuple=True)[0].numpy()
        
        if len(normal_indices) > 0:
            # Split into continuous chunks
            chunks = np.split(normal_indices, np.where(np.diff(normal_indices) != 1)[0] + 1)
            
            # Process chunks that are long enough
            for chunk in chunks:
                if len(chunk) >= self.seq_len:
                    # Calculate stride for overlapping windows
                    stride = max(1, int(self.seq_len * (1 - self.overlap_ratio)))
                    
                    # Create overlapping sequences
                    for start_idx in range(0, len(chunk) - self.seq_len + 1, stride):
                        # Extract indices for this sequence
                        seq_indices = chunk[start_idx:start_idx + self.seq_len]
                        
                        # Extract data
                        feat_seq = feature_data[seq_indices]
                        seq_labels = labels[seq_indices]
                        seq_timestamps = timestamps[seq_indices].tolist()
                        
                        # Store sequence
                        self.sequences.append(feat_seq)
                        self.labels.append(seq_labels)
                        self.timestamps.append(seq_timestamps)
    
    def _process_test_sequences(self, feature_data, labels, timestamps):
        """Process sequences for testing - using all data with non-overlapping windows"""
        total_length = len(feature_data)
        
        # Create non-overlapping sequences
        for start_idx in range(0, total_length, self.seq_len):
            end_idx = min(start_idx + self.seq_len, total_length)
            
            # Skip if sequence would be too short
            if end_idx - start_idx < self.seq_len:
                # Create a sequence with the remaining data
                start_idx = max(0, total_length - self.seq_len)
                end_idx = total_length
                
                # Skip if already processed
                if start_idx < (total_length - self.seq_len):
                    continue
            
            # Extract sequence
            feat_seq = feature_data[start_idx:end_idx]
            seq_labels = labels[start_idx:end_idx]
            seq_timestamps = timestamps[start_idx:end_idx].tolist()
            
            # Pad if needed
            if len(feat_seq) < self.seq_len:
                pad_length = self.seq_len - len(feat_seq)
                
                feat_padding = torch.zeros(pad_length, len(self.all_features), dtype=torch.long)
                label_padding = torch.zeros(pad_length, dtype=torch.long)
                timestamp_padding = [""] * pad_length
                
                # Pad at the end for testing
                feat_seq = torch.cat([feat_seq, feat_padding], dim=0)
                seq_labels = torch.cat([seq_labels, label_padding], dim=0)
                seq_timestamps = seq_timestamps + timestamp_padding
            
            # Store sequence
            self.sequences.append(feat_seq)
            self.labels.append(seq_labels)
            self.timestamps.append(seq_timestamps)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'feature_data': self.sequences[idx],  # [seq_len, num_features]
            'label': self.labels[idx],            # [seq_len]
            'timestamp': self.timestamps[idx]     # List of length seq_len
        }
    
    def report_statistics(self):
        """Report dataset statistics"""
        logger.info(f"\nDataset Statistics:")
        logger.info(f"Total sequences: {len(self.sequences)}")
        logger.info(f"Mode: {'Training (Normal only)' if self.training_mode else 'Testing (All)'}")
        
        # Count instances by label
        zero_label_count = sum((seq_labels == 0).sum().item() for seq_labels in self.labels)
        one_label_count = sum((seq_labels == 1).sum().item() for seq_labels in self.labels)
        two_label_count = sum((seq_labels == 2).sum().item() for seq_labels in self.labels)
        three_label_count = sum((seq_labels == 3).sum().item() for seq_labels in self.labels)
        four_label_count = sum((seq_labels == 4).sum().item() for seq_labels in self.labels)
        
        total_elements = sum(len(seq_labels) for seq_labels in self.labels)
        
        logger.info(f"Label distribution in sequences:")
        logger.info(f"  Normal (0): {zero_label_count} ({zero_label_count/total_elements:.2%})")
        logger.info(f"  UReTx (1): {one_label_count} ({one_label_count/total_elements:.2%})")
        logger.info(f"  MReTx (2): {two_label_count} ({two_label_count/total_elements:.2%})")
        logger.info(f"  NoDReTx (3): {three_label_count} ({three_label_count/total_elements:.2%})")
        logger.info(f"  MaxReTx (4): {four_label_count} ({four_label_count/total_elements:.2%})")

# Custom collate function
def custom_collate_fn(batch):
    """Custom collate function for DataLoader"""
    feature_data = torch.stack([item['feature_data'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    timestamps = [item['timestamp'] for item in batch]
    
    return {
        'feature_data': feature_data,
        'label': labels,
        'timestamp': timestamps
    }

# Function to create data loaders
def create_data_loaders(
    train_parquet_path,
    test_parquet_path,
    all_features,
    all_feature_dims,
    seq_len=100,
    batch_size=32,
    num_workers=4,
    sample_fraction=1.0
):
    """Create data loaders for training and testing"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create training dataset (normal only)
    logger.info("Creating training dataset (normal sequences only)...")
    train_dataset = NormalSequenceDataset(
        parquet_path=train_parquet_path,
        all_features=all_features,
        all_feature_dims=all_feature_dims,
        seq_len=seq_len,
        sample_fraction=sample_fraction,
        training_mode=True,
        device=device
    )
    
    # Create validation split
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create test dataset (all sequences)
    logger.info("Creating test dataset (all sequences)...")
    test_dataset = NormalSequenceDataset(
        parquet_path=test_parquet_path,
        all_features=all_features,
        all_feature_dims=all_feature_dims,
        seq_len=seq_len,
        training_mode=False,
        device=device
    )
    
    # Report statistics
    train_dataset.report_statistics()
    test_dataset.report_statistics()
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=(device == 'cuda')
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=(device == 'cuda')
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=(device == 'cuda')
    )
    
    logger.info(f"Created data loaders - Training: {len(train_loader)} batches, "
               f"Validation: {len(val_loader)} batches, Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

# Function to train the LSTM Autoencoder
def train_lstm_autoencoder(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs=50,
    device='cuda',
    scheduler=None,
    early_stopping_patience=10
):
    """Train the LSTM Autoencoder with early stopping"""
    # Initialize tracking variables
    best_val_loss = float('inf')
    early_stopping_counter = 0
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        
        for batch in train_progress:
            feature_data = batch['feature_data'].to(device)
            
            # Forward pass
            outputs = model(feature_data)
            
            # Calculate loss (MSE reconstruction error)
            loss = torch.mean(outputs['overall_error'])
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            
            # Update progress bar
            train_progress.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
        
        with torch.no_grad():
            for batch in val_progress:
                feature_data = batch['feature_data'].to(device)
                
                # Forward pass
                outputs = model(feature_data)
                
                # Calculate loss
                loss = torch.mean(outputs['overall_error'])
                
                # Track loss
                val_loss += loss.item()
                
                # Update progress bar
                val_progress.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        
        # Update learning rate scheduler if provided
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        
        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_lstm_autoencoder.pth')
            logger.info(f"New best model saved with validation loss: {avg_val_loss:.4f}")
        else:
            early_stopping_counter += 1
            logger.info(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
            
            if early_stopping_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load the best model
    model.load_state_dict(torch.load('best_lstm_autoencoder.pth'))
    
    return model, train_losses, val_losses

# Function to evaluate the LSTM Autoencoder
def evaluate_lstm_autoencoder(
    model,
    test_loader,
    device='cuda',
    threshold=None
):
    """
    Evaluate the LSTM Autoencoder on test data.
    If threshold is None, determine the optimal threshold using ROC curve.
    """
    model.eval()
    
    # Initialize lists to store data
    all_errors = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            feature_data = batch['feature_data'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(feature_data)
            
            # Store reconstruction errors and labels
            all_errors.append(outputs['error_per_timestep'].cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate and flatten for analysis
    all_errors = np.concatenate(all_errors, axis=0)  # [num_samples, seq_len]
    all_labels = np.concatenate(all_labels, axis=0)  # [num_samples, seq_len]
    
    all_errors_flat = all_errors.flatten()
    all_labels_flat = all_labels.flatten()
    all_binary_labels_flat = (all_labels_flat > 0).astype(int)
    
    # Find optimal threshold if not provided
    if threshold is None:
        precision, recall, thresholds = precision_recall_curve(
            all_binary_labels_flat, all_errors_flat
        )
        
        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Find threshold that maximizes F1 score
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx - 1] if best_idx > 0 else 0
        threshold = best_threshold
        
        logger.info(f"Optimal threshold: {threshold:.6f} (F1: {f1_scores[best_idx]:.4f})")
    
    # Get predictions using threshold
    binary_preds_flat = (all_errors_flat >= threshold).astype(int)
    
    # Calculate metrics
    binary_accuracy = np.mean(binary_preds_flat == all_binary_labels_flat)
    binary_f1 = f1_score(all_binary_labels_flat, binary_preds_flat)
    binary_cm = confusion_matrix(all_binary_labels_flat, binary_preds_flat)
    binary_auc = roc_auc_score(all_binary_labels_flat, all_errors_flat)
    
    # Log results
    logger.info("\nEvaluation Results:")
    logger.info(f"Threshold: {threshold:.6f}")
    logger.info(f"Binary Classification - Accuracy: {binary_accuracy:.4f}, "
               f"F1: {binary_f1:.4f}, AUC: {binary_auc:.4f}")
    
    logger.info("\nBinary Confusion Matrix:")
    logger.info(f"TN: {binary_cm[0, 0]}, FP: {binary_cm[0, 1]}")
    logger.info(f"FN: {binary_cm[1, 0]}, TP: {binary_cm[1, 1]}")
    
    # Report detection rate by class
    class_names = ['Normal', 'Unnecessary Retx', 'Missing Retx', 'New Data No Retx', 'Max Retx Achieved']
    logger.info("\nDetection Rate by Class:")
    
    class_detection_rates = {}
    for i in range(5):
        class_mask = all_labels_flat == i
        class_total = np.sum(class_mask)
        
        if class_total > 0:
            if i == 0:  # Normal class
                class_correct = np.sum((binary_preds_flat == 0) & class_mask)
            else:  # Anomaly classes
                class_correct = np.sum((binary_preds_flat == 1) & class_mask)
                
            detection_rate = class_correct / class_total
            class_detection_rates[i] = detection_rate
            logger.info(f"Class {i} ({class_names[i]}): {int(class_correct)}/{int(class_total)} = "
                       f"{detection_rate:.2%}")
        else:
            class_detection_rates[i] = 0.0
            logger.info(f"Class {i} ({class_names[i]}): 0/0 = 0.00%")
    
    return {
        'threshold': threshold,
        'binary_accuracy': binary_accuracy,
        'binary_f1': binary_f1,
        'binary_auc': binary_auc,
        'binary_cm': binary_cm,
        'class_detection_rates': class_detection_rates
    }

# Main function to run the LSTM Autoencoder baseline
def main():
    # Configuration (similar to your original model)
    config = {
        # Data paths
        'train_parquet_path': 'unscaled_pdsch_val.parquet',
        'test_parquet_path': 'unscaled_pdsch_val_min.parquet',
        
        # Feature configuration
        'all_features': ['SFN', 'Slot', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI'],
        'all_feature_dims': [1024, 31, 16, 33, 2, 9, 2],
        
        # Dataset parameters
        'seq_len': 100,  # Match your model
        'batch_size': 32,
        'num_workers': 0 if torch.cuda.is_available() else min(os.cpu_count(), 4),
        'sample_fraction': 1.0,
        
        # Embedding layer
        'embedding_dim': 14,  # Match your model
        'position_encoding': True,
        
        # LSTM parameters (adjusted to match your GRU size)
        'hidden_dim': 410,  # Same as your GRU
        'num_layers': 2,
        'dropout': 0.25,
        
        # Training parameters
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'num_epochs': 50,
        'early_stopping_patience': 10,
        
        # Random seed
        'seed': 42,
    }
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_parquet_path=config['train_parquet_path'],
        test_parquet_path=config['test_parquet_path'],
        all_features=config['all_features'],
        all_feature_dims=config['all_feature_dims'],
        seq_len=config['seq_len'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sample_fraction=config['sample_fraction']
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = LSTMAutoencoder(
        feature_dims=config['all_feature_dims'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        position_encoding=config['position_encoding'],
        device=device
    ).to(device)
    
    # Print model size
    total_params = model.count_parameters()
    logger.info(f"Model size: {total_params:,} parameters ({total_params/1000:.2f}K)")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Train the model
    logger.info("Starting LSTM Autoencoder training...")
    model, train_losses, val_losses = train_lstm_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=config['num_epochs'],
        device=device,
        scheduler=scheduler,
        early_stopping_patience=config['early_stopping_patience']
    )
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('LSTM Autoencoder Training and Validation Loss')
    plt.legend()
    plt.savefig('lstm_ae_loss_curve.png')
    
    # Evaluate model (find optimal threshold)
    logger.info("Evaluating LSTM Autoencoder...")
    evaluation_results = evaluate_lstm_autoencoder(
        model=model,
        test_loader=test_loader,
        device=device
    )
    
    # Save model and results
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'evaluation_results': evaluation_results
    }, 'lstm_autoencoder_model.pth')
    
    logger.info("LSTM Autoencoder baseline complete!")
    return evaluation_results

if __name__ == "__main__":
    main()