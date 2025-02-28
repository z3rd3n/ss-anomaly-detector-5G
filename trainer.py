import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, f1_score
import os
import json

class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism to focus on important timesteps.
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_dim]
        scores = self.attention(x)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(scores, dim=1)  # [batch_size, seq_len, 1]
        context_vector = torch.sum(x * attention_weights, dim=1)  # [batch_size, hidden_dim]
        return context_vector, attention_weights

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

# Binary classifier using sequential processing with attention
class AttentionBinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer to incorporate reconstruction error per timestep
        self.error_integration = nn.Linear(input_dim + 1, input_dim)  # +1 for error
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # Output layer
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x, error_per_timestep):
        # Apply self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)  # Residual connection
        x = self.dropout1(x)
        
        # Incorporate error per timestep
        error_expanded = error_per_timestep.unsqueeze(-1)  # [batch_size, seq_len, 1]
        combined = torch.cat([x, error_expanded], dim=-1)  # [batch_size, seq_len, input_dim+1]
        
        # Process combined features
        x = self.error_integration(combined)
        x = self.layer_norm2(x)
        x = self.dropout2(x)
        
        # Final binary prediction
        return self.linear(x)  # [batch_size, seq_len, 1]

class BiGRUAnomalyDetector(nn.Module):
    """
    BiGRU model for anomaly detection with embedding for all features
    """
    def __init__(
        self,
        feature_dims,
        hidden_dim=128,
        embedding_dim=8,
        num_layers=2,
        dropout=0.2,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(BiGRUAnomalyDetector, self).__init__()
        
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Define features for reference
        self.features = ['SFN', 'Slot', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI']
        
        # Embedding layer for all features
        self.feature_embedding = EmbeddingLayer(
            feature_dims, 
            embedding_dim=embedding_dim
        )
        
        # Combined input dimension after embedding all features
        self.combined_feature_dim = len(feature_dims) * embedding_dim
        
        # Bidirectional GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=self.combined_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output dimension from bidirectional GRU
        self.gru_output_dim = hidden_dim * 2  # bidirectional
        
        # Temporal attention for sequence-level context
        self.temporal_attention = TemporalAttention(self.gru_output_dim)
        
        # Reconstruction layer for embedded features (for the error loss)
        self.reconstruction_layer = nn.Linear(self.gru_output_dim, self.combined_feature_dim)
        
        # Binary Classification head using attention
        self.binary_classifier = AttentionBinaryClassifier(
            input_dim=self.gru_output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
    def forward(self, feature_data):
        # feature_data: [batch_size, seq_len, num_features]
        batch_size, seq_len, _ = feature_data.size()
        
        # Embed all features
        embedded_features = self.feature_embedding(feature_data, seq_len)
        
        # Process with bidirectional GRU
        gru_output, _ = self.gru(embedded_features)
        # gru_output: [batch_size, seq_len, hidden_dim * 2]
        
        # Apply temporal attention for sequence-level context
        context_vector, temporal_attn_weights = self.temporal_attention(gru_output)
        
        # Reconstruction of embedded features for each timestep
        feature_reconstruction = self.reconstruction_layer(gru_output)
        
        # Calculate reconstruction error for each timestep
        reconstruction_error = torch.pow(embedded_features - feature_reconstruction, 2)
        error_per_timestep = torch.mean(reconstruction_error, dim=2)  # [B, S]
        overall_error = torch.mean(error_per_timestep, dim=1)  # [B]
        
        # Binary classification for each timestep
        binary_logits = self.binary_classifier(gru_output, error_per_timestep)
        binary_probs = torch.sigmoid(binary_logits)
        
        # Calculate instance-level attention for anomaly localization
        instance_attn_scores = binary_logits.clone()
        instance_attn_weights = F.softmax(instance_attn_scores.squeeze(-1), dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        return {
            'gru_output': gru_output,
            'feature_reconstruction': feature_reconstruction,
            'error_per_timestep': error_per_timestep,
            'overall_error': overall_error,
            'binary_logits': binary_logits,
            'binary_probs': binary_probs,
            'temporal_attn_weights': temporal_attn_weights,
            'instance_attn_weights': instance_attn_weights,
            'context_vector': context_vector
        }
    
    def count_parameters(self):
        """Count the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class AnomalyTypeLoss(nn.Module):
    """
    Combined loss function for anomaly detection with simplified type regularization.
    
    L_total = λ_rec · L_rec + λ_bin · L_bin + λ_reg · L_reg
    
    where L_reg is simplified to use anomaly counts from the dataset.
    """
    def __init__(
        self, 
        binary_weight=1.0,
        reconstruction_weight=1.0,
        regularization_weight=0.5,
        anomaly_counts=None,
        num_batches=None
    ):
        super(AnomalyTypeLoss, self).__init__()
        self.binary_weight = binary_weight
        self.reconstruction_weight = reconstruction_weight
        self.regularization_weight = regularization_weight
        
        # Store anomaly counts and number of batches for regularization
        self.anomaly_counts = anomaly_counts
        self.num_batches = num_batches
        
        # Create binary focal loss
        self.binary_focal_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, model_output, labels):
        # Extract components from model output
        binary_logits = model_output['binary_logits']  # [batch_size, seq_len, 1]
        error_per_timestep = model_output['error_per_timestep']  # [batch_size, seq_len]
        
        # Create binary labels from multiclass labels (labels > 0 means anomaly)
        binary_labels = (labels > 0).float()
        
        # 1. Reconstruction Loss
        reconstruction_loss = torch.mean(error_per_timestep)
        
        # 2. Binary Classification Loss
        binary_logits_flat = binary_logits.reshape(-1, 1)  # [batch_size * seq_len, 1]
        binary_labels_flat = binary_labels.reshape(-1, 1)  # [batch_size * seq_len, 1]
        binary_loss = self.binary_focal_loss(binary_logits_flat, binary_labels_flat)
        
        # 3. Simplified Anomaly Type Regularization
        regularization_loss = self.compute_simplified_anomaly_type_regularization(
            model_output, 
            labels
        )
        
        # Combine all losses with their respective weights
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.binary_weight * binary_loss +
            self.regularization_weight * regularization_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'binary_loss': binary_loss,
            'regularization_loss': regularization_loss
        }
    
    def compute_simplified_anomaly_type_regularization(self, model_output, labels):
        """
        Compute simplified regularization term using anomaly counts.
        Formula: sum_over_types[ (detected in batch) / (expected per batch) ]
        Expected per batch = total type count / number of batches
        """
        # If anomaly counts or num_batches not provided, return zero loss
        if self.anomaly_counts is None or self.num_batches is None:
            return torch.tensor(0.0, device=labels.device)
        
        # Get binary predictions
        binary_probs = model_output['binary_probs']  # [batch_size, seq_len, 1]
        binary_preds = (binary_probs >= 0.5).float().squeeze(-1)  # [batch_size, seq_len]
        
        # Initialize regularization loss
        regularization_loss = torch.tensor(0.0, device=labels.device)
        
        # Process each anomaly type separately (1 through 4)
        anomaly_types = [1, 2, 3, 4]
        
        for anomaly_type in anomaly_types:
            # Skip if no instances of this type in the dataset
            if anomaly_type not in self.anomaly_counts or self.anomaly_counts[anomaly_type] == 0:
                continue
            
            # Expected count per batch for this type
            expected_count = self.anomaly_counts[anomaly_type] / self.num_batches
            
            # Find positions with this anomaly type
            type_mask = (labels == anomaly_type).float()  # [batch_size, seq_len]
            
            # Calculate detected count in this batch
            detected_mask = type_mask * binary_preds  # Only count correct detections
            detected_count = torch.sum(detected_mask)
            
            # Add component to regularization loss
            # We want to maximize detected_count/expected_count, so we minimize -log(detected/expected)
            if detected_count > 0:
                type_loss = -torch.log((detected_count + 1e-6) / expected_count)

                # Apply additional weights to anomaly types 2 and 3
                if anomaly_type in [2, 3]:
                    type_loss *= 2.0  # Double the weight for types 2 and 3
                regularization_loss += type_loss
            
        return regularization_loss

def train_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler=None,
    num_epochs=50,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    early_stopping_patience=10,
    binary_weight=1.0,
    reconstruction_weight=1.0,
    regularization_weight=0.5
):
    """
    Train the model with early stopping based on F1 score and log false positives
    """
    # Get anomaly counts from the dataset
    anomaly_counts = train_loader.dataset.anomaly_counts
    num_batches = len(train_loader)
    
    # Create loss function with anomaly counts
    criterion = AnomalyTypeLoss(
        binary_weight=binary_weight,
        reconstruction_weight=reconstruction_weight,
        regularization_weight=regularization_weight,
        anomaly_counts=anomaly_counts,
        num_batches=num_batches
    )
    
    # Initialize tracking variables
    best_val_f1 = 0.0  # Changed from best_val_loss to best_val_f1
    early_stopping_counter = 0
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_rec_loss = 0.0
        train_binary_loss = 0.0
        train_reg_loss = 0.0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        
        for batch in train_progress:
            feature_data = batch['feature_data'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(feature_data)
            
            # Calculate loss
            loss_dict = criterion(outputs, labels)
            total_loss = loss_dict['total_loss']
            
            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track losses
            train_loss += total_loss.item()
            train_rec_loss += loss_dict['reconstruction_loss'].item()
            train_binary_loss += loss_dict['binary_loss'].item()
            train_reg_loss += loss_dict['regularization_loss'].item()
            
            # Update progress bar with current losses
            train_progress.set_postfix({
                'loss': f"{total_loss.item():.4f}", 
                'rec': f"{loss_dict['reconstruction_loss'].item():.4f}",
                'bin': f"{loss_dict['binary_loss'].item():.4f}",
                'reg': f"{loss_dict['regularization_loss'].item():.4f}"
            })
        
        # Calculate average training losses
        num_batches = len(train_loader)
        train_loss /= num_batches
        train_rec_loss /= num_batches
        train_binary_loss /= num_batches
        train_reg_loss /= num_batches
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_rec_loss = 0.0
        val_binary_loss = 0.0
        val_reg_loss = 0.0
        
        all_binary_preds = []
        all_labels = []
        all_binary_labels = []
        all_binary_probs = []  # Added to store probabilities
        all_timestamps = []    # Added to store timestamps
        all_feature_data = []  # Added to store feature data
        
        val_progress = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
        
        with torch.no_grad():
            for batch in val_progress:
                feature_data = batch['feature_data'].to(device)
                labels = batch['label'].to(device)
                timestamps = batch['timestamp']  # Get timestamps
                
                # Forward pass
                outputs = model(feature_data)
                
                # Calculate loss
                loss_dict = criterion(outputs, labels)
                total_loss = loss_dict['total_loss']
                
                # Get predictions for each timestep
                binary_probs = outputs['binary_probs'].squeeze(-1)  # [batch_size, seq_len]
                binary_preds = (binary_probs >= 0.5).float()
                
                # Create binary labels
                binary_labels = (labels > 0).float()
                
                # Collect predictions and labels for metrics
                all_binary_preds.append(binary_preds.cpu())
                all_labels.append(labels.cpu())
                all_binary_labels.append(binary_labels.cpu())
                all_binary_probs.append(binary_probs.cpu())  # Store probabilities
                all_timestamps.append(timestamps)  # Store timestamps
                all_feature_data.append(feature_data.cpu())  # Store feature data
                
                # Track losses
                val_loss += total_loss.item()
                val_rec_loss += loss_dict['reconstruction_loss'].item()
                val_binary_loss += loss_dict['binary_loss'].item()
                val_reg_loss += loss_dict['regularization_loss'].item()
                
                # Update progress bar
                val_progress.set_postfix({
                    'loss': f"{total_loss.item():.4f}", 
                    'rec': f"{loss_dict['reconstruction_loss'].item():.4f}",
                    'bin': f"{loss_dict['binary_loss'].item():.4f}",
                    'reg': f"{loss_dict['regularization_loss'].item():.4f}"
                })
        
        # Calculate average validation losses
        num_val_batches = len(test_loader)
        val_loss /= num_val_batches
        val_rec_loss /= num_val_batches
        val_binary_loss /= num_val_batches
        val_reg_loss /= num_val_batches
        val_losses.append(val_loss)
        
        # Concatenate all predictions and labels
        all_binary_preds = torch.cat(all_binary_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_binary_labels = torch.cat(all_binary_labels, dim=0).numpy()
        all_binary_probs = torch.cat(all_binary_probs, dim=0).numpy()
        all_feature_data = torch.cat(all_feature_data, dim=0).numpy()
        
        # Flatten for metrics calculation
        all_binary_preds_flat = all_binary_preds.reshape(-1)
        all_labels_flat = all_labels.reshape(-1)
        all_binary_labels_flat = all_binary_labels.reshape(-1)
        all_binary_probs_flat = all_binary_probs.reshape(-1)
        
        # Calculate binary metrics
        binary_acc = np.mean(all_binary_preds_flat == all_binary_labels_flat)
        binary_f1 = f1_score(all_binary_labels_flat, all_binary_preds_flat)
        
        # Display results
        print(f"\nValidation Results (Epoch {epoch+1}):")
        print(f"Overall Loss: {val_loss:.4f} (Rec: {val_rec_loss:.4f}, "
              f"Bin: {val_binary_loss:.4f}, Reg: {val_reg_loss:.4f})")
        print(f"Binary Detection - Accuracy: {binary_acc:.4f}, F1: {binary_f1:.4f}")
        
        # Print confusion matrices
        binary_cm = confusion_matrix(all_binary_labels_flat, all_binary_preds_flat)
        print("\nBinary Confusion Matrix (Normal vs Anomaly):")
        print("                  Predicted")
        print("                Normal  Anomaly")
        print(f"Actual Normal   {binary_cm[0][0]:<8} {binary_cm[0][1]:<8}")
        print(f"Actual Anomaly  {binary_cm[1][0]:<8} {binary_cm[1][1]:<8}")
        
        # Log false positives
        log_false_positives(all_binary_preds, all_binary_labels, all_binary_probs, 
                           all_feature_data, all_timestamps, model.features, epoch)
        
        # Calculate detection rate by class
        class_names = ['Norm', 'UReTx', 'MReTx', 'NoDReTx', 'MaxReTx']
        print("\nDetection Rate by Class:")
        for i in range(5):
            # Count total instances of this class
            class_total = np.sum(all_labels_flat == i)
            
            if class_total > 0:
                # Count true anomaly detections for this class
                if i == 0:  # Normal
                    class_correct = np.sum((all_binary_preds_flat == 0) & (all_labels_flat == 0))
                else:  # Anomaly classes
                    class_correct = np.sum((all_binary_preds_flat == 1) & (all_labels_flat == i))
                
                detection_rate = class_correct / class_total
                print(f"Class {i} ({class_names[i]}): {int(class_correct)}/{int(class_total)} = {detection_rate:.2%}")
            else:
                print(f"Class {i} ({class_names[i]}): 0/0 = 0.00%")
        
        # Update learning rate scheduler if provided
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Check for early stopping based on F1 score
        if binary_f1 > best_val_f1:  # Changed to compare F1 scores instead of loss
            best_val_f1 = binary_f1
            early_stopping_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with validation F1: {binary_f1:.4f}")
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
            
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    return model, train_losses, val_losses

def log_false_positives(binary_preds, binary_labels, binary_probs, feature_data, 
                       timestamps_list, feature_names, epoch):
    """
    Log all false positives with their features and probabilities
    """
    # Create flat list of all timestamps
    all_timestamps = []
    for batch_timestamps in timestamps_list:
        all_timestamps.extend([ts for sublist in batch_timestamps for ts in sublist])
    
    # Initialize lists to store false positives
    fp_timestamps = []
    fp_probs = []
    fp_features = []
    
    # Find all false positives (predicted as anomaly but actually normal)
    for batch_idx in range(binary_preds.shape[0]):
        for seq_idx in range(binary_preds.shape[1]):
            # Check if this is a false positive
            if binary_preds[batch_idx, seq_idx] == 1 and binary_labels[batch_idx, seq_idx] == 0:
                # Calculate flattened index to get the timestamp
                flat_idx = batch_idx * binary_preds.shape[1] + seq_idx
                
                # Skip if the timestamp is empty (could be padding)
                if flat_idx >= len(all_timestamps) or not all_timestamps[flat_idx]:
                    continue
                
                # Add to lists
                fp_timestamps.append(all_timestamps[flat_idx])
                fp_probs.append(binary_probs[batch_idx, seq_idx])
                
                # Get the feature values
                features = feature_data[batch_idx, seq_idx].tolist()
                
                fp_features.append(features)
    
    # Create a dictionary to store unique false positives
    unique_fps = {}
    for ts, prob, feat in zip(fp_timestamps, fp_probs, fp_features):
        if ts not in unique_fps or prob > unique_fps[ts][0]:
            unique_fps[ts] = (prob, feat)
    
    # Sort by probability in descending order
    sorted_fps = sorted(unique_fps.items(), key=lambda x: x[1][0], reverse=True)
    
    # Write to file
    with open(f'false_positives_epoch_{epoch+1}.csv', 'w') as f:
        f.write('timestamp,probability,' + ','.join(feature_names) + '\n')
        for ts, (prob, feat) in sorted_fps:
            f.write(f'{ts},{prob:.6f},' + ','.join(map(str, feat)) + '\n')
    
    print(f"\nLogged {len(sorted_fps)} unique false positives to false_positives_epoch_{epoch+1}.csv")

def evaluate_model(
    model,
    test_loader,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Evaluate the model on test data and calculate metrics
    """
    model.eval()
    
    # Initialize lists to store predictions and true labels
    all_binary_preds = []
    all_labels = []
    all_binary_probs = []
    
    # Track attention weights and reconstruction errors
    all_temporal_attn = []
    all_instance_attn = []
    all_errors = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            feature_data = batch['feature_data'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(feature_data)
            
            # Get binary predictions
            binary_probs = outputs['binary_probs'].squeeze(-1)  # [batch_size, seq_len]
            binary_preds = (binary_probs >= 0.5).float()
            
            # Store predictions and true labels
            all_binary_preds.append(binary_preds.cpu())
            all_labels.append(labels.cpu())
            all_binary_probs.append(binary_probs.cpu())
            
            # Store attention weights and reconstruction errors
            all_temporal_attn.append(outputs['temporal_attn_weights'].cpu())
            all_instance_attn.append(outputs['instance_attn_weights'].cpu())
            all_errors.append(outputs['error_per_timestep'].cpu())
    
    # Concatenate all predictions and labels
    all_binary_preds = torch.cat(all_binary_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_binary_probs = torch.cat(all_binary_probs, dim=0).numpy()
    
    # Flatten for metrics calculation
    all_binary_preds_flat = all_binary_preds.reshape(-1)
    all_labels_flat = all_labels.reshape(-1)
    all_binary_probs_flat = all_binary_probs.reshape(-1)
    all_binary_labels_flat = (all_labels_flat > 0).astype(int)
    
    # Calculate binary classification metrics
    binary_accuracy = np.mean(all_binary_preds_flat == all_binary_labels_flat)
    binary_f1 = f1_score(all_binary_labels_flat, all_binary_preds_flat)
    
    # Create confusion matrices
    binary_cm = confusion_matrix(all_binary_labels_flat, all_binary_preds_flat)
    
    # Calculate ROC AUC for binary classification
    binary_auc = roc_auc_score(all_binary_labels_flat, all_binary_probs_flat)
    
    print("\nFinal Evaluation Results:")
    print("Binary Classification Results:")
    print(f"Accuracy: {binary_accuracy:.4f}")
    print(f"F1 Score: {binary_f1:.4f}")
    print(f"ROC AUC: {binary_auc:.4f}")
    print(f"Confusion Matrix:\n{binary_cm}")
    
    # Detection rate by class
    class_names = ['Normal', 'Unnecessary Retx', 'Missing Retx', 'New Data No Retx', 'Max Retx Achieved']
    print("\nDetection Rate by Class:")
    for i in range(5):
        # Count total instances of this class
        class_total = np.sum(all_labels_flat == i)
        
        if class_total > 0:
            # Count true anomaly detections for this class
            if i == 0:  # Normal
                class_correct = np.sum((all_binary_preds_flat == 0) & (all_labels_flat == 0))
            else:  # Anomaly classes
                class_correct = np.sum((all_binary_preds_flat == 1) & (all_labels_flat == i))
            
            detection_rate = class_correct / class_total
            print(f"Class {i} ({class_names[i]}): {int(class_correct)}/{int(class_total)} = {detection_rate:.2%}")
        else:
            print(f"Class {i} ({class_names[i]}): 0/0 = 0.00%")
    
    return {
        'binary_accuracy': binary_accuracy,
        'binary_f1': binary_f1,
        'binary_auc': binary_auc,
        'binary_cm': binary_cm,
        'binary_preds': all_binary_preds,
        'true_labels': all_labels,
        'binary_probs': all_binary_probs,
        'all_errors': [x.numpy() for x in all_errors]
    }

if __name__ == "__main__":
    # Configuration
    config = {
        # Data paths
        'train_parquet_path': 'unscaled_pdsch_val.parquet',
        'test_parquet_path': 'unscaled_pdsch_val_min.parquet',
        
        # Feature configuration
        'all_features': ['SFN', 'Slot', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI'],
        'all_feature_dims': [1024, 31, 16, 33, 2, 9, 2],
        
        # Dataset parameters
        'seq_len': 20,
        'batch_size': 64,
        'num_workers': 0 if torch.cuda.is_available() else min(os.cpu_count(), 4),
        'sample_fraction': 0.1,
        
        # Model architecture
        'hidden_dim': 128,
        'embedding_dim': 8,
        'num_layers': 2,
        'dropout': 0.3,
        
        # Training parameters
        'learning_rate': 0.001,
        'num_epochs': 50,
        'early_stopping_patience': 10,
        
        # Loss weights
        'binary_weight': 1.0,
        'reconstruction_weight': 0.5,
        'regularization_weight': 0.3
    }
    
    # Create data loaders
    print("Creating data loaders...")
    from dataset import create_data_loaders
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
    
    # Create model
    print("Creating BiGRU model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BiGRUAnomalyDetector(
        feature_dims=config['all_feature_dims'],
        hidden_dim=config['hidden_dim'],
        embedding_dim=config['embedding_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        device=device
    ).to(device)
    
    # Print model size
    total_params = model.count_parameters()
    print(f"Model size: {total_params:,} parameters ({total_params/1000:.2f}K)")
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Train the model
    print("Training model...")
    model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config['num_epochs'],
        device=device,
        early_stopping_patience=config['early_stopping_patience'],
        binary_weight=config['binary_weight'],
        reconstruction_weight=config['reconstruction_weight'],
        regularization_weight=config['regularization_weight']
    )
    
    # Evaluate the model
    print("Evaluating model...")
    evaluation_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device
    )
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.close()
    
    # Save model
    model_filename = "anomaly_detector_bigru_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'evaluation_results': {k: v for k, v in evaluation_results.items() 
                              if not isinstance(v, np.ndarray) or v.size < 1000}
    }, model_filename)
    
    print(f"Model training and evaluation complete! Model saved as {model_filename}")