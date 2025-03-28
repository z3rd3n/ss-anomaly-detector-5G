import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, f1_score
import os
import json
import logging
import time
from datetime import timedelta
import seaborn as sns


# Set plot style for academic presentation - use safe default styles
try:
    plt.style.use('seaborn-whitegrid')  # Try older naming convention
except:
    try:
        plt.style.use('ggplot')  # Fallback to ggplot
    except:
        pass  # Use default style if all else fails

# Set general plot parameters
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 1


def setup_logger(log_file='anomaly_detector.log'):
    """
    Set up logger with file and console handlers
    """
    # Create logger
    logger = logging.getLogger('anomaly_detector')
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
    def __init__(self, input_dim, hidden_dim, dropout, num_heads=4):
        super().__init__()
        
        # Make input dimension compatible with num_heads
        if input_dim % num_heads != 0:
            # Adjust to nearest multiple
            adjusted_dim = ((input_dim // num_heads) + 1) * num_heads
            self.dim_adapter = nn.Linear(input_dim, adjusted_dim)
            self.input_dim = adjusted_dim
        else:
            self.dim_adapter = None
            self.input_dim = input_dim
        
        self.attention = nn.MultiheadAttention(
            embed_dim=self.input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer to incorporate reconstruction error per timestep
        self.error_integration = nn.Linear(input_dim + 1, input_dim)  # +1 for error
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # Hidden layer
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.layer_norm3 = nn.LayerNorm(hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        
        # Output layer
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, error_per_timestep):
        # Apply self-attention
        if self.dim_adapter is not None:
            x = self.dim_adapter(x)
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
        
        # Pass through hidden layer
        x = self.hidden(x)
        x = self.activation(x)
        x = self.layer_norm3(x)
        x = self.dropout3(x)
        
        # Final binary prediction
        return self.linear(x)  # [batch_size, seq_len, 1]

class BiGRUAnomalyDetector(nn.Module):
    """
    BiGRU model for anomaly detection with embedding for all features
    """
    def __init__(
        self,
        feature_dims,
        embedding_dim=8,
        hidden_dim=128,
        num_layers=2,
        gru_dropout=0.2,
        position_encoding=True,
        bidirectional=True,
        attention_heads=4,
        attention_dropout=0.2,
        classifier_hidden_dim=64,
        classifier_dropout=0.3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(BiGRUAnomalyDetector, self).__init__()
        
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.device = device
        self.bidirectional = bidirectional
        
        # Define features for reference
        self.features = ['SFN', 'Slot', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI']
        
        # Embedding layer for all features
        self.feature_embedding = EmbeddingLayer(
            feature_dims, 
            embedding_dim=embedding_dim,
            position_encoding=position_encoding
        )
        
        # Combined input dimension after embedding all features
        self.combined_feature_dim = len(feature_dims) * embedding_dim
        
        # Bidirectional GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=self.combined_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension from GRU
        self.gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        
        # Reconstruction layer for embedded features (for the error loss)
        self.reconstruction_layer = nn.Linear(self.gru_output_dim, self.combined_feature_dim)
        
        # Binary Classification head using attention
        self.binary_classifier = AttentionBinaryClassifier(
            input_dim=self.gru_output_dim,
            hidden_dim=classifier_hidden_dim,
            dropout=classifier_dropout,
            num_heads=attention_heads
        )
        
    def forward(self, feature_data):
        # feature_data: [batch_size, seq_len, num_features]
        batch_size, seq_len, _ = feature_data.size()
        
        # Embed all features
        embedded_features = self.feature_embedding(feature_data, seq_len)
        
        # Process with bidirectional GRU
        gru_output, _ = self.gru(embedded_features)
        # gru_output: [batch_size, seq_len, hidden_dim * 2] if bidirectional
        
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
            'instance_attn_weights': instance_attn_weights,
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
            expected_count = self.anomaly_counts[anomaly_type]
            
            # Find positions with this anomaly type
            type_mask = (labels == anomaly_type).float()  # [batch_size, seq_len]
            
            # Calculate detected count in this batch
            detected_mask = type_mask * binary_preds  # Only count correct detections
            detected_count = torch.sum(detected_mask)
            
            # Add component to regularization loss
            # We want to maximize detected_count/expected_count, so we minimize -log(detected/expected)
            if detected_count > 0:
                if detected_count > expected_count:
                    logger.warning(f"Detected count ({detected_count}) > expected count ({expected_count}) for anomaly type {anomaly_type}")
                    
                type_loss = -torch.log((detected_count + 1e-6) / expected_count)

                # Apply additional weights to anomaly types 2 and 3
                if anomaly_type in [2, 3]:
                    type_loss *= 2.0  # Double the weight for types 2 and 3
                regularization_loss += type_loss
            
        return regularization_loss

def plot_batch_losses(batch_losses, title_prefix="Training", output_dir="plots", window_size=10):
    """
    Plot the batch-level losses with moving average smoothing.
    Higher quality version for thesis presentation.
    
    Args:
        batch_losses: Dictionary containing lists of different loss types
        title_prefix: Prefix for the plot title (e.g., "Training" or "Evaluation")
        output_dir: Directory to save plots
        window_size: Window size for moving average smoothing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create batch indices
    batch_indices = np.arange(1, len(batch_losses['total']) + 1)
    
    # Apply moving average smoothing
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    # Plot combined losses
    plt.figure(figsize=(12, 8))
    
    for loss_type, values in batch_losses.items():
        if len(values) >= window_size:
            smoothed_values = moving_average(values, window_size)
            smoothed_indices = batch_indices[window_size-1:]
            plt.plot(smoothed_indices, smoothed_values, linewidth=2, label=f'{loss_type.capitalize()} Loss')
        else:
            plt.plot(batch_indices, values, linewidth=2, label=f'{loss_type.capitalize()} Loss')
    
    plt.xlabel('Batch', fontweight='bold')
    plt.ylabel('Loss Value', fontweight='bold')
    plt.title(f'{title_prefix} Batch-Level Losses', fontweight='bold')
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    # Add descriptive text
    plt.annotate(f'Window Size: {window_size}', xy=(0.02, 0.02), xycoords='figure fraction',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Save the figure with high DPI
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{title_prefix.lower()}_batch_losses.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create separate plots for each loss component with detailed statistics
    for loss_type, values in batch_losses.items():
        plt.figure(figsize=(12, 8))
        
        # Calculate statistics
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        # Plot raw values with light color
        plt.plot(batch_indices, values, alpha=0.4, color='gray', label='Raw Values')
        
        # Plot smoothed values if enough data points
        if len(values) >= window_size:
            smoothed_values = moving_average(values, window_size)
            smoothed_indices = batch_indices[window_size-1:]
            plt.plot(smoothed_indices, smoothed_values, linewidth=3, 
                    label=f'Smoothed (window={window_size})')
        
        # Add mean and median lines
        plt.axhline(y=mean_val, color='r', linestyle='-', alpha=0.7, label=f'Mean: {mean_val:.4f}')
        plt.axhline(y=median_val, color='g', linestyle='--', alpha=0.7, label=f'Median: {median_val:.4f}')
        
        # Add standard deviation band
        plt.fill_between(batch_indices, mean_val - std_val, mean_val + std_val, 
                        color='blue', alpha=0.1, label=f'Std Dev: {std_val:.4f}')
            
        plt.xlabel('Batch', fontweight='bold')
        plt.ylabel(f'{loss_type.capitalize()} Loss', fontweight='bold')
        plt.title(f'{title_prefix} {loss_type.capitalize()} Loss Over Batches', fontweight='bold')
        
        # Add statistics text box
        stats_text = (f"Statistics:\n"
                    f"Mean: {mean_val:.4f}\n"
                    f"Median: {median_val:.4f}\n"
                    f"Std Dev: {std_val:.4f}\n"
                    f"Min: {min_val:.4f}\n"
                    f"Max: {max_val:.4f}")
        
        plt.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{title_prefix.lower()}_{loss_type}_loss.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_combined_batch_losses(train_batch_losses, eval_batch_losses, output_dir="plots", window_size=10):
    """
    Plot training and evaluation losses together for comparison.
    
    Args:
        train_batch_losses: Dictionary containing training batch losses
        eval_batch_losses: Dictionary containing evaluation batch losses
        output_dir: Directory to save plots
        window_size: Window size for moving average smoothing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have evaluation data
    if not eval_batch_losses or len(eval_batch_losses['total']) == 0:
        return
    
    # Apply moving average smoothing
    def moving_average(data, window_size):
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    # Plot each loss type separately
    loss_types = ['total', 'reconstruction', 'binary', 'regularization']
    
    for loss_type in loss_types:
        if loss_type in train_batch_losses and loss_type in eval_batch_losses:
            plt.figure(figsize=(12, 8))
            
            # Get data
            train_values = train_batch_losses[loss_type]
            eval_values = eval_batch_losses[loss_type]
            
            # Create batch indices
            train_indices = np.arange(1, len(train_values) + 1)
            eval_indices = np.arange(1, len(eval_values) + 1)
            
            # Apply smoothing
            train_smoothed = moving_average(train_values, min(window_size, len(train_values)))
            eval_smoothed = moving_average(eval_values, min(window_size, len(eval_values)))
            
            # Adjust indices for smoothed data
            if len(train_values) >= window_size:
                train_smoothed_indices = train_indices[window_size-1:]
            else:
                train_smoothed_indices = train_indices
                
            if len(eval_values) >= window_size:
                eval_smoothed_indices = eval_indices[window_size-1:]
            else:
                eval_smoothed_indices = eval_indices
            
            # Plot smoothed values
            plt.plot(train_smoothed_indices, train_smoothed, 'b-', linewidth=2, 
                    label=f'Training {loss_type.capitalize()} Loss')
            plt.plot(eval_smoothed_indices, eval_smoothed, 'r-', linewidth=2, 
                    label=f'Evaluation {loss_type.capitalize()} Loss')
            
            # Add raw data as light scatter points
            plt.scatter(train_indices, train_values, color='blue', alpha=0.1, s=10)
            plt.scatter(eval_indices, eval_values, color='red', alpha=0.1, s=10)
            
            # Calculate key statistics
            train_mean = np.mean(train_values)
            eval_mean = np.mean(eval_values)
            
            # Add mean lines
            plt.axhline(y=train_mean, color='blue', linestyle='--', alpha=0.5, 
                      label=f'Training Mean: {train_mean:.4f}')
            plt.axhline(y=eval_mean, color='red', linestyle='--', alpha=0.5, 
                      label=f'Evaluation Mean: {eval_mean:.4f}')
            
            plt.xlabel('Batch', fontweight='bold')
            plt.ylabel(f'{loss_type.capitalize()} Loss', fontweight='bold')
            plt.title(f'Training vs. Evaluation {loss_type.capitalize()} Loss', fontweight='bold')
            plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
            plt.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = (f"Training Stats:\n"
                        f"  Mean: {np.mean(train_values):.4f}\n"
                        f"  Std Dev: {np.std(train_values):.4f}\n\n"
                        f"Evaluation Stats:\n"
                        f"  Mean: {np.mean(eval_values):.4f}\n"
                        f"  Std Dev: {np.std(eval_values):.4f}")
            
            plt.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/combined_{loss_type}_loss.png', dpi=300, bbox_inches='tight')
            plt.close()

def plot_batch_metrics(batch_metrics, output_dir="plots", window_size=10):
    """
    Plot batch-level evaluation metrics with moving average smoothing.
    
    Args:
        batch_metrics: Dictionary containing lists of different metrics
        output_dir: Directory to save plots
        window_size: Window size for moving average smoothing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create batch indices
    batch_indices = np.arange(1, len(next(iter(batch_metrics.values()))) + 1)
    
    # Apply moving average smoothing
    def moving_average(data, window_size):
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    # Plot combined metrics
    plt.figure(figsize=(12, 8))
    
    for metric_name, values in batch_metrics.items():
        if metric_name in ['accuracy', 'f1', 'auc']:  # Only plot these three together
            if len(values) >= window_size:
                smoothed_values = moving_average(values, window_size)
                smoothed_indices = batch_indices[window_size-1:]
                plt.plot(smoothed_indices, smoothed_values, linewidth=2, label=f'{metric_name.capitalize()}')
            else:
                plt.plot(batch_indices, values, linewidth=2, label=f'{metric_name.capitalize()}')
    
    plt.xlabel('Batch', fontweight='bold')
    plt.ylabel('Score', fontweight='bold')
    plt.title('Evaluation Metrics Across Batches', fontweight='bold')
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)  # Metrics are typically in range [0,1]
    
    # Add descriptive text
    plt.annotate(f'Window Size: {window_size}', xy=(0.02, 0.02), xycoords='figure fraction',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Save the figure with high DPI
    plt.tight_layout()
    plt.savefig(f'{output_dir}/batch_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create separate plots for each metric with detailed statistics
    for metric_name, values in batch_metrics.items():
        plt.figure(figsize=(12, 8))
        
        # Calculate statistics
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        # Plot raw values with light color
        plt.plot(batch_indices, values, alpha=0.4, color='gray', label='Raw Values')
        
        # Plot smoothed values if enough data points
        if len(values) >= window_size:
            smoothed_values = moving_average(values, window_size)
            smoothed_indices = batch_indices[window_size-1:]
            plt.plot(smoothed_indices, smoothed_values, linewidth=3, 
                    label=f'Smoothed (window={window_size})')
        
        # Add mean and median lines
        plt.axhline(y=mean_val, color='r', linestyle='-', alpha=0.7, label=f'Mean: {mean_val:.4f}')
        plt.axhline(y=median_val, color='g', linestyle='--', alpha=0.7, label=f'Median: {median_val:.4f}')
        
        # Add standard deviation band
        plt.fill_between(batch_indices, mean_val - std_val, mean_val + std_val, 
                        color='blue', alpha=0.1, label=f'Std Dev: {std_val:.4f}')
            
        plt.xlabel('Batch', fontweight='bold')
        plt.ylabel(f'{metric_name.capitalize()}', fontweight='bold')
        plt.title(f'Batch-Level {metric_name.capitalize()} Scores', fontweight='bold')
        
        if metric_name in ['accuracy', 'f1', 'auc']:
            plt.ylim(0, 1.05)  # Set y-axis limits for these metrics
        
        # Add statistics text box
        stats_text = (f"Statistics:\n"
                    f"Mean: {mean_val:.4f}\n"
                    f"Median: {median_val:.4f}\n"
                    f"Std Dev: {std_val:.4f}\n"
                    f"Min: {min_val:.4f}\n"
                    f"Max: {max_val:.4f}")
        
        plt.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/batch_{metric_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_confusion_matrix(cm, class_names, output_dir="plots"):
    """
    Generate a high-quality confusion matrix plot for a thesis.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a matplotlib figure
    plt.figure(figsize=(10, 8))
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
               xticklabels=class_names, yticklabels=class_names)
    
    # Add normalized values as text
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.7, f"({cm_norm[i, j]:.1%})", 
                    ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white",
                    fontsize=9)
    
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.title('Confusion Matrix', fontweight='bold')
    
    # Add metrics derived from confusion matrix
    if cm.shape[0] == 2:
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        if tp + fp > 0:  # Avoid division by zero
            precision = tp / (tp + fp)
        else:
            precision = 0
        if tp + fn > 0:  # Avoid division by zero
            recall = tp / (tp + fn)
        else:
            recall = 0
        if precision + recall > 0:  # Avoid division by zero
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
            
        metrics_text = (f"Metrics:\n"
                      f"Accuracy: {accuracy:.4f}\n"
                      f"Precision: {precision:.4f}\n"
                      f"Recall: {recall:.4f}\n"
                      f"F1 Score: {f1:.4f}")
        
        plt.annotate(metrics_text, xy=(0.02, 0.02), xycoords='figure fraction',
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_detection_rates(detection_rates, class_names, output_dir="plots"):
    """
    Create a high-quality bar chart of detection rates by class.
    
    Args:
        detection_rates: Dictionary mapping class indices to detection rates
        class_names: List of class names
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Extract data
    indices = []
    rates = []
    labels = []
    
    for i, name in enumerate(class_names):
        if i in detection_rates:
            indices.append(i)
            rates.append(detection_rates[i])
            labels.append(name)
    
    # Define a custom colormap
    if len(indices) > 0:
        colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
        
        # Create bars with numeric values displayed
        bars = plt.bar(range(len(rates)), rates, color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.2%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(range(len(rates)), labels, rotation=45, ha='right')
        plt.ylim(0, max(1.0, max(rates) * 1.1))  # Set upper limit to at least 1.0
        plt.xlabel('Class', fontweight='bold')
        plt.ylabel('Detection Rate', fontweight='bold')
        plt.title('Class-Specific Detection Rates', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add descriptive statistics
        stats_text = "Detection Rates:\n"
        for i, rate in enumerate(rates):
            stats_text += f"{labels[i]}: {rate:.2%}\n"
        
        plt.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/class_detection_rates.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_roc_curve(y_true, y_scores, output_dir="plots"):
    """
    Generate a high-quality ROC curve suitable for a thesis.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities or scores
        output_dir: Directory to save the plot
    """
    from sklearn.metrics import roc_curve, auc
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random classifier')
    
    # Add markers for specific thresholds
    threshold_markers = [0.1, 0.3, 0.5, 0.7, 0.9]
    for threshold in threshold_markers:
        # Find the closest threshold value
        idx = (np.abs(thresholds - threshold)).argmin()
        if idx < len(fpr):  # Ensure index is valid
            plt.scatter(fpr[idx], tpr[idx], marker='o', color='red', s=50,
                      label=f'Threshold = {threshold:.1f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
    plt.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    # Add some analysis text
    if roc_auc > 0.9:
        quality = "Excellent"
    elif roc_auc > 0.8:
        quality = "Good"
    elif roc_auc > 0.7:
        quality = "Fair"
    elif roc_auc > 0.6:
        quality = "Poor"
    else:
        quality = "Failing"
        
    analysis_text = (f"ROC Curve Analysis:\n"
                   f"AUC: {roc_auc:.4f}\n"
                   f"Model Quality: {quality}\n\n"
                   f"Selected thresholds are\n"
                   f"marked with red dots.")
    
    plt.annotate(analysis_text, xy=(0.02, 0.97), xycoords='axes fraction',
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_evaluation_summary(eval_results, output_dir="plots"):
    """
    Create a comprehensive evaluation summary dashboard.
    
    Args:
        eval_results: Dictionary of evaluation results
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    fig.suptitle('Anomaly Detection Model Evaluation Summary', fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Key Metrics (Top Left)
    metrics = ['binary_accuracy', 'binary_f1', 'binary_auc']
    metric_names = ['Accuracy', 'F1 Score', 'AUC']
    values = [eval_results.get(m, 0) for m in metrics]
    
    axes[0, 0].bar(metric_names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].set_title('Performance Metrics', fontweight='bold')
    axes[0, 0].set_ylabel('Score', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add values on top of bars
    for i, v in enumerate(values):
        axes[0, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # 2. Loss Components (Top Right)
    if 'test_losses' in eval_results:
        loss_types = list(eval_results['test_losses'].keys())
        loss_values = list(eval_results['test_losses'].values())
        
        axes[0, 1].bar(loss_types, loss_values, 
                     color=['#d62728', '#9467bd', '#8c564b', '#e377c2'])
        axes[0, 1].set_title('Loss Components', fontweight='bold')
        axes[0, 1].set_ylabel('Loss Value', fontweight='bold')
        axes[0, 1].set_yscale('log')  # Log scale to show all components clearly
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add values on bars
        for i, v in enumerate(loss_values):
            axes[0, 1].text(i, v * 1.1, f'{v:.4f}', ha='center', rotation=0,
                          fontsize=9, fontweight='bold')
    
    # 3. Confusion Matrix (Bottom Left)
    if 'binary_cm' in eval_results:
        cm = eval_results['binary_cm']
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                   xticklabels=['Normal', 'Anomaly'], 
                   yticklabels=['Normal', 'Anomaly'],
                   ax=axes[1, 0])
        
        axes[1, 0].set_title('Confusion Matrix', fontweight='bold')
        axes[1, 0].set_xlabel('Predicted Label', fontweight='bold')
        axes[1, 0].set_ylabel('True Label', fontweight='bold')
        
        # Calculate metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Add metrics as text
        metrics_text = (f"Precision: {precision:.4f}\n"
                      f"Recall: {recall:.4f}\n"
                      f"TNR: {tn/(tn+fp):.4f}\n"
                      f"FPR: {fp/(tn+fp):.4f}")
        
        axes[1, 0].text(0.05, -0.2, metrics_text, transform=axes[1, 0].transAxes,
                      fontsize=10, verticalalignment='top')
    
    # 4. Class Detection Rates (Bottom Right)
    if 'class_detection_rates' in eval_results:
        detection_rates = eval_results['class_detection_rates']
        class_names = ['Normal', 'Type 1', 'Type 2', 'Type 3', 'Type 4']
        # Short names for the plot
        short_names = ['Normal', 'Type 1', 'Type 2', 'Type 3', 'Type 4']
        
        # Extract data for plotting
        indices = []
        rates = []
        names = []
        
        for i, name in enumerate(short_names):
            if i in detection_rates:
                indices.append(i)
                rates.append(detection_rates[i])
                names.append(name)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
        axes[1, 1].bar(names, rates, color=colors)
        axes[1, 1].set_title('Class-Specific Detection Rates', fontweight='bold')
        axes[1, 1].set_ylabel('Detection Rate', fontweight='bold')
        axes[1, 1].set_ylim(0, 1.0)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add labels on bars
        for i, v in enumerate(rates):
            axes[1, 1].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    
    # Add timestamp
    plt.figtext(0.5, 0.01, f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}", 
              ha="center", fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout with space for the title
    plt.savefig(f'{output_dir}/evaluation_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def log_false_positives(binary_preds, binary_labels, binary_probs, feature_data, 
                       timestamps_list, feature_names, epoch):
    """
    Log all false positives with their features and probabilities
    and identify the most suspicious features
    """
    # Create flat list of all timestamps
    all_timestamps = []
    for batch_timestamps in timestamps_list:
        all_timestamps.extend([ts for sublist in batch_timestamps for ts in sublist])
    
    # Initialize lists to store false positives and true negatives (normal instances)
    fp_timestamps = []
    fp_probs = []
    fp_features = []
    
    # Lists to store normal instances for reference
    normal_features = []
    
    # Find all false positives and collect normal instances
    for batch_idx in range(binary_preds.shape[0]):
        for seq_idx in range(binary_preds.shape[1]):
            # Check if this is a false positive (predicted anomaly but actually normal)
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
            
            # Collect normal instances (true negatives)
            elif binary_preds[batch_idx, seq_idx] == 0 and binary_labels[batch_idx, seq_idx] == 0:
                features = feature_data[batch_idx, seq_idx].tolist()
                normal_features.append(features)
    
    # Calculate reference statistics for normal data
    normal_features_array = np.array(normal_features)
    normal_means = np.mean(normal_features_array, axis=0)
    normal_stds = np.std(normal_features_array, axis=0)
    
    def identify_suspicious_features(features, normal_means, normal_stds, feature_names, threshold=1.5):
        """
        Enhanced method to identify suspicious features using weighted statistical deviations
        and domain-specific importance.
        """
        suspicious = []
        
        # Feature importance weights based on domain knowledge for 5G HARQ process
        # Adjusted to give higher weight to more anomaly-relevant features
        importance_weights = {
            'SFN': 0.7,     # System Frame Number - lower relevance
            'Slot': 0.7,    # Slot - lower relevance
            'HARQ': 1.5,    # HARQ process ID - highly relevant for anomaly detection
            'MCS': 1.2,     # Modulation and Coding Scheme - important for performance
            'CRC': 2.0,     # CRC result - critical error indicator
            'ReTx': 2.5,    # Retransmission - directly relates to anomalies
            'NDI': 1.8      # New Data Indicator - strong indicator of protocol behavior
        }
        
        # Calculate feature distributions in percentile terms
        feature_ranks = []
        for i, (feat, mean, std) in enumerate(zip(features, normal_means, normal_stds)):
            feature_name = feature_names[i]
            weight = importance_weights.get(feature_name, 1.0)
            
            if std > 0:  # Avoid division by zero
                # Calculate z-score
                z_score = abs(feat - mean) / std
                
                # Apply non-linear scaling to emphasize extreme values
                scaled_score = np.tanh(z_score) * 2.0  # Tanh to cap extreme values
                
                # Apply feature-specific weight
                weighted_score = scaled_score * weight
                
                if weighted_score > threshold:
                    suspicious.append((feature_name, weighted_score))
                    
                    # Store additional info about direction of deviation (higher/lower than normal)
                    direction = "+" if feat > mean else "-"
                    feature_ranks.append((feature_name, weighted_score, direction))
        
        # Sort by weighted score (most anomalous first)
        suspicious.sort(key=lambda x: x[1], reverse=True)
        
        # Format with more detailed information
        if suspicious:
            result = []
            for name, score in suspicious[:3]:  # Limit to top 3 most suspicious
                direction = "+" if features[feature_names.index(name)] > normal_means[feature_names.index(name)] else "-"
                result.append(f"{name}{direction} ({score:.2f})")
            return "--".join(result)
        else:
            return "None"
    
    # Create a dictionary to store unique false positives
    unique_fps = {}
    for ts, prob, feat in zip(fp_timestamps, fp_probs, fp_features):
        if ts not in unique_fps or prob > unique_fps[ts][0]:
            # Identify suspicious features
            suspicious = identify_suspicious_features(feat, normal_means, normal_stds, feature_names)
            unique_fps[ts] = (prob, feat, suspicious)
    
    # Sort by probability in descending order
    sorted_fps = sorted(unique_fps.items(), key=lambda x: x[1][0], reverse=True)
    
    # Write to file
    with open(f'false_positives_{epoch}.csv', 'w') as f:
        f.write('timestamp,probability,suspicious_features,' + ','.join(feature_names) + '\n')
        for ts, (prob, feat, suspicious) in sorted_fps:
            f.write(f'{ts},{prob:.6f},{suspicious},' + ','.join(map(str, feat)) + '\n')
    
    logger.info(f"\nLogged {len(sorted_fps)} unique false positives to false_positives_{epoch}.csv")
    

def plot_enhanced_training_losses(batch_losses, output_dir="plots", window_size=100):
    """
    Create enhanced training loss visualizations suitable for thesis/publication.
    
    Args:
        batch_losses: Dictionary containing lists of different loss types
        output_dir: Directory to save plots
        window_size: Window size for moving average smoothing
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import LogLocator, AutoMinorLocator
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define a better color palette (colorblind-friendly)
    colors = {
        'total': '#1f77b4',      # blue
        'binary': '#2ca02c',     # green
        'reconstruction': '#ff7f0e',  # orange
        'regularization': '#d62728'   # red
    }
    
    # Create batch indices
    batch_indices = np.arange(1, len(batch_losses['total']) + 1)
    
    # Apply moving average smoothing
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    # 1. MULTI-PANEL VISUALIZATION
    # Create a 2x2 grid for individual and combined views
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2])
    
    # FIRST ROW: Binary and Reconstruction losses (linear scale)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    
    # Plot Binary Loss (first subplot)
    raw_binary = batch_losses['binary']
    if len(raw_binary) >= window_size:
        smoothed_binary = moving_average(raw_binary, window_size)
        smoothed_indices = batch_indices[window_size-1:]
        ax1.plot(smoothed_indices, smoothed_binary, color=colors['binary'], linewidth=2.5, label='Smoothed')
    ax1.plot(batch_indices, raw_binary, color=colors['binary'], alpha=0.2, linewidth=0.8, label='Raw')
    ax1.set_title('Binary Loss', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Batch', fontsize=14)
    ax1.set_ylabel('Loss Value', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Add stats to binary loss plot
    binary_mean = np.mean(raw_binary)
    binary_median = np.median(raw_binary)
    binary_std = np.std(raw_binary)
    stats_text = f"Mean: {binary_mean:.4f}\nMedian: {binary_median:.4f}\nStd Dev: {binary_std:.4f}"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot Reconstruction Loss (second subplot)
    raw_recon = batch_losses['reconstruction']
    if len(raw_recon) >= window_size:
        smoothed_recon = moving_average(raw_recon, window_size)
        ax2.plot(smoothed_indices, smoothed_recon, color=colors['reconstruction'], linewidth=2.5, label='Smoothed')
    ax2.plot(batch_indices, raw_recon, color=colors['reconstruction'], alpha=0.2, linewidth=0.8, label='Raw')
    ax2.set_title('Reconstruction Loss', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Batch', fontsize=14)
    ax2.set_ylabel('Loss Value', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Add stats to reconstruction loss plot
    recon_mean = np.mean(raw_recon)
    recon_median = np.median(raw_recon)
    recon_std = np.std(raw_recon)
    stats_text = f"Mean: {recon_mean:.4f}\nMedian: {recon_median:.4f}\nStd Dev: {recon_std:.4f}"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # SECOND ROW: Regularization and Total losses (linear scale)
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    
    # Plot Regularization Loss
    raw_reg = batch_losses['regularization']
    if len(raw_reg) >= window_size:
        smoothed_reg = moving_average(raw_reg, window_size)
        ax3.plot(smoothed_indices, smoothed_reg, color=colors['regularization'], linewidth=2.5, label='Smoothed')
    ax3.plot(batch_indices, raw_reg, color=colors['regularization'], alpha=0.2, linewidth=0.8, label='Raw')
    ax3.set_title('Regularization Loss', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Batch', fontsize=14)
    ax3.set_ylabel('Loss Value', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    
    # Add stats to regularization loss plot
    reg_mean = np.mean(raw_reg)
    reg_median = np.median(raw_reg)
    reg_std = np.std(raw_reg)
    stats_text = f"Mean: {reg_mean:.4f}\nMedian: {reg_median:.4f}\nStd Dev: {reg_std:.4f}"
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot Total Loss
    raw_total = batch_losses['total']
    if len(raw_total) >= window_size:
        smoothed_total = moving_average(raw_total, window_size)
        ax4.plot(smoothed_indices, smoothed_total, color=colors['total'], linewidth=2.5, label='Smoothed')
    ax4.plot(batch_indices, raw_total, color=colors['total'], alpha=0.2, linewidth=0.8, label='Raw')
    ax4.set_title('Total Loss', fontsize=16, fontweight='bold')
    ax4.set_xlabel('Batch', fontsize=14)
    ax4.set_ylabel('Loss Value', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right')
    
    # Add stats to total loss plot
    total_mean = np.mean(raw_total)
    total_median = np.median(raw_total)
    total_std = np.std(raw_total)
    stats_text = f"Mean: {total_mean:.4f}\nMedian: {total_median:.4f}\nStd Dev: {total_std:.4f}"
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # THIRD ROW: Combined losses view (left: linear, right: logarithmic)
    ax5 = plt.subplot(gs[2, 0])
    ax6 = plt.subplot(gs[2, 1])
    
    # Calculate the stable region (after initial convergence)
    # Find where the first derivative stabilizes (use total loss as reference)
    if len(raw_total) >= window_size*2:
        # Use smoothed gradient to find stabilization point
        smoothed = moving_average(raw_total, window_size)
        gradient = np.gradient(smoothed)
        smooth_gradient = moving_average(np.abs(gradient), window_size)
        
        # Find where gradient becomes small and consistent
        threshold = 0.1 * np.max(smooth_gradient[:len(smooth_gradient)//3])
        stable_indices = np.where(smooth_gradient < threshold)[0]
        
        if len(stable_indices) > 0:
            stable_point = stable_indices[0] + window_size*2
            stable_point = min(stable_point, len(batch_indices) // 5)  # Cap at 20% of total batches
        else:
            stable_point = len(batch_indices) // 5
    else:
        stable_point = len(batch_indices) // 5  # Default: 20% of total batches
    
    # Linear scale combined plot (emphasis on stable region)
    for loss_type, color_key in zip(['total', 'regularization', 'binary', 'reconstruction'], 
                                   ['total', 'regularization', 'binary', 'reconstruction']):
        if len(batch_losses[loss_type]) >= window_size:
            smoothed = moving_average(batch_losses[loss_type], window_size)
            ax5.plot(smoothed_indices, smoothed, color=colors[color_key], linewidth=2, 
                    label=f'{loss_type.capitalize()}')
    
    # Mark the stable region
    ax5.axvline(x=stable_point, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    ax5.text(stable_point + 50, ax5.get_ylim()[1]*0.9, 'Stable Region', 
             rotation=0, fontsize=12, backgroundcolor='white', alpha=0.7)
    
    ax5.set_title('Combined Training Losses (Linear Scale)', fontsize=16, fontweight='bold')
    ax5.set_xlabel('Batch', fontsize=14)
    ax5.set_ylabel('Loss Value', fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper right')
    
    # Logarithmic scale combined plot (for magnitude comparison)
    for loss_type, color_key in zip(['total', 'regularization', 'binary', 'reconstruction'], 
                                   ['total', 'regularization', 'binary', 'reconstruction']):
        if len(batch_losses[loss_type]) >= window_size:
            smoothed = moving_average(batch_losses[loss_type], window_size)
            ax6.semilogy(smoothed_indices, smoothed, color=colors[color_key], linewidth=2, 
                        label=f'{loss_type.capitalize()}')
    
    ax6.set_title('Combined Training Losses (Log Scale)', fontsize=16, fontweight='bold')
    ax6.set_xlabel('Batch', fontsize=14)
    ax6.set_ylabel('Loss Value (log)', fontsize=14)
    ax6.grid(True, which="both", alpha=0.3)
    ax6.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/enhanced_training_losses.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 2. STABLE REGION FOCUS
    # Create a plot focused on the stable learning region
    if len(batch_indices) > stable_point:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot linear scale
        for loss_type, color_key in zip(['total', 'regularization', 'binary', 'reconstruction'], 
                                       ['total', 'regularization', 'binary', 'reconstruction']):
            if len(batch_losses[loss_type]) >= window_size:
                smoothed = moving_average(batch_losses[loss_type], window_size)
                stable_idx_start = max(0, stable_point - window_size + 1)
                if stable_idx_start < len(smoothed):
                    ax1.plot(smoothed_indices[stable_idx_start:], smoothed[stable_idx_start:], 
                            color=colors[color_key], linewidth=2, label=f'{loss_type.capitalize()}')
        
        ax1.set_title('Stable Region Losses (Linear Scale)', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Batch', fontsize=14)
        ax1.set_ylabel('Loss Value', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # Plot log scale
        for loss_type, color_key in zip(['total', 'regularization', 'binary', 'reconstruction'], 
                                       ['total', 'regularization', 'binary', 'reconstruction']):
            if len(batch_losses[loss_type]) >= window_size:
                smoothed = moving_average(batch_losses[loss_type], window_size)
                stable_idx_start = max(0, stable_point - window_size + 1)
                if stable_idx_start < len(smoothed):
                    ax2.semilogy(smoothed_indices[stable_idx_start:], smoothed[stable_idx_start:], 
                                color=colors[color_key], linewidth=2, label=f'{loss_type.capitalize()}')
        
        ax2.set_title('Stable Region Losses (Log Scale)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Batch', fontsize=14)
        ax2.set_ylabel('Loss Value (log)', fontsize=14)
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/stable_region_losses.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. LOSS CONTRIBUTION STACKED PLOT
    if all(k in batch_losses for k in ['binary', 'reconstruction', 'regularization']):
        # Calculate relative contributions after weighting
        binary_weight = 1.0  # You may need to adjust these based on your actual weights
        recon_weight = 0.5
        reg_weight = 0.3
        
        binary_contrib = np.array(batch_losses['binary']) * binary_weight
        recon_contrib = np.array(batch_losses['reconstruction']) * recon_weight
        reg_contrib = np.array(batch_losses['regularization']) * reg_weight
        
        # Create stacked area plot of loss contributions
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # For clearer visuals in stack plot, use moving average
        if len(binary_contrib) >= window_size:
            binary_contrib_smooth = moving_average(binary_contrib, window_size)
            recon_contrib_smooth = moving_average(recon_contrib, window_size)
            reg_contrib_smooth = moving_average(reg_contrib, window_size)
            stack_indices = batch_indices[window_size-1:]
        else:
            binary_contrib_smooth = binary_contrib
            recon_contrib_smooth = recon_contrib
            reg_contrib_smooth = reg_contrib
            stack_indices = batch_indices
        
        # Create stack plot
        ax.stackplot(stack_indices, 
                    [binary_contrib_smooth, recon_contrib_smooth, reg_contrib_smooth],
                    labels=['Binary Loss', 'Reconstruction Loss', 'Regularization Loss'],
                    colors=[colors['binary'], colors['reconstruction'], colors['regularization']],
                    alpha=0.7)
        
        ax.set_title('Weighted Loss Contribution Analysis', fontsize=16, fontweight='bold')
        ax.set_xlabel('Batch', fontsize=14)
        ax.set_ylabel('Weighted Loss Contribution', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/loss_contribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. TRAINING PHASES ANALYSIS
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # 4.1 Linear scale plot with phase annotations
    total_batches = len(batch_indices)
    
    # Define phases (adjust based on your specific training)
    phases = [
        (0, stable_point, "Initial Convergence"),
        (stable_point, total_batches, "Stable Training")
    ]
    
    # Optional: If you can identify more phases
    # middle_point = (stable_point + total_batches) // 2
    # phases = [
    #     (0, stable_point, "Initial Convergence"),
    #     (stable_point, middle_point, "Refinement"),
    #     (middle_point, total_batches, "Fine-tuning")
    # ]
    
    # Plot total loss with phases highlighted
    if len(batch_losses['total']) >= window_size:
        smoothed_total = moving_average(batch_losses['total'], window_size)
        ax1.plot(smoothed_indices, smoothed_total, color=colors['total'], linewidth=2.5)
    ax1.plot(batch_indices, batch_losses['total'], color=colors['total'], alpha=0.15, linewidth=0.8)
    
    # Highlight phases with different background colors
    phase_colors = ['#ffeeee', '#eeffee', '#eeeeff']
    for i, (start, end, name) in enumerate(phases):
        if i < len(phase_colors):
            ax1.axvspan(start, end, alpha=0.2, color=phase_colors[i])
        ax1.axvline(x=start, color='black', linestyle='--', alpha=0.7)
        
        # Place text in the middle of the phase region
        mid_x = (start + end) / 2
        y_pos = ax1.get_ylim()[1] * 0.9
        ax1.text(mid_x, y_pos, name, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_title('Training Loss with Phase Analysis', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Total Loss', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 4.2 Log scale plot with phase annotations
    # Plot all loss components in log scale
    for loss_type, color_key in zip(['total', 'regularization', 'binary', 'reconstruction'], 
                                   ['total', 'regularization', 'binary', 'reconstruction']):
        if len(batch_losses[loss_type]) >= window_size:
            smoothed = moving_average(batch_losses[loss_type], window_size)
            ax2.semilogy(smoothed_indices, smoothed, color=colors[color_key], linewidth=2, 
                        label=f'{loss_type.capitalize()}')
    
    # Add phase separators
    for start, _, _ in phases:
        ax2.axvline(x=start, color='black', linestyle='--', alpha=0.7)
    
    ax2.set_title('Log Scale Loss Components with Training Phases', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Batch', fontsize=14)
    ax2.set_ylabel('Loss Value (log)', fontsize=14)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_phases_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. CREATE PUBLICATION-GRADE INDIVIDUAL PLOTS
    for loss_type in batch_losses.keys():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get the raw values
        raw_values = batch_losses[loss_type]
        
        # Calculate statistics
        mean_val = np.mean(raw_values)
        median_val = np.median(raw_values)
        std_val = np.std(raw_values)
        min_val = np.min(raw_values)
        max_val = np.max(raw_values)
        
        # Plot raw values with light color
        ax.plot(batch_indices, raw_values, alpha=0.3, color='gray', linewidth=0.8, label='Raw Values')
        
        # Plot smoothed values if enough data points
        if len(raw_values) >= window_size:
            smoothed_values = moving_average(raw_values, window_size)
            smoothed_indices = batch_indices[window_size-1:]
            ax.plot(smoothed_indices, smoothed_values, linewidth=2.5, 
                  color=colors.get(loss_type, '#1f77b4'),
                  label=f'Smoothed (window={window_size})')
        
        # Add mean and median lines
        ax.axhline(y=mean_val, color='r', linestyle='-', alpha=0.7, label=f'Mean: {mean_val:.4f}')
        ax.axhline(y=median_val, color='g', linestyle='--', alpha=0.7, label=f'Median: {median_val:.4f}')
        
        # Add standard deviation band
        ax.fill_between(batch_indices, mean_val - std_val, mean_val + std_val, 
                      color='blue', alpha=0.1, label=f'±1 Std Dev: {std_val:.4f}')
        
        # Mark phases
        for i, (start, end, name) in enumerate(phases):
            ax.axvline(x=start, color='black', linestyle='--', alpha=0.7)
            if i == 0:  # Only add label for the first one to avoid legend clutter
                ax.text(start + 50, max_val * 0.95, name, rotation=0, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Batch', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{loss_type.capitalize()} Loss', fontsize=14, fontweight='bold')
        ax.set_title(f'Training {loss_type.capitalize()} Loss Analysis', fontsize=16, fontweight='bold')
        
        # Add statistics text box
        stats_text = (f"Statistics:\n"
                    f"Mean: {mean_val:.6f}\n"
                    f"Median: {median_val:.6f}\n"
                    f"Std Dev: {std_val:.6f}\n"
                    f"Min: {min_val:.6f}\n"
                    f"Max: {max_val:.6f}")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
              verticalalignment='top', fontsize=11,
              bbox=dict(boxstyle="round", facecolor='white', ec="gray", alpha=0.8))
        
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/publication_{loss_type}_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. LOG SCALE INDIVIDUAL PLOTS (especially useful for small loss values)
        if loss_type in ['binary', 'reconstruction']:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot raw values with light color in log scale
            ax.semilogy(batch_indices, raw_values, alpha=0.3, color='gray', linewidth=0.8, label='Raw Values')
            
            # Plot smoothed values if enough data points
            if len(raw_values) >= window_size:
                smoothed_values = moving_average(raw_values, window_size)
                ax.semilogy(smoothed_indices, smoothed_values, linewidth=2.5, 
                          color=colors.get(loss_type, '#1f77b4'),
                          label=f'Smoothed (window={window_size})')
            
            ax.set_xlabel('Batch', fontsize=14, fontweight='bold')
            ax.set_ylabel(f'{loss_type.capitalize()} Loss (log scale)', fontsize=14, fontweight='bold')
            ax.set_title(f'Training {loss_type.capitalize()} Loss (Logarithmic)', fontsize=16, fontweight='bold')
            
            # Add statistics
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                  verticalalignment='top', fontsize=11,
                  bbox=dict(boxstyle="round", facecolor='white', ec="gray", alpha=0.8))
            
            ax.legend(loc='upper right')
            ax.grid(True, which="both", alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/publication_{loss_type}_loss_log.png', dpi=300, bbox_inches='tight')
            plt.close()

    print(f"Enhanced training loss visualizations saved to {output_dir}/")
    return

def plot_normalized_confusion_matrix(cm, class_names, output_dir="plots", filename="normalized_confusion_matrix.png", figsize=(12, 10)):
    """
    Plot a normalized confusion matrix with both raw counts and percentages.
    
    Args:
        cm: Confusion matrix array (raw counts)
        class_names: List of class names
        output_dir: Directory to save the plot
        filename: Filename for the saved plot
        figsize: Figure size as (width, height) tuple
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure
    plt.figure(figsize=figsize)
    
    # Create a normalized version of the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Set colormap that works well for normalized values
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    # Create the heatmap
    ax = sns.heatmap(cm_norm, annot=False, cmap=cmap, vmin=0, vmax=1, 
                square=True, xticklabels=class_names, yticklabels=class_names, cbar=True)
    
    # Add text annotations with both counts and percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Calculate percentage for cell (handle div by zero)
            if np.sum(cm[i, :]) > 0:
                percentage = cm[i, j] / np.sum(cm[i, :]) * 100
            else:
                percentage = 0
                
            # For the text color, use white for darker cells, black for lighter cells
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            
            # Show both count and percentage
            plt.text(j + 0.5, i + 0.5, f"{cm[i, j]}\n({percentage:.1f}%)", 
                    ha="center", va="center", fontsize=11, color=color,
                    fontweight="bold")
    
    # Add titles and labels
    plt.xlabel("Predicted Label", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=14, fontweight="bold")
    plt.title("Normalized Confusion Matrix", fontsize=16, fontweight="bold")
    
    # Calculate and display accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.5, 0.01, f"Overall Accuracy: {accuracy:.2%}", 
               ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Also create a version showing only percentages (sometimes cleaner for publication)
    plt.figure(figsize=figsize)
    
    # Create heatmap with only normalized values
    ax = sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap=cmap, vmin=0, vmax=1,
                square=True, xticklabels=class_names, yticklabels=class_names, cbar=True)
    
    # Adjust annotation color based on cell darkness
    for text in ax.texts:
        val = float(text.get_text().strip('%')) / 100
        text.set_color("white" if val > 0.5 else "black")
        text.set_fontweight("bold")
    
    plt.xlabel("Predicted Label", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=14, fontweight="bold")
    plt.title("Normalized Confusion Matrix (Percentages)", fontsize=16, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "percentage_" + filename), dpi=300, bbox_inches="tight")
    plt.close()
    
    # For direct integration with your existing code, also return the normalized confusion matrix
    return cm_norm

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
    regularization_weight=0.5,
    threshold=0.5,
    log_batch_interval=10  # Log losses every N batches
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
    best_val_f1 = 0.0
    early_stopping_counter = 0
    train_losses = []
    val_losses = []
    
    # Initialize batch-level loss tracking
    batch_losses = {
        'total': [],
        'reconstruction': [],
        'binary': [],
        'regularization': []
    }
    
    batch_times = []
    
    # Initialize best metrics dictionary
    best_val_metrics = {
        'binary_f1': 0.0,
        'binary_acc': 0.0,
        'binary_auc': 0.0,
        'epoch': 0
    }
    
    # Track total training time
    total_start_time = time.time()
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_rec_loss = 0.0
        train_binary_loss = 0.0
        train_reg_loss = 0.0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        
        batch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_progress):
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
            
            # Calculate batch time
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            batch_times.append(batch_duration)
            batch_start_time = batch_end_time  # Reset for next batch
            
            # Track batch-level losses
            batch_losses['total'].append(total_loss.item())
            batch_losses['reconstruction'].append(loss_dict['reconstruction_loss'].item())
            batch_losses['binary'].append(loss_dict['binary_loss'].item())
            batch_losses['regularization'].append(loss_dict['regularization_loss'].item())
            
            # Update progress bar with current losses and timing
            train_progress.set_postfix({
                'loss': f"{total_loss.item():.4f}", 
                'rec': f"{loss_dict['reconstruction_loss'].item():.4f}",
                'bin': f"{loss_dict['binary_loss'].item():.4f}",
                'reg': f"{loss_dict['regularization_loss'].item():.4f}",
                'time/batch': f"{batch_duration:.3f}s"
            })
            
            # Log batch losses at intervals
            if (batch_idx + 1) % log_batch_interval == 0:
                logger.info(f"Batch {batch_idx+1}/{len(train_loader)} - "
                          f"Loss: {total_loss.item():.4f}, "
                          f"Rec: {loss_dict['reconstruction_loss'].item():.4f}, "
                          f"Bin: {loss_dict['binary_loss'].item():.4f}, "
                          f"Reg: {loss_dict['regularization_loss'].item():.4f}, "
                          f"Time: {batch_duration:.3f}s")
        
        # Calculate average training losses
        num_batches = len(train_loader)
        train_loss /= num_batches
        train_rec_loss /= num_batches
        train_binary_loss /= num_batches
        train_reg_loss /= num_batches
        train_losses.append(train_loss)
        
        # Calculate epoch time
        epoch_duration = time.time() - epoch_start_time
        
        # Validation phase
        val_start_time = time.time()
        model.eval()
        val_loss = 0.0
        val_rec_loss = 0.0
        val_binary_loss = 0.0
        val_reg_loss = 0.0
        
        all_binary_preds = []
        all_labels = []
        all_binary_labels = []
        all_binary_probs = []
        all_timestamps = []
        all_feature_data = []
        
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
                binary_preds = (binary_probs >= threshold).float()
                
                # Create binary labels
                binary_labels = (labels > 0).float()
                
                # Collect predictions and labels for metrics
                all_binary_preds.append(binary_preds.cpu())
                all_labels.append(labels.cpu())
                all_binary_labels.append(binary_labels.cpu())
                all_binary_probs.append(binary_probs.cpu())
                all_timestamps.append(timestamps)
                all_feature_data.append(feature_data.cpu())
                
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
        
        # Calculate validation time
        val_duration = time.time() - val_start_time
        
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
        try:
            binary_auc = roc_auc_score(all_binary_labels_flat, all_binary_probs_flat)
        except:
            # This can happen if all samples are of one class
            binary_auc = 0.0
        
        # Display results with timing information
        logger.info(f"\nValidation Results (Epoch {epoch+1}):")
        logger.info(f"Overall Loss: {val_loss:.4f} (Rec: {val_rec_loss:.4f}, "
               f"Bin: {val_binary_loss:.4f}, Reg: {val_reg_loss:.4f})")
        logger.info(f"Binary Detection - Accuracy: {binary_acc:.4f}, F1: {binary_f1:.4f}, AUC: {binary_auc:.4f}")
        logger.info(f"Epoch Time: {timedelta(seconds=int(epoch_duration))}, "
               f"Training: {timedelta(seconds=int(epoch_duration-val_duration))}, "
               f"Validation: {timedelta(seconds=int(val_duration))}")
        
        # Calculate and log speed metrics
        samples_per_epoch = len(train_loader.dataset)
        samples_per_second = samples_per_epoch / (epoch_duration - val_duration)
        logger.info(f"Training Speed: {samples_per_second:.2f} samples/second")
        
        # Log confusion matrices
        binary_cm = confusion_matrix(all_binary_labels_flat, all_binary_preds_flat)
        logger.info("\nBinary Confusion Matrix (Normal vs Anomaly):")
        logger.info("                  Predicted")
        logger.info("                Normal  Anomaly")
        logger.info(f"Actual Normal   {binary_cm[0][0]:<8} {binary_cm[0][1]:<8}")
        logger.info(f"Actual Anomaly  {binary_cm[1][0]:<8} {binary_cm[1][1]:<8}")
        
        # Calculate detection rate by class
        class_names = ['Normal', 'Type 1', 'Type 2', 'Type 3', 'Type 4']
        logger.info("\nDetection Rate by Class:")
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
                logger.info(f"Class {i} ({class_names[i]}): {int(class_correct)}/{int(class_total)} = {detection_rate:.2%}")
            else:
                logger.info(f"Class {i} ({class_names[i]}): 0/0 = 0.00%")
        
        # Update learning rate scheduler if provided
        if scheduler is not None:
            scheduler.step(val_loss)
            
        detection_rates = {
            i: (np.sum((all_binary_preds_flat == (0 if i == 0 else 1)) & (all_labels_flat == i)) / np.sum(all_labels_flat == i))
            for i in range(5) if np.sum(all_labels_flat == i) > 0
        }
            
        # Create the models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Save the model in the models directory
        model_filename = os.path.join('models', f"anomaly_detector_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_f1': binary_f1,
        }, model_filename)
        
        # Check for early stopping based on F1 score
        if binary_f1 > best_val_f1:
            best_val_f1 = binary_f1
            early_stopping_counter = 0
            # Save the best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_f1': binary_f1,
            }, 'best_model.pth')
            logger.info(f"New best model saved with validation F1: {binary_f1:.4f}")
            
            # Update best metrics
            best_val_metrics = {
                'binary_f1': binary_f1,
                'binary_acc': binary_acc,
                'binary_auc': binary_auc,
                'epoch': epoch + 1
            }
        else:
            early_stopping_counter += 1
            logger.info(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
            
            if early_stopping_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Plot batch-level losses after each epoch
        plot_batch_losses(batch_losses, title_prefix="Training", output_dir="plots", window_size=log_batch_interval)
    
    # Calculate total training time
    total_duration = time.time() - total_start_time
    logger.info(f"\nTotal Training Time: {timedelta(seconds=int(total_duration))}")
    logger.info(f"Average Batch Time: {np.mean(batch_times):.4f}s")
    logger.info(f"Average Samples/Second: {len(train_loader.dataset) * num_epochs / total_duration:.2f}")
    
    # Load the best model
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, train_losses, val_losses, batch_losses, {
        'total_time': total_duration,
        'batch_times': batch_times,
        'avg_batch_time': np.mean(batch_times),
        'samples_per_second': len(train_loader.dataset) * num_epochs / total_duration
    }

def evaluate_model(
    model,
    test_loader,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    threshold=0.5
):
    """
    Evaluate the model on test data and calculate metrics with timing information and batch-wise tracking
    """
    model.eval()
    
    # Get anomaly counts from the dataset
    anomaly_counts = test_loader.dataset.anomaly_counts if hasattr(test_loader.dataset, 'anomaly_counts') else None
    num_batches = len(test_loader)
    
    # Create loss function with anomaly counts
    criterion = AnomalyTypeLoss(
        binary_weight=1.0,
        reconstruction_weight=1.0,
        regularization_weight=0.5,
        anomaly_counts=anomaly_counts,
        num_batches=num_batches
    )
    
    # Initialize lists to store predictions and true labels
    all_binary_preds = []
    all_labels = []
    all_binary_probs = []
    
    # Track attention weights and reconstruction errors
    all_temporal_attn = []
    all_instance_attn = []
    all_errors = []
    
    # Track batch-wise losses
    eval_batch_losses = {
        'total': [],
        'reconstruction': [],
        'binary': [],
        'regularization': []
    }
    
    # Track batch-wise metrics
    batch_metrics = {
        'accuracy': [],
        'f1': [],
        'auc': [],
        'precision': [],
        'recall': []
    }
    
    # Track timing information
    inference_times = []
    start_time = time.time()
    
    eval_progress = tqdm(test_loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_progress):
            feature_data = batch['feature_data'].to(device)
            labels = batch['label'].to(device)
            
            # Measure inference time
            batch_start = time.time()
            
            # Forward pass
            outputs = model(feature_data)
            
            # Calculate inference time
            inference_time = time.time() - batch_start
            inference_times.append(inference_time)
            
            # Calculate loss
            loss_dict = criterion(outputs, labels)
            
            # Track batch-wise losses
            eval_batch_losses['total'].append(loss_dict['total_loss'].item())
            eval_batch_losses['reconstruction'].append(loss_dict['reconstruction_loss'].item())
            eval_batch_losses['binary'].append(loss_dict['binary_loss'].item())
            eval_batch_losses['regularization'].append(loss_dict['regularization_loss'].item())
            
            # Get binary predictions
            binary_probs = outputs['binary_probs'].squeeze(-1)  # [batch_size, seq_len]
            binary_preds = (binary_probs >= threshold).float()
            
            # Create binary labels from multiclass labels (labels > 0 means anomaly)
            binary_labels = (labels > 0).float()
            
            # Calculate batch-level metrics
            batch_binary_preds = binary_preds.cpu().numpy().flatten()
            batch_binary_labels = binary_labels.cpu().numpy().flatten()
            batch_binary_probs = binary_probs.cpu().numpy().flatten()
            
            batch_acc = np.mean(batch_binary_preds == batch_binary_labels)
            
            # Handle edge cases for metrics calculation
            if np.sum(batch_binary_labels) > 0 and np.sum(batch_binary_preds) > 0:
                batch_f1 = f1_score(batch_binary_labels, batch_binary_preds)
                try:
                    batch_auc = roc_auc_score(batch_binary_labels, batch_binary_probs)
                except:
                    batch_auc = 0.5  # Default for random classifier
                    
                # Calculate precision and recall
                true_positives = np.sum((batch_binary_preds == 1) & (batch_binary_labels == 1))
                false_positives = np.sum((batch_binary_preds == 1) & (batch_binary_labels == 0))
                false_negatives = np.sum((batch_binary_preds == 0) & (batch_binary_labels == 1))
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            else:
                batch_f1 = 0.0
                batch_auc = 0.5
                precision = 0.0
                recall = 0.0
            
            # Store batch metrics
            batch_metrics['accuracy'].append(batch_acc)
            batch_metrics['f1'].append(batch_f1)
            batch_metrics['auc'].append(batch_auc)
            batch_metrics['precision'].append(precision)
            batch_metrics['recall'].append(recall)
            
            # Store predictions and true labels
            all_binary_preds.append(binary_preds.cpu())
            all_labels.append(labels.cpu())
            all_binary_probs.append(binary_probs.cpu())
            
            # Store attention weights and reconstruction errors
            if 'instance_attn_weights' in outputs:
                all_instance_attn.append(outputs['instance_attn_weights'].cpu())
            if 'error_per_timestep' in outputs:
                all_errors.append(outputs['error_per_timestep'].cpu())
                
            # Update progress bar
            current_metrics = {
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'acc': f"{batch_acc:.4f}",
                'f1': f"{batch_f1:.4f}",
                'time': f"{inference_time:.3f}s"
            }
            eval_progress.set_postfix(current_metrics)
    
    # Calculate total evaluation time
    total_eval_time = time.time() - start_time
    
    # Calculate average test losses
    avg_test_losses = {
        'total': np.mean(eval_batch_losses['total']),
        'reconstruction': np.mean(eval_batch_losses['reconstruction']),
        'binary': np.mean(eval_batch_losses['binary']),
        'regularization': np.mean(eval_batch_losses['regularization'])
    }
    
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
    try:
        binary_auc = roc_auc_score(all_binary_labels_flat, all_binary_probs_flat)
    except:
        binary_auc = 0.0
        
    # Plot binary confusion matrix
    plot_confusion_matrix(binary_cm, ['Normal', 'Anomaly'], output_dir="plots")

    # For multiclass confusion matrix
    # For multiclass confusion matrix
    if len(np.unique(all_labels_flat)) > 1:
        # Create class-specific predictions (using true labels for labeling the anomaly types)
        multi_preds = np.zeros_like(all_labels_flat)
        # Normal predictions (class 0)
        multi_preds[(all_binary_preds_flat == 0)] = 0
        # For anomalies, use the true anomaly type when correctly detected
        for i in range(1, 5):  # Type 1-4
            # When true label is type i and prediction is anomaly, assign type i
            correctly_detected = (all_labels_flat == i) & (all_binary_preds_flat == 1)
            multi_preds[correctly_detected] = i

        # Create multiclass confusion matrix
        multi_cm = confusion_matrix(all_labels_flat, multi_preds, labels=range(5))

        # Define class names
        multi_class_names = ['Normal', 'Type 1', 'Type 2', 'Type 3', 'Type 4']

        # Plot normalized multiclass confusion matrix
        plot_normalized_confusion_matrix(multi_cm, multi_class_names, output_dir="plots")
        logger.info("\nMulticlass confusion matrix saved to plots directory")

    # Calculate timing statistics
    avg_inference_time = np.mean(inference_times)
    samples_per_second = len(test_loader.dataset) / total_eval_time
    
    logger.info("\nFinal Evaluation Results:")
    logger.info(f"Test Loss: {avg_test_losses['total']:.4f} (Rec: {avg_test_losses['reconstruction']:.4f}, "
           f"Bin: {avg_test_losses['binary']:.4f}, Reg: {avg_test_losses['regularization']:.4f})")
    logger.info("Binary Classification Results:")
    logger.info(f"Accuracy: {binary_accuracy:.4f}")
    logger.info(f"F1 Score: {binary_f1:.4f}")
    logger.info(f"ROC AUC: {binary_auc:.4f}")
    logger.info(f"Confusion Matrix:\n{binary_cm}")
    
    # Log timing information
    logger.info("\nTiming Information:")
    logger.info(f"Total Evaluation Time: {timedelta(seconds=int(total_eval_time))}")
    logger.info(f"Average Inference Time per Batch: {avg_inference_time:.6f} seconds")
    logger.info(f"Inference Speed: {samples_per_second:.2f} samples/second")
    
    # Detection rate by class
    class_names = ['Normal', 'Type 1', 'Type 2', 'Type 3', 'Type 4']
    logger.info("\nDetection Rate by Class:")
    class_detection_rates = {}
    
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
            class_detection_rates[i] = detection_rate
            logger.info(f"Class {i} ({class_names[i]}): {int(class_correct)}/{int(class_total)} = {detection_rate:.2%}")
        else:
            class_detection_rates[i] = 0.0
            logger.info(f"Class {i} ({class_names[i]}): 0/0 = 0.00%")
    
    # Plot evaluation metrics and create visualizations
    plot_batch_losses(eval_batch_losses, title_prefix="Evaluation", output_dir="plots", window_size=min(10, len(eval_batch_losses['total'])))
    plot_batch_metrics(batch_metrics, output_dir="plots", window_size=min(10, len(batch_metrics['accuracy'])))
    plot_class_detection_rates(class_detection_rates, class_names, output_dir="plots")
    
    # Plot ROC curve if possible
    plot_roc_curve(all_binary_labels_flat, all_binary_probs_flat, output_dir="plots")
    
    # Create overall evaluation summary
    plot_evaluation_summary({
        'binary_accuracy': binary_accuracy,
        'binary_f1': binary_f1,
        'binary_auc': binary_auc,
        'binary_cm': binary_cm,
        'class_detection_rates': class_detection_rates,
        'test_losses': avg_test_losses
    }, output_dir="plots")
    
    return {
        'binary_accuracy': binary_accuracy,
        'binary_f1': binary_f1,
        'binary_auc': binary_auc,
        'binary_cm': binary_cm,
        'class_detection_rates': class_detection_rates,
        'binary_preds': all_binary_preds,
        'true_labels': all_labels,
        'binary_probs': all_binary_probs,
        'binary_labels': all_binary_labels_flat,
        'all_errors': [x.numpy() for x in all_errors] if all_errors else [],
        'test_losses': avg_test_losses,
        'batch_losses': eval_batch_losses,
        'batch_metrics': batch_metrics,
        'timing': {
            'total_eval_time': total_eval_time,
            'avg_inference_time': avg_inference_time,
            'samples_per_second': samples_per_second
        }
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
        'seq_len': 88,
        'batch_size': 32,
        'num_workers': 0 if torch.cuda.is_available() else min(os.cpu_count(), 4),
        'sample_fraction': 1.0,
        
        # Embedding layer
        'embedding_dim': 14,
        'position_encoding': True,
        
        # GRU layer
        'hidden_dim': 410,
        'num_layers': 1, 
        'bidirectional': False,
        'gru_dropout': 0.25,
        
        # Attention layers
        'attention_heads': 10,
        'attention_dropout': 0.35,
        
        # Output layers
        'classifier_hidden_dim': 38,
        'classifier_dropout': 0.05,
        
        # Training parameters
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'num_epochs': 1,  # Set to 1 as requested
        'early_stopping_patience': 10,
        'lr_scheduler_patience': 5, 
        'lr_scheduler_factor': 0.5,
        'log_batch_interval': 100,  # Log losses every 100 batches
        
        # Loss weights
        'binary_weight': 1.0,
        'reconstruction_weight': 0.5,
        'regularization_weight': 0.3,
        
        # Evaluation parameters
        'threshold': 0.4,  # Threshold for binary classification
        
        # Random seed for reproducibility
        'seed': 42,
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
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        gru_dropout=config['gru_dropout'],
        position_encoding=config['position_encoding'],
        bidirectional=config['bidirectional'],
        attention_heads=config['attention_heads'],
        classifier_hidden_dim=config['classifier_hidden_dim'],
        classifier_dropout=config['classifier_dropout'],
        device=device
    ).to(device)
    
    # Print model size
    total_params = model.count_parameters()
    print(f"Model size: {total_params:,} parameters ({total_params/1000:.2f}K)")
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config['lr_scheduler_factor'], 
        patience=config['lr_scheduler_patience'], 
        verbose=True
    )
    
    # Train the model with timing
    print("Training model...")
    training_start_time = time.time()
    
    model, train_losses, val_losses, batch_losses, timing_stats = train_model(
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
        regularization_weight=config['regularization_weight'],
        threshold=config['threshold'],
        log_batch_interval=config['log_batch_interval']
    )
    
    training_duration = time.time() - training_start_time
    print(f"Training completed in {timedelta(seconds=int(training_duration))}")
    
    # Evaluate the model with timing
    print("Evaluating model...")
    evaluation_start_time = time.time()
    
    evaluation_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        threshold=config['threshold']
    )
    
    evaluation_duration = time.time() - evaluation_start_time
    print(f"Evaluation completed in {timedelta(seconds=int(evaluation_duration))}")
    
    # Combine training and evaluation batch results for comparison
    plot_combined_batch_losses(
        batch_losses, 
        evaluation_results['batch_losses'], 
        output_dir="plots"
    )
    
    # Create enhanced visualizations
    plot_enhanced_training_losses(
        batch_losses=batch_losses,
        output_dir="plots/enhanced",
        window_size=100
    )
    
    # Print final timing report
    print("\n===== Model Performance Report =====")
    print(f"Device: {device}")
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Training time: {timedelta(seconds=int(training_duration))}")
    print(f"Evaluation time: {timedelta(seconds=int(evaluation_duration))}")
    print(f"Average batch processing time: {timing_stats['avg_batch_time']:.6f} seconds")
    print(f"Training throughput: {timing_stats['samples_per_second']:.2f} samples/second")
    print(f"Inference throughput: {evaluation_results['timing']['samples_per_second']:.2f} samples/second")
    print("====================================")
    
    # Save model with timing information
    model_filename = "anomaly_detector_bigru_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'evaluation_results': {k: v for k, v in evaluation_results.items() 
                              if not isinstance(v, np.ndarray) or v.size < 1000},
        'timing_stats': timing_stats,
        'test_losses': evaluation_results['test_losses']
    }, model_filename)
    
    print(f"Model training and evaluation complete! Model saved as {model_filename}")
    print(f"Evaluation plots saved to the 'plots' directory")