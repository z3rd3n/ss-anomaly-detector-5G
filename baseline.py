import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, f1_score
import os
import json
import logging
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

# Setup logger
def setup_logger(log_file='ablation_study_regularization.log'):
    """Set up logger with file and console handlers"""
    logger = logging.getLogger('ablation_study')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
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
    """Embedding layer for all features with learned positional encoding."""
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
    def __init__(self, input_dim, hidden_dim, dropout, num_heads=4, use_error_feature=True):
        super().__init__()
        
        # Flag to enable/disable error feature
        self.use_error_feature = use_error_feature
        
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
        self.layer_norm1 = nn.LayerNorm(self.input_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer to incorporate reconstruction error per timestep (conditional)
        if use_error_feature:
            self.error_integration = nn.Linear(self.input_dim + 1, self.input_dim)  # +1 for error
            self.layer_norm2 = nn.LayerNorm(self.input_dim)
            self.dropout2 = nn.Dropout(dropout)
        
        # Hidden layer
        self.hidden = nn.Linear(self.input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.layer_norm3 = nn.LayerNorm(hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        
        # Output layer
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, error_per_timestep=None):
        # Apply self-attention
        if self.dim_adapter is not None:
            x = self.dim_adapter(x)
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)  # Residual connection
        x = self.dropout1(x)
     
        # Incorporate error per timestep if enabled
        if self.use_error_feature and error_per_timestep is not None:
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
    """BiGRU model for anomaly detection with embedding for all features"""
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
        use_error_feature=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(BiGRUAnomalyDetector, self).__init__()
        
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.device = device
        self.bidirectional = bidirectional
        self.use_error_feature = use_error_feature
        
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
            num_heads=attention_heads,
            use_error_feature=use_error_feature
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
        if self.use_error_feature:
            binary_logits = self.binary_classifier(gru_output, error_per_timestep)
        else:
            binary_logits = self.binary_classifier(gru_output)
        
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
        
        # 3. Simplified Anomaly Type Regularization (conditionally applied)
        if self.regularization_weight > 0:
            regularization_loss = self.compute_simplified_anomaly_type_regularization(
                model_output, 
                labels
            )
        else:
            regularization_loss = torch.tensor(0.0, device=labels.device)
        
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
            if detected_count > 0:
                if detected_count > expected_count:
                    logger.warning(f"Detected count ({detected_count}) > expected count ({expected_count}) for anomaly type {anomaly_type}")
                    
                type_loss = -torch.log((detected_count + 1e-6) / expected_count)

                # Apply additional weights to anomaly types 2 and 3
                if anomaly_type in [2, 3]:
                    type_loss *= 2.0  # Double the weight for types 2 and 3
                regularization_loss += type_loss
            
        return regularization_loss

def train_model_for_ablation(
    config,
    train_loader,
    test_loader,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    num_epochs=2
):
    """
    Train and evaluate a model configuration for the ablation study
    """
    logger.info(f"\nRunning configuration: {config['name']}")
    logger.info(f"MSE as feature: {config['use_error_feature']}, Regularization weight: {config['regularization_weight']}")
    
    # Create model
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
        use_error_feature=config['use_error_feature'],
        device=device
    ).to(device)
    
    # Log model size
    total_params = model.count_parameters()
    logger.info(f"Model size: {total_params:,} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Get anomaly counts from the dataset (assuming this is an attribute of the dataset)
    anomaly_counts = getattr(train_loader.dataset, 'anomaly_counts', {1: 1000, 2: 100, 3: 1000, 4: 100})
    num_batches = len(train_loader)
    
    # Create loss function
    criterion = AnomalyTypeLoss(
        binary_weight=config['binary_weight'],
        reconstruction_weight=config['reconstruction_weight'],
        regularization_weight=config['regularization_weight'],
        anomaly_counts=anomaly_counts,
        num_batches=num_batches
    )
    
    # Dictionary to store all results
    results = {
        'config_name': config['name'],
        'use_error_feature': config['use_error_feature'],
        'regularization_weight': config['regularization_weight'],
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'class_detection_rates': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_rec_loss = 0.0
        train_binary_loss = 0.0
        train_reg_loss = 0.0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        
        batch_train_losses = []
        batch_rec_losses = []
        batch_bin_losses = []
        batch_reg_losses = []
        
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
            batch_train_losses.append(total_loss.item())
            batch_rec_losses.append(loss_dict['reconstruction_loss'].item())
            batch_bin_losses.append(loss_dict['binary_loss'].item())
            batch_reg_losses.append(loss_dict['regularization_loss'].item())
            
            train_loss += total_loss.item()
            train_rec_loss += loss_dict['reconstruction_loss'].item()
            train_binary_loss += loss_dict['binary_loss'].item()
            train_reg_loss += loss_dict['regularization_loss'].item()
            
            # Update progress bar
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
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        all_binary_preds = []
        all_labels = []
        all_binary_labels = []
        all_binary_probs = []
        
        val_progress = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
        
        with torch.no_grad():
            for batch in val_progress:
                feature_data = batch['feature_data'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(feature_data)
                
                # Calculate loss
                loss_dict = criterion(outputs, labels)
                total_loss = loss_dict['total_loss']
                
                # Get predictions
                binary_probs = outputs['binary_probs'].squeeze(-1)  # [batch_size, seq_len]
                binary_preds = (binary_probs >= config['threshold']).float()
                
                # Create binary labels
                binary_labels = (labels > 0).float()
                
                # Collect predictions and labels for metrics
                all_binary_preds.append(binary_preds.cpu())
                all_labels.append(labels.cpu())
                all_binary_labels.append(binary_labels.cpu())
                all_binary_probs.append(binary_probs.cpu())
                
                # Track validation loss
                val_loss += total_loss.item()
        
        # Calculate average validation loss
        val_loss /= len(test_loader)
        
        # Concatenate all predictions and labels
        all_binary_preds = torch.cat(all_binary_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_binary_labels = torch.cat(all_binary_labels, dim=0).numpy()
        all_binary_probs = torch.cat(all_binary_probs, dim=0).numpy()
        
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
            binary_auc = 0.0
            
        # Calculate binary confusion matrix
        binary_cm = confusion_matrix(all_binary_labels_flat, all_binary_preds_flat)
        
        # Calculate detection rate by class
        class_names = ['Normal', 'UReTx', 'MReTx', 'NoDReTx', 'MaxReTx']
        class_detection_rates = {}
        
        for i in range(5):
            # Count total instances of this class
            class_total = np.sum(all_labels_flat == i)
            
            if class_total > 0:
                # Count true detections for this class
                if i == 0:  # Normal
                    class_correct = np.sum((all_binary_preds_flat == 0) & (all_labels_flat == 0))
                else:  # Anomaly classes
                    class_correct = np.sum((all_binary_preds_flat == 1) & (all_labels_flat == i))
                
                detection_rate = class_correct / class_total
                class_detection_rates[i] = detection_rate
            else:
                class_detection_rates[i] = 0.0
        
        # Display results
        logger.info(f"\nValidation Results (Epoch {epoch+1}):")
        logger.info(f"Binary Metrics - F1: {binary_f1:.4f}, Accuracy: {binary_acc:.4f}, AUC: {binary_auc:.4f}")
        
        # Log confusion matrix
        logger.info("\nBinary Confusion Matrix:")
        logger.info(f"TN: {binary_cm[0,0]}, FP: {binary_cm[0,1]}")
        logger.info(f"FN: {binary_cm[1,0]}, TP: {binary_cm[1,1]}")
        
        # Log detection rates by class
        logger.info("\nDetection Rate by Class:")
        for i in range(5):
            logger.info(f"Class {i} ({class_names[i]}): {class_detection_rates[i]:.2%}")
        
        # Store results for this epoch
        results['epochs'].append(epoch + 1)
        results['train_losses'].append({
            'total': train_loss,
            'reconstruction': train_rec_loss,
            'binary': train_binary_loss,
            'regularization': train_reg_loss,
            'batch_losses': batch_train_losses,
            'batch_rec_losses': batch_rec_losses,
            'batch_bin_losses': batch_bin_losses,
            'batch_reg_losses': batch_reg_losses
        })
        results['val_losses'].append(val_loss)
        results['class_detection_rates'].append(class_detection_rates)
    
    # Calculate average detection rate across epochs
    avg_detection_rates = {}
    for i in range(5):
        avg_detection_rates[i] = np.mean([rates[i] for rates in results['class_detection_rates']])
    
    # Add average results to results dictionary
    results['avg_detection_rates'] = avg_detection_rates
    results['avg_train_loss'] = np.mean([loss['total'] for loss in results['train_losses']])
    results['avg_val_loss'] = np.mean(results['val_losses'])
    
    return results, model

def create_plots(baseline_results, enhanced_results, save_dir='regularization_ablation_plots'):
    """Create comparison plots between baseline and enhanced models"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Detection rates by class
    plt.figure(figsize=(12, 7))
    class_names = ['Normal', 'Type 1\n(UReTx)', 'Type 2\n(MReTx)', 'Type 3\n(NoDReTx)', 'Type 4\n(MaxReTx)']
    
    baseline_rates = [baseline_results['avg_detection_rates'][i] for i in range(5)]
    enhanced_rates = [enhanced_results['avg_detection_rates'][i] for i in range(5)]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, baseline_rates, width, label='Baseline: No MSE, No Reg', 
                   color='#ff9999', edgecolor='darkred', linewidth=1.5)
    rects2 = ax.bar(x + width/2, enhanced_rates, width, label='Enhanced: With MSE, With Reg', 
                   color='#66b3ff', edgecolor='darkblue', linewidth=1.5)
    
    # Add detection rate values on top of bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
    
    add_labels(rects1)
    add_labels(rects2)
    
    # Set chart title and labels
    ax.set_title('Detection Rate by Anomaly Type', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Detection Rate', fontsize=14, fontweight='bold')
    ax.set_xlabel('Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    # Set y-axis to go from 0 to 1.1
    ax.set_ylim(0, 1.1)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Tight layout and save figure
    plt.tight_layout()
    plt.savefig(f'{save_dir}/detection_rates_comparison.png', dpi=300, bbox_inches='tight')
    
    # 2. Training Loss Components
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # Baseline model
    baseline_losses = baseline_results['train_losses']
    bar_width = 0.6
    components = ['Total', 'Reconstruction', 'Binary', 'Regularization']
    
    for i, epoch_losses in enumerate(baseline_losses):
        values = [epoch_losses['total'], epoch_losses['reconstruction'], 
                 epoch_losses['binary'], epoch_losses['regularization']]
        axs[0].bar(np.arange(len(components)) + i*bar_width/len(baseline_losses), 
                 values, bar_width/len(baseline_losses), 
                 label=f'Epoch {i+1}')
    
    axs[0].set_title('Baseline: No MSE, No Reg', fontsize=14, fontweight='bold')
    axs[0].set_xticks(np.arange(len(components)) + bar_width/2 - bar_width/(2*len(baseline_losses)))
    axs[0].set_xticklabels(components)
    axs[0].set_ylabel('Loss Value', fontsize=12, fontweight='bold')
    axs[0].legend()
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Enhanced model
    enhanced_losses = enhanced_results['train_losses']
    
    for i, epoch_losses in enumerate(enhanced_losses):
        values = [epoch_losses['total'], epoch_losses['reconstruction'], 
                 epoch_losses['binary'], epoch_losses['regularization']]
        axs[1].bar(np.arange(len(components)) + i*bar_width/len(enhanced_losses), 
                 values, bar_width/len(enhanced_losses), 
                 label=f'Epoch {i+1}')
    
    axs[1].set_title('Enhanced: With MSE, With Reg', fontsize=14, fontweight='bold')
    axs[1].set_xticks(np.arange(len(components)) + bar_width/2 - bar_width/(2*len(enhanced_losses)))
    axs[1].set_xticklabels(components)
    axs[1].set_ylabel('Loss Value', fontsize=12, fontweight='bold')
    axs[1].legend()
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/loss_components_comparison.png', dpi=300)
    
    # 3. Training loss curves by batch
    plt.figure(figsize=(14, 8))
    
    # Get batch losses for both configurations
    baseline_batch_losses = []
    for epoch_loss in baseline_results['train_losses']:
        baseline_batch_losses.extend(epoch_loss['batch_losses'])
    
    enhanced_batch_losses = []
    for epoch_loss in enhanced_results['train_losses']:
        enhanced_batch_losses.extend(epoch_loss['batch_losses'])
    
    # Plot total loss
    plt.subplot(2, 2, 1)
    plt.plot(baseline_batch_losses, label='Baseline', color='red', alpha=0.7)
    plt.plot(enhanced_batch_losses, label='Enhanced', color='blue', alpha=0.7)
    plt.title('Total Loss per Batch', fontsize=12, fontweight='bold')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Get reconstruction losses
    baseline_rec_losses = []
    for epoch_loss in baseline_results['train_losses']:
        baseline_rec_losses.extend(epoch_loss['batch_rec_losses'])
    
    enhanced_rec_losses = []
    for epoch_loss in enhanced_results['train_losses']:
        enhanced_rec_losses.extend(epoch_loss['batch_rec_losses'])
    
    # Plot reconstruction loss
    plt.subplot(2, 2, 2)
    plt.plot(baseline_rec_losses, label='Baseline', color='red', alpha=0.7)
    plt.plot(enhanced_rec_losses, label='Enhanced', color='blue', alpha=0.7)
    plt.title('Reconstruction Loss per Batch', fontsize=12, fontweight='bold')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Get binary losses
    baseline_bin_losses = []
    for epoch_loss in baseline_results['train_losses']:
        baseline_bin_losses.extend(epoch_loss['batch_bin_losses'])
    
    enhanced_bin_losses = []
    for epoch_loss in enhanced_results['train_losses']:
        enhanced_bin_losses.extend(epoch_loss['batch_bin_losses'])
    
    # Plot binary loss
    plt.subplot(2, 2, 3)
    plt.plot(baseline_bin_losses, label='Baseline', color='red', alpha=0.7)
    plt.plot(enhanced_bin_losses, label='Enhanced', color='blue', alpha=0.7)
    plt.title('Binary Loss per Batch', fontsize=12, fontweight='bold')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Get regularization losses
    baseline_reg_losses = []
    for epoch_loss in baseline_results['train_losses']:
        baseline_reg_losses.extend(epoch_loss['batch_reg_losses'])
    
    enhanced_reg_losses = []
    for epoch_loss in enhanced_results['train_losses']:
        enhanced_reg_losses.extend(epoch_loss['batch_reg_losses'])
    
    # Plot regularization loss
    plt.subplot(2, 2, 4)
    plt.plot(baseline_reg_losses, label='Baseline', color='red', alpha=0.7)
    plt.plot(enhanced_reg_losses, label='Enhanced', color='blue', alpha=0.7)
    plt.title('Regularization Loss per Batch', fontsize=12, fontweight='bold')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/batch_losses.png', dpi=300)
    
    # 4. Radar chart for detection performance
    plt.figure(figsize=(10, 10))
    
    # Prepare data for radar chart
    categories = ['Normal', 'Type 1\n(UReTx)', 'Type 2\n(MReTx)', 'Type 3\n(NoDReTx)', 'Type 4\n(MaxReTx)']
    N = len(categories)
    
    # Create angles for the radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Prepare data
    baseline_values = [baseline_results['avg_detection_rates'][i] for i in range(5)]
    baseline_values += baseline_values[:1]  # Close the loop
    
    enhanced_values = [enhanced_results['avg_detection_rates'][i] for i in range(5)]
    enhanced_values += enhanced_values[:1]  # Close the loop
    
    # Create radar chart
    ax = plt.subplot(111, polar=True)
    
    # Plot baseline
    ax.plot(angles, baseline_values, 'o-', linewidth=2, 
            label='Baseline: No MSE, No Reg', color='red')
    ax.fill(angles, baseline_values, 'red', alpha=0.1)
    
    # Plot enhanced
    ax.plot(angles, enhanced_values, 'o-', linewidth=2, 
            label='Enhanced: With MSE, With Reg', color='blue')
    ax.fill(angles, enhanced_values, 'blue', alpha=0.1)
    
    # Set labels and ticks
    plt.xticks(angles[:-1], categories, fontsize=12)
    
    # Draw y-axis lines (circles)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.set_ylim(0, 1.05)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Detection Rate by Class (Radar Chart)', size=16, fontweight='bold', y=1.08)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/radar_chart.png', dpi=300)
    
    # 5. Diversity Analysis
    plt.figure(figsize=(12, 7))
    
    # Calculate diversity metrics (focusing on anomaly types 1-4)
    baseline_anomaly_rates = [baseline_results['avg_detection_rates'][i] for i in range(1, 5)]
    enhanced_anomaly_rates = [enhanced_results['avg_detection_rates'][i] for i in range(1, 5)]
    
    # Calculate:
    # 1. Mean detection rate
    # 2. Standard deviation (measure of spread)
    # 3. Min detection rate
    # 4. Max detection rate
    # 5. Range
    baseline_metrics = {
        'Mean': np.mean(baseline_anomaly_rates),
        'Std Dev': np.std(baseline_anomaly_rates),
        'Min Rate': np.min(baseline_anomaly_rates),
        'Max Rate': np.max(baseline_anomaly_rates),
        'Range': np.max(baseline_anomaly_rates) - np.min(baseline_anomaly_rates)
    }
    
    enhanced_metrics = {
        'Mean': np.mean(enhanced_anomaly_rates),
        'Std Dev': np.std(enhanced_anomaly_rates),
        'Min Rate': np.min(enhanced_anomaly_rates),
        'Max Rate': np.max(enhanced_anomaly_rates),
        'Range': np.max(enhanced_anomaly_rates) - np.min(enhanced_anomaly_rates)
    }
    
    # Prepare data for plotting
    metric_names = list(baseline_metrics.keys())
    baseline_values = list(baseline_metrics.values())
    enhanced_values = list(enhanced_metrics.values())
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, baseline_values, width, label='Baseline: No MSE, No Reg', 
                   color='#ff9999', edgecolor='darkred', linewidth=1.5)
    rects2 = ax.bar(x + width/2, enhanced_values, width, label='Enhanced: With MSE, With Reg', 
                   color='#66b3ff', edgecolor='darkblue', linewidth=1.5)
    
    # Add values on top of bars
    add_labels(rects1)
    add_labels(rects2)
    
    # Chart formatting
    ax.set_title('Diversity Analysis of Anomaly Detection', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Value', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/diversity_analysis.png', dpi=300, bbox_inches='tight')
    
    # 6. Combined detection rate bar chart (all classes)
    plt.figure(figsize=(14, 8))
    
    # Get data for all anomaly types together
    class_names = ['All Types\n(Binary)', 'Normal', 'Type 1\n(UReTx)', 'Type 2\n(MReTx)', 
                  'Type 3\n(NoDReTx)', 'Type 4\n(MaxReTx)']

    # Calculate binary metrics (average over anomaly types)
    baseline_binary_avg = np.mean([baseline_results['avg_detection_rates'][i] for i in range(1, 5)])
    enhanced_binary_avg = np.mean([enhanced_results['avg_detection_rates'][i] for i in range(1, 5)])

    # Combine metrics
    baseline_rates = [baseline_binary_avg] + [baseline_results['avg_detection_rates'][i] for i in range(5)]
    enhanced_rates = [enhanced_binary_avg] + [enhanced_results['avg_detection_rates'][i] for i in range(5)]

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, baseline_rates, width, label='Baseline: No MSE, No Reg', 
                   color='#ff9999', edgecolor='darkred', linewidth=1.5)
    rects2 = ax.bar(x + width/2, enhanced_rates, width, label='Enhanced: With MSE, With Reg', 
                   color='#66b3ff', edgecolor='darkblue', linewidth=1.5)

    # Add detection rate values on top of bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    add_labels(rects1)
    add_labels(rects2)

    # Chart formatting
    ax.set_title('Detection Rate Comparison (All Classes)', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Detection Rate', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    # Set y-axis to go from 0 to 1.1
    ax.set_ylim(0, 1.1)

    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a horizontal line at y=0.5 for reference
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/combined_detection_rates.png', dpi=300, bbox_inches='tight')

    # 7. Heatmap of detection rate improvement
    plt.figure(figsize=(10, 6))

    # Calculate improvement percentages
    improvements = []
    for i in range(5):
        baseline = baseline_results['avg_detection_rates'][i]
        enhanced = enhanced_results['avg_detection_rates'][i]

        if baseline > 0:
            relative_improvement = (enhanced - baseline) / baseline * 100
        else:
            relative_improvement = np.inf if enhanced > 0 else 0

        improvements.append(relative_improvement)

    # Create heatmap
    improvements_array = np.array(improvements).reshape(1, -1)
    ax = sns.heatmap(
        improvements_array, 
        annot=True, 
        fmt=".1f", 
        cmap="RdYlGn", 
        cbar_kws={'label': 'Relative Improvement (%)'}, 
        linewidths=0.5,
        xticklabels=class_names[1:],  # Skip the 'All Types' label
        yticklabels=['Improvement (%)']
    )

    ax.set_title('Relative Improvement in Detection Rate (Enhanced vs Baseline)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/improvement_heatmap.png', dpi=300)

    # Save comparison summary as text file
    with open(f'{save_dir}/comparison_summary.txt', 'w') as f:
        f.write("DETECTION RATE COMPARISON: BASELINE VS ENHANCED MODEL\n")
        f.write("=" * 70 + "\n\n")

        f.write("MODEL CONFIGURATIONS:\n")
        f.write(f"  Baseline: No MSE as feature input, Regularization weight = {baseline_results['regularization_weight']}\n")
        f.write(f"  Enhanced: MSE as feature input, Regularization weight = {enhanced_results['regularization_weight']}\n\n")

        f.write("OVERALL PERFORMANCE:\n")
        f.write(f"  Average Training Loss - Baseline: {baseline_results['avg_train_loss']:.4f}, Enhanced: {enhanced_results['avg_train_loss']:.4f}\n")
        f.write(f"  Average Validation Loss - Baseline: {baseline_results['avg_val_loss']:.4f}, Enhanced: {enhanced_results['avg_val_loss']:.4f}\n\n")

        f.write("DETECTION RATES BY CLASS:\n")
        for i, class_name in enumerate(['Normal', 'Type 1 (UReTx)', 'Type 2 (MReTx)', 'Type 3 (NoDReTx)', 'Type 4 (MaxReTx)']):
            baseline_rate = baseline_results['avg_detection_rates'][i]
            enhanced_rate = enhanced_results['avg_detection_rates'][i]
            diff = enhanced_rate - baseline_rate
            f.write(f"  {class_name:<20}: Baseline = {baseline_rate:.4f}, Enhanced = {enhanced_rate:.4f}, Diff = {diff:+.4f}\n")

        f.write("\nDIVERSITY METRICS (ANOMALY TYPES ONLY):\n")
        baseline_anomaly_rates = [baseline_results['avg_detection_rates'][i] for i in range(1, 5)]
        enhanced_anomaly_rates = [enhanced_results['avg_detection_rates'][i] for i in range(1, 5)]

        f.write(f"  Mean Detection Rate - Baseline: {np.mean(baseline_anomaly_rates):.4f}, Enhanced: {np.mean(enhanced_anomaly_rates):.4f}\n")
        f.write(f"  Std Deviation - Baseline: {np.std(baseline_anomaly_rates):.4f}, Enhanced: {np.std(enhanced_anomaly_rates):.4f}\n")
        f.write(f"  Min Rate - Baseline: {np.min(baseline_anomaly_rates):.4f}, Enhanced: {np.min(enhanced_anomaly_rates):.4f}\n")
        f.write(f"  Max Rate - Baseline: {np.max(baseline_anomaly_rates):.4f}, Enhanced: {np.max(enhanced_anomaly_rates):.4f}\n")
        f.write(f"  Range - Baseline: {np.max(baseline_anomaly_rates) - np.min(baseline_anomaly_rates):.4f}, Enhanced: {np.max(enhanced_anomaly_rates) - np.min(enhanced_anomaly_rates):.4f}\n")

def run_ablation_study():
    """
    Run the complete ablation study to compare models with and without 
    MSE as feature input to the classifier and regularization
    """
    # Create dataset
    from dataset import create_data_loaders

    # Define configuration for both models
    base_config = {
        # Data paths
        'train_parquet_path': 'unscaled_pdsch_val.parquet',
        'test_parquet_path': 'unscaled_pdsch_val_min.parquet',

        # Feature configuration
        'all_features': ['SFN', 'Slot', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI'],
        'all_feature_dims': [1024, 31, 16, 33, 2, 9, 2],

        # Dataset parameters
        'seq_len': 100,  # Changed to 100 per your request
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
        'num_epochs': 1,  # Train for 2 epochs as requested

        # Loss weights
        'binary_weight': 1.0,
        'reconstruction_weight': 0.5,

        # Evaluation parameters
        'threshold': 0.4,
    }

    # Create configurations for ablation study
    baseline_config = base_config.copy()
    baseline_config.update({
        'name': 'Baseline Model', 
        'use_error_feature': False,  # No MSE as input feature
        'regularization_weight': 0.3  # No regularization
    })

    enhanced_config = base_config.copy()
    enhanced_config.update({
        'name': 'Enhanced Model',
        'use_error_feature': True,  # MSE as input feature 
        'regularization_weight': 0.3  # With regularization
    })

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, test_loader, _ = create_data_loaders(
        train_parquet_path=base_config['train_parquet_path'],
        test_parquet_path=base_config['test_parquet_path'],
        all_features=base_config['all_features'],
        all_feature_dims=base_config['all_feature_dims'],
        seq_len=base_config['seq_len'],
        batch_size=base_config['batch_size'],
        num_workers=base_config['num_workers'],
        sample_fraction=base_config['sample_fraction']
    )

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Train and evaluate baseline model
    logger.info("\n" + "="*50)
    logger.info("TRAINING BASELINE MODEL: No MSE Feature, No Regularization")
    logger.info("="*50)
    baseline_results, baseline_model = train_model_for_ablation(
        baseline_config, train_loader, test_loader, device, 
        num_epochs=base_config['num_epochs']
    )

    # Train and evaluate enhanced model
    logger.info("\n" + "="*50)
    logger.info("TRAINING ENHANCED MODEL: With MSE Feature, With Regularization")
    logger.info("="*50)
    enhanced_results, enhanced_model = train_model_for_ablation(
        enhanced_config, train_loader, test_loader, device,
        num_epochs=base_config['num_epochs']
    )

    # Create comparison plots
    logger.info("\n" + "="*50)
    logger.info("CREATING COMPARISON PLOTS")
    logger.info("="*50)
    create_plots(baseline_results, enhanced_results)

    # Print summary of results
    logger.info("\n" + "="*50)
    logger.info("ABLATION STUDY SUMMARY")
    logger.info("="*50)

    # Compare detection rates
    logger.info("\nDetection Rates by Class:")
    for i, class_name in enumerate(['Normal', 'Type 1 (UReTx)', 'Type 2 (MReTx)', 'Type 3 (NoDReTx)', 'Type 4 (MaxReTx)']):
        baseline_rate = baseline_results['avg_detection_rates'][i]
        enhanced_rate = enhanced_results['avg_detection_rates'][i]
        diff = enhanced_rate - baseline_rate
        logger.info(f"  {class_name:<20}: Baseline = {baseline_rate:.4f}, Enhanced = {enhanced_rate:.4f}, Diff = {diff:+.4f}")

    # Compare diversity metrics
    baseline_anomaly_rates = [baseline_results['avg_detection_rates'][i] for i in range(1, 5)]
    enhanced_anomaly_rates = [enhanced_results['avg_detection_rates'][i] for i in range(1, 5)]

    logger.info("\nDiversity Metrics (Anomaly Types Only):")
    logger.info(f"  Mean Detection Rate - Baseline: {np.mean(baseline_anomaly_rates):.4f}, Enhanced: {np.mean(enhanced_anomaly_rates):.4f}")
    logger.info(f"  Std Deviation - Baseline: {np.std(baseline_anomaly_rates):.4f}, Enhanced: {np.std(enhanced_anomaly_rates):.4f}")

    # Calculate coefficient of variation (normalized std deviation)
    baseline_cv = np.std(baseline_anomaly_rates) / np.mean(baseline_anomaly_rates) if np.mean(baseline_anomaly_rates) > 0 else float('inf')
    enhanced_cv = np.std(enhanced_anomaly_rates) / np.mean(enhanced_anomaly_rates) if np.mean(enhanced_anomaly_rates) > 0 else float('inf')

    logger.info(f"  Coefficient of Variation - Baseline: {baseline_cv:.4f}, Enhanced: {enhanced_cv:.4f}")

    return {
        'baseline': baseline_results,
        'enhanced': enhanced_results,
        'baseline_model': baseline_model,
        'enhanced_model': enhanced_model
    }
    
class DummyDataset:
    def __init__(self, seq_len=100, num_samples=1000):
        self.anomaly_counts = {1: 100, 2: 50, 3: 75, 4: 25}
        self.seq_len = seq_len
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random features and labels
        features = torch.randint(0, 10, (self.seq_len, 7))
        
        # Ensure some samples have anomalies
        if idx % 10 == 0:  # 10% anomaly rate
            label_type = (idx // 10) % 4 + 1  # Cycle through anomaly types 1-4
            labels = torch.zeros(self.seq_len)
            # Place anomaly at random position
            pos = torch.randint(0, self.seq_len, (1,)).item()
            labels[pos] = label_type
        else:
            labels = torch.zeros(self.seq_len)
        
        return {
            'feature_data': features,
            'label': labels,
            'timestamp': [f"ts_{idx}_{i}" for i in range(self.seq_len)]
        }

# Create a dummy data loader for testing
def create_dummy_data_loaders(batch_size=32):
    train_dataset = DummyDataset(num_samples=320)  # 10 batches
    test_dataset = DummyDataset(num_samples=160)   # 5 batches
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, test_loader

# Main execution
if __name__ == "__main__":
    # Setup logging
    logger.info("Starting MSE Feature and Regularization Ablation Study")
    
    try:
        # Check if dataset module is available
        from dataset import create_data_loaders
        logger.info("Dataset module found. Using real data.")
        run_ablation_study()
    except ImportError:
        # If not, use dummy data for demonstration
        logger.info("Dataset module not found. Using dummy data for demonstration.")
        
        # Create dummy data loaders
        train_loader, test_loader = create_dummy_data_loaders()
        
        # Define simplified configs for demonstration
        baseline_config = {
            'name': 'Baseline Model',
            'all_feature_dims': [10, 10, 10, 10, 10, 10, 10],
            'embedding_dim': 8,
            'hidden_dim': 64,
            'num_layers': 1,
            'gru_dropout': 0.2,
            'position_encoding': True,
            'bidirectional': False,
            'attention_heads': 4,
            'attention_dropout': 0.2,
            'classifier_hidden_dim': 32,
            'classifier_dropout': 0.2,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'binary_weight': 1.0,
            'reconstruction_weight': 0.5,
            'use_error_feature': False,
            'regularization_weight': 0.0,
            'threshold': 0.4,
            'seq_len': 100
        }
        
        enhanced_config = baseline_config.copy()
        enhanced_config.update({
            'name': 'Enhanced Model',
            'use_error_feature': True,
            'regularization_weight': 0.3
        })
        
        # Train models on dummy data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("\nTraining baseline model on dummy data...")
        baseline_results, _ = train_model_for_ablation(
            baseline_config, train_loader, test_loader, device, num_epochs=2
        )
        
        logger.info("\nTraining enhanced model on dummy data...")
        enhanced_results, _ = train_model_for_ablation(
            enhanced_config, train_loader, test_loader, device, num_epochs=2
        )
        
        # Create comparison plots
        logger.info("\nCreating comparison plots...")
        create_plots(baseline_results, enhanced_results)
        
    logger.info("Ablation study completed successfully!")