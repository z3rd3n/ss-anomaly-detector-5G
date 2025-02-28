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

class FeatureManager:
    """
    Manages feature names, indices, and types to avoid confusion between numerical and categorical features
    """
    def __init__(self, numerical_features, categorical_features, categorical_dims):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.categorical_dims = categorical_dims
        
        # Create feature name to index mapping
        self.feature_names = numerical_features + categorical_features
        self.feature_indices = {name: idx for idx, name in enumerate(self.feature_names)}
        
        # Type mapping
        self.feature_types = {}
        for feature in numerical_features:
            self.feature_types[feature] = 'numerical'
        for feature in categorical_features:
            self.feature_types[feature] = 'categorical'
        
        # Categorical feature mapping
        self.categorical_mapping = {}
        for i, feature in enumerate(categorical_features):
            self.categorical_mapping[feature] = {
                'index': i,
                'dim': categorical_dims[i]
            }
    
    def get_feature_index(self, feature_name):
        """Get the global index of a feature"""
        return self.feature_indices.get(feature_name, -1)
    
    def get_feature_type(self, feature_name):
        """Get the type of a feature: numerical or categorical"""
        return self.feature_types.get(feature_name, None)
    
    def get_categorical_info(self, feature_name):
        """Get information about a categorical feature"""
        return self.categorical_mapping.get(feature_name, None)
    
    def get_numerical_indices(self):
        """Get numerical feature indices in the numerical feature array"""
        return list(range(len(self.numerical_features)))
    
    def get_categorical_indices(self):
        """Get categorical feature indices in the categorical feature array"""
        return list(range(len(self.categorical_features)))
    
    def __str__(self):
        """String representation of the feature manager"""
        result = "FeatureManager:\n"
        result += "  Numerical features: " + ", ".join(self.numerical_features) + "\n"
        result += "  Categorical features: " + ", ".join(self.categorical_features) + "\n"
        result += "  Feature indices:\n"
        for name, idx in self.feature_indices.items():
            result += f"    {name}: {idx}\n"
        return result

class FeatureAttention(nn.Module):
    """
    Feature attention mechanism that emphasizes important features.
    """
    def __init__(self, feature_dim, hidden_dim=64):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        
        # Calculate attention scores
        scores = self.attention(x)  # [batch_size, seq_len, feature_dim]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=2)  # [batch_size, seq_len, feature_dim]
        
        # Apply attention weights to the input
        attended_features = x * attention_weights
        
        return attended_features, attention_weights

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
        
        # Calculate attention scores
        scores = self.attention(x)  # [batch_size, seq_len, 1]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)  # [batch_size, seq_len, 1]
        
        # Apply attention weights to the input
        context_vector = torch.sum(x * attention_weights, dim=1)  # [batch_size, hidden_dim]
        
        return context_vector, attention_weights

class EmbeddingLayer(nn.Module):
    """
    Embedding layer for categorical features with learned positional encoding.
    """
    def __init__(self, categorical_dims, embedding_dim=8, position_encoding=True):
        super(EmbeddingLayer, self).__init__()
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(dim, embedding_dim) for dim in categorical_dims
        ])
        self.position_encoding = position_encoding
        self.embedding_dim = embedding_dim
        
    def forward(self, x, seq_len):
        # x shape: [batch_size, seq_len, num_categorical_features]
        batch_size, seq_len, num_features = x.size()
        
        # Apply embedding for each categorical feature
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


# For BiGRUAnomalyDetector model:
# Binary classifier using sequential processing
class SequentialBinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        output, _ = self.gru(x)
        output = self.dropout(output)
        return self.linear(output)

# Multi-class classifier using sequential processing
class SequentialMultiClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(64, 5)
        
    def forward(self, x):
        output, _ = self.gru(x)
        output = self.dropout1(output)
        output = self.linear1(output)
        output = self.relu(output)
        output = self.dropout2(output)
        return self.linear2(output)


class BiGRUAnomalyDetector(nn.Module):
    """
    Bidirectional GRU model with Feature Attention and Temporal Attention for Anomaly Detection
    
    Key components:
    1. Learnable feature attention mechanism 
    2. Bidirectional GRU for sequence modeling
    3. Temporal attention for improved context
    4. Timestep-level classification (binary and multi-class)
    """
    def __init__(
        self,
        numerical_feature_dim,
        categorical_dims,
        feature_manager,
        hidden_dim=128,
        embedding_dim=8,
        num_layers=2,
        dropout=0.2,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(BiGRUAnomalyDetector, self).__init__()
        
        self.numerical_feature_dim = numerical_feature_dim
        self.categorical_dims = categorical_dims
        self.feature_manager = feature_manager
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        
        # Embedding layer for categorical features
        self.categorical_embedding = EmbeddingLayer(
            categorical_dims, 
            embedding_dim=embedding_dim
        )
        
        # Feature attention for numerical features
        self.numerical_feature_attention = FeatureAttention(
            numerical_feature_dim, 
            hidden_dim=hidden_dim // 2
        )
        
        # Combined input dimension after processing numerical and categorical features
        self.combined_feature_dim = numerical_feature_dim + len(categorical_dims) * embedding_dim
        
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
        
        # Reconstruction layer for numerical features
        self.reconstruction_layer = nn.Linear(self.gru_output_dim, numerical_feature_dim)
        
        # Binary Classification head for each timestep
        self.binary_classifier = SequentialBinaryClassifier(
            input_dim=self.gru_output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.multiclass_classifier = SequentialMultiClassifier(
            input_dim=self.gru_output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
    def forward(self, numerical_data, categorical_data):
        # numerical_data: [batch_size, seq_len, numerical_feature_dim]
        # categorical_data: [batch_size, seq_len, num_categorical_features]
        batch_size, seq_len, _ = numerical_data.size()
        
        # Apply feature attention to numerical data
        attended_numerical, numerical_attn_weights = self.numerical_feature_attention(numerical_data)
        
        # Embed categorical data
        categorical_embeddings = self.categorical_embedding(categorical_data, seq_len)
        
        # Combine features
        combined_features = torch.cat([attended_numerical, categorical_embeddings], dim=2)
        
        # Process with bidirectional GRU
        gru_output, _ = self.gru(combined_features)
        # gru_output: [batch_size, seq_len, hidden_dim * 2]
        
        # Apply temporal attention for sequence-level context
        context_vector, temporal_attn_weights = self.temporal_attention(gru_output)
        
        # Reconstruction of numerical data for each timestep
        numerical_reconstruction = self.reconstruction_layer(gru_output)
        
        # Calculate reconstruction error for each timestep
        mse_per_timestep = torch.mean(torch.pow(numerical_data - numerical_reconstruction, 2), dim=2)
        overall_mse = torch.mean(mse_per_timestep, dim=1)
        
        # Binary classification for each timestep
        binary_logits = self.binary_classifier(gru_output)  # [batch_size, seq_len, 1]
        binary_probs = torch.sigmoid(binary_logits)
        
        # Multi-class classification for each timestep
        multiclass_logits = self.multiclass_classifier(gru_output)  # [batch_size, seq_len, 5]
        
        # Calculate instance-level attention for anomaly localization
        instance_attn_scores = binary_logits.clone()
        instance_attn_weights = F.softmax(instance_attn_scores.squeeze(-1), dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        return {
            'gru_output': gru_output,
            'numerical_reconstruction': numerical_reconstruction,
            'mse_per_timestep': mse_per_timestep,
            'overall_mse': overall_mse,
            'binary_logits': binary_logits,
            'binary_probs': binary_probs,
            'multiclass_logits': multiclass_logits,
            'numerical_attn_weights': numerical_attn_weights,
            'temporal_attn_weights': temporal_attn_weights,
            'instance_attn_weights': instance_attn_weights,
            'context_vector': context_vector
        }
    
    def count_parameters(self):
        """Count the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class TransformerAnomalyDetector(nn.Module):
    """
    Transformer-based Sequential Anomaly Detection model
    
    Key components:
    1. Learnable feature attention mechanism 
    2. Transformer encoder for sequence modeling
    3. Timestep-level classification (binary and multi-class)
    """
    def __init__(
        self,
        numerical_feature_dim,
        categorical_dims,
        feature_manager,
        hidden_dim=128,
        embedding_dim=8,
        num_layers=2,
        num_heads=4,
        dropout=0.2,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(TransformerAnomalyDetector, self).__init__()
        
        self.numerical_feature_dim = numerical_feature_dim
        self.categorical_dims = categorical_dims
        self.feature_manager = feature_manager
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        
        # Embedding layer for categorical features
        self.categorical_embedding = EmbeddingLayer(
            categorical_dims, 
            embedding_dim=embedding_dim,
            position_encoding=False  # We'll use transformer's positional encoding
        )
        
        # Feature attention for numerical features
        self.numerical_feature_attention = FeatureAttention(
            numerical_feature_dim, 
            hidden_dim=hidden_dim // 2
        )
        
        # Combined input dimension after processing numerical and categorical features
        self.combined_feature_dim = numerical_feature_dim + len(categorical_dims) * embedding_dim
        
        # Project to hidden dimension expected by transformer
        self.input_projection = nn.Linear(self.combined_feature_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, hidden_dim))  # Max seq len = 100
        nn.init.xavier_uniform_(self.positional_encoding)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Reconstruction layer
        self.reconstruction_layer = nn.Linear(hidden_dim, numerical_feature_dim)
        
        # Binary Classification head for each timestep
        # Binary Classification head for each timestep
        self.binary_classifier = SequentialBinaryClassifier(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            dropout=dropout
        )

        self.multiclass_classifier = SequentialMultiClassifier(
            input_dim=self.hidden_dim,
            hidden_dim=hidden_dim // 2,
            dropout=dropout
        )
        
    def forward(self, numerical_data, categorical_data):
        # numerical_data: [batch_size, seq_len, numerical_feature_dim]
        # categorical_data: [batch_size, seq_len, num_categorical_features]
        batch_size, seq_len, _ = numerical_data.size()
        
        # Apply feature attention to numerical data
        attended_numerical, numerical_attn_weights = self.numerical_feature_attention(numerical_data)
        
        # Embed categorical data
        categorical_embeddings = self.categorical_embedding(categorical_data, seq_len)
        
        # Combine features
        combined_features = torch.cat([attended_numerical, categorical_embeddings], dim=2)
        
        # Project to hidden dimension
        hidden_representation = self.input_projection(combined_features)
        
        # Add positional encoding
        hidden_representation = hidden_representation + self.positional_encoding[:, :seq_len, :]
        
        # Process with transformer
        transformer_output = self.transformer_encoder(hidden_representation)
        # transformer_output: [batch_size, seq_len, hidden_dim]
        
        # Reconstruction of numerical data for each timestep
        numerical_reconstruction = self.reconstruction_layer(transformer_output)
        
        # Calculate reconstruction error for each timestep
        mse_per_timestep = torch.mean(torch.pow(numerical_data - numerical_reconstruction, 2), dim=2)
        overall_mse = torch.mean(mse_per_timestep, dim=1)
        
        # Binary classification for each timestep
        binary_logits = self.binary_classifier(transformer_output)  # [batch_size, seq_len, 1]
        binary_probs = torch.sigmoid(binary_logits)
        
        # Multi-class classification for each timestep
        multiclass_logits = self.multiclass_classifier(transformer_output)  # [batch_size, seq_len, 5]
        
        # For temporal attention, use mean pooling of transformer outputs
        context_vector = torch.mean(transformer_output, dim=1)
        
        # Calculate instance-level attention for anomaly localization
        instance_attn_scores = binary_logits.clone()
        instance_attn_weights = F.softmax(instance_attn_scores.squeeze(-1), dim=1).unsqueeze(-1)
        
        # Return temporal attention weights as None since transformer uses self-attention
        temporal_attn_weights = None
        
        return {
            'transformer_output': transformer_output,
            'numerical_reconstruction': numerical_reconstruction,
            'mse_per_timestep': mse_per_timestep,
            'overall_mse': overall_mse,
            'binary_logits': binary_logits,
            'binary_probs': binary_probs,
            'multiclass_logits': multiclass_logits,
            'numerical_attn_weights': numerical_attn_weights,
            'temporal_attn_weights': temporal_attn_weights,
            'instance_attn_weights': instance_attn_weights,
            'context_vector': context_vector
        }
    
    def count_parameters(self):
        """Count the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weight for each class
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # For binary case
        if inputs.size(-1) == 1:
            # Make sure both inputs and targets have the same shape
            inputs = inputs.squeeze(-1)  # Remove last dimension
            targets = targets.squeeze(-1) if targets.dim() > 1 else targets
            
            # Calculate binary focal loss
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-BCE_loss)  # Probability of the model prediction
            
            if self.alpha is not None:
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                focal_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
            else:
                focal_loss = (1 - pt) ** self.gamma * BCE_loss
        
        # For multi-class case - rest of the code remains the same
        
        # For multi-class case
        else:
            # Convert targets to one-hot encoding for weighting
            targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
            
            # Calculate multi-class focal loss
            CE_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-CE_loss)
            
            if self.alpha is not None:
                # Apply class weights
                alpha_t = torch.sum(self.alpha * targets_one_hot, dim=1)
                focal_loss = alpha_t * (1 - pt) ** self.gamma * CE_loss
            else:
                focal_loss = (1 - pt) ** self.gamma * CE_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

class SequentialAnomalyLoss(nn.Module):
    """
    Combined loss function for sequential anomaly detection.
    
    L_total = λ_rec · L_rec + λ_bin · L_bin + λ_mul · L_mul
    """
    def __init__(
        self, 
        class_weights=None,
        binary_weight=1.0,
        multiclass_weight=1.0,
        reconstruction_weight=1.0,
        gamma=2.0
    ):
        super(SequentialAnomalyLoss, self).__init__()
        self.class_weights = class_weights
        self.binary_weight = binary_weight
        self.multiclass_weight = multiclass_weight
        self.reconstruction_weight = reconstruction_weight
        self.gamma = gamma
        
        # Create binary and multi-class focal losses
        self.binary_focal_loss = FocalLoss(alpha=0.25, gamma=gamma)
        
        if class_weights is not None:
            self.multiclass_focal_loss = FocalLoss(alpha=class_weights, gamma=gamma)
        else:
            self.multiclass_focal_loss = FocalLoss(gamma=gamma)
    
    def forward(self, model_output, labels):
        # Extract components from model output
        binary_logits = model_output['binary_logits']  # [batch_size, seq_len, 1]
        multiclass_logits = model_output['multiclass_logits']  # [batch_size, seq_len, 5]
        mse_per_timestep = model_output['mse_per_timestep']  # [batch_size, seq_len]
        
        # Create binary labels from multiclass labels (labels > 0 means anomaly)
        binary_labels = (labels > 0).float()
        
        # Reshape logits and labels for loss calculation
        batch_size, seq_len = labels.size()
        
        # 1. Reconstruction Loss
        reconstruction_loss = torch.mean(mse_per_timestep)
        
        # 2. Binary Classification Loss - Fix reshape operation
        binary_logits_flat = binary_logits.reshape(-1, 1)  # [batch_size * seq_len, 1]
        binary_labels_flat = binary_labels.reshape(-1)  # [batch_size * seq_len]
        binary_loss = self.binary_focal_loss(binary_logits_flat, binary_labels_flat)
        
        # 3. Multi-class Classification Loss
        multiclass_logits_flat = multiclass_logits.reshape(-1, 5)  # [batch_size * seq_len, 5]
        multiclass_labels_flat = labels.reshape(-1)  # [batch_size * seq_len]
        multiclass_loss = self.multiclass_focal_loss(multiclass_logits_flat, multiclass_labels_flat)
        
        # Combine all losses with their respective weights
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.binary_weight * binary_loss +
            self.multiclass_weight * multiclass_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'binary_loss': binary_loss,
            'multiclass_loss': multiclass_loss
        }

def train_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler=None,
    num_epochs=50,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    early_stopping_patience=10,
    class_weights=None,
    binary_weight=1.0,
    multiclass_weight=1.0,
    reconstruction_weight=1.0
):
    """
    Train the model with early stopping and learning rate scheduling
    """
    # Create loss function
    criterion = SequentialAnomalyLoss(
        class_weights=class_weights,
        binary_weight=binary_weight,
        multiclass_weight=multiclass_weight,
        reconstruction_weight=reconstruction_weight
    )
    
    # Initialize tracking variables
    best_val_loss = float('inf')
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
        train_multiclass_loss = 0.0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        
        for batch in train_progress:
            numerical_data = batch['numerical_data'].to(device)
            categorical_data = batch['categorical_data'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(numerical_data, categorical_data)
            
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
            train_multiclass_loss += loss_dict['multiclass_loss'].item()
            
            # Update progress bar with current losses
            train_progress.set_postfix({
                'loss': f"{total_loss.item():.4f}", 
                'rec': f"{loss_dict['reconstruction_loss'].item():.4f}",
                'bin': f"{loss_dict['binary_loss'].item():.4f}",
                'mul': f"{loss_dict['multiclass_loss'].item():.4f}"
            })
        
        # Calculate average training losses
        num_batches = len(train_loader)
        train_loss /= num_batches
        train_rec_loss /= num_batches
        train_binary_loss /= num_batches
        train_multiclass_loss /= num_batches
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_rec_loss = 0.0
        val_binary_loss = 0.0
        val_multiclass_loss = 0.0
        
        all_preds = []
        all_labels = []
        all_binary_preds = []
        all_binary_labels = []
        
        val_progress = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
        
        with torch.no_grad():
            for batch in val_progress:
                numerical_data = batch['numerical_data'].to(device)
                categorical_data = batch['categorical_data'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(numerical_data, categorical_data)
                
                # Calculate loss
                loss_dict = criterion(outputs, labels)
                total_loss = loss_dict['total_loss']
                
                # Get predictions for each timestep
                binary_probs = outputs['binary_probs'].squeeze(-1)  # [batch_size, seq_len]
                binary_preds = (binary_probs >= 0.5).float()
                
                multiclass_preds = torch.argmax(outputs['multiclass_logits'], dim=2)  # [batch_size, seq_len]
                
                # Create binary labels
                binary_labels = (labels > 0).float()
                
                # Collect predictions and labels for metrics
                all_preds.append(multiclass_preds.cpu())
                all_labels.append(labels.cpu())
                all_binary_preds.append(binary_preds.cpu())
                all_binary_labels.append(binary_labels.cpu())
                
                # Track losses
                val_loss += total_loss.item()
                val_rec_loss += loss_dict['reconstruction_loss'].item()
                val_binary_loss += loss_dict['binary_loss'].item()
                val_multiclass_loss += loss_dict['multiclass_loss'].item()
                
                # Update progress bar
                val_progress.set_postfix({
                    'loss': f"{total_loss.item():.4f}", 
                    'rec': f"{loss_dict['reconstruction_loss'].item():.4f}",
                    'bin': f"{loss_dict['binary_loss'].item():.4f}",
                    'mul': f"{loss_dict['multiclass_loss'].item():.4f}"
                })
        
        # Calculate average validation losses
        num_val_batches = len(test_loader)
        val_loss /= num_val_batches
        val_rec_loss /= num_val_batches
        val_binary_loss /= num_val_batches
        val_multiclass_loss /= num_val_batches
        val_losses.append(val_loss)
        
        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_binary_preds = torch.cat(all_binary_preds, dim=0).numpy()
        all_binary_labels = torch.cat(all_binary_labels, dim=0).numpy()
        
        # Flatten for metrics calculation
        all_preds_flat = all_preds.reshape(-1)
        all_labels_flat = all_labels.reshape(-1)
        all_binary_preds_flat = all_binary_preds.reshape(-1)
        all_binary_labels_flat = all_binary_labels.reshape(-1)
        
        # Calculate binary metrics
        binary_acc = np.mean(all_binary_preds_flat == all_binary_labels_flat)
        binary_f1 = f1_score(all_binary_labels_flat, all_binary_preds_flat)
        
        # Calculate multiclass metrics
        multiclass_acc = np.mean(all_preds_flat == all_labels_flat)
        
        # Display results
        print(f"\nValidation Results (Epoch {epoch+1}):")
        print(f"Overall Loss: {val_loss:.4f} (Rec: {val_rec_loss:.4f}, "
              f"Bin: {val_binary_loss:.4f}, Mul: {val_multiclass_loss:.4f})")
        print(f"Binary Detection - Accuracy: {binary_acc:.4f}, F1: {binary_f1:.4f}")
        print(f"Multiclass Detection - Accuracy: {multiclass_acc:.4f}")
        
        # Print confusion matrices
        binary_cm = confusion_matrix(all_binary_labels_flat, all_binary_preds_flat)
        print("\nBinary Confusion Matrix (Normal vs Anomaly):")
        print("                  Predicted")
        print("                Normal  Anomaly")
        print(f"Actual Normal   {binary_cm[0][0]:<8} {binary_cm[0][1]:<8}")
        print(f"Actual Anomaly  {binary_cm[1][0]:<8} {binary_cm[1][1]:<8}")

        # Calculate and display multiclass confusion matrix
        multi_cm = confusion_matrix(all_labels_flat, all_preds_flat)
        print("\nMulticlass Confusion Matrix:")
        class_names = ['Norm', 'UReTx', 'MReTx', 'NoDReTx', 'MaxReTx']
        print("            Predicted")
        print("            " + "  ".join(f"{name:<7}" for name in class_names))

        for i, row in enumerate(multi_cm):
            row_str = " ".join(f"{cell:<7}" for cell in row)
            print(f"Actual {class_names[i]:<4} {row_str}")

        # Calculate class counts for detection rate
        class_total = np.bincount(all_labels_flat.astype(int), minlength=5)
        class_correct = np.zeros(5)
        for i in range(5):
            class_correct[i] = np.sum((all_preds_flat == i) & (all_labels_flat == i))

        # Detection rate by class (recall)
        print("\nDetection Rate by Class:")
        for i in range(5):
            if class_total[i] > 0:
                true_pred = class_correct[i]
                total = class_total[i]
                print(f"Class {i} ({class_names[i]}): {int(true_pred)}/{int(total)} = {true_pred/total:.2%}")
            else:
                print(f"Class {i} ({class_names[i]}): 0/0 = 0.00%")

        # Print per-class metrics
        print("\nPer-class metrics:")
        for i in range(5):
            if class_total[i] > 0:
                accuracy = class_correct[i] / class_total[i]
                # Calculate precision, recall, and F1 for this class
                true_binary = (all_labels_flat == i).astype(int)
                pred_binary = (all_preds_flat == i).astype(int)
                
                # Handle potential division by zero
                precision = np.sum((pred_binary == 1) & (true_binary == 1)) / max(np.sum(pred_binary == 1), 1)
                recall = np.sum((pred_binary == 1) & (true_binary == 1)) / max(np.sum(true_binary == 1), 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-10)
                
                print(f"Class {i} ({class_names[i]}): "
                    f"Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}, "
                    f"Count={int(class_total[i])}")
            else:
                print(f"Class {i} ({class_names[i]}): No samples")
        
        # Update learning rate scheduler if provided
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with validation loss: {val_loss:.4f}")
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
            
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    return model, train_losses, val_losses

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
    all_multiclass_preds = []
    all_binary_preds = []
    all_labels = []
    all_binary_probs = []
    
    # Track attention weights and reconstruction errors
    all_numerical_attn = []
    all_temporal_attn = []
    all_instance_attn = []
    all_mse = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            numerical_data = batch['numerical_data'].to(device)
            categorical_data = batch['categorical_data'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(numerical_data, categorical_data)
            
            # Get binary predictions
            binary_probs = outputs['binary_probs'].squeeze(-1)  # [batch_size, seq_len]
            binary_preds = (binary_probs >= 0.5).float()
            
            # Get multiclass predictions
            multiclass_preds = torch.argmax(outputs['multiclass_logits'], dim=2)  # [batch_size, seq_len]
            
            # Store predictions and true labels
            all_multiclass_preds.append(multiclass_preds.cpu())
            all_binary_preds.append(binary_preds.cpu())
            all_labels.append(labels.cpu())
            all_binary_probs.append(binary_probs.cpu())
            
            # Store attention weights and reconstruction errors
            all_numerical_attn.append(outputs['numerical_attn_weights'].cpu())
            if outputs['temporal_attn_weights'] is not None:  # For GRU model
                all_temporal_attn.append(outputs['temporal_attn_weights'].cpu())
            all_instance_attn.append(outputs['instance_attn_weights'].cpu())
            all_mse.append(outputs['mse_per_timestep'].cpu())
    
    # Concatenate all predictions and labels
    all_multiclass_preds = torch.cat(all_multiclass_preds, dim=0).numpy()
    all_binary_preds = torch.cat(all_binary_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_binary_probs = torch.cat(all_binary_probs, dim=0).numpy()
    
    # Flatten for metrics calculation
    all_multiclass_preds_flat = all_multiclass_preds.reshape(-1)
    all_binary_preds_flat = all_binary_preds.reshape(-1)
    all_labels_flat = all_labels.reshape(-1)
    all_binary_probs_flat = all_binary_probs.reshape(-1)
    all_binary_labels_flat = (all_labels_flat > 0).astype(int)
    
    # Calculate binary classification metrics
    binary_accuracy = np.mean(all_binary_preds_flat == all_binary_labels_flat)
    binary_f1 = f1_score(all_binary_labels_flat, all_binary_preds_flat)
    
    # Calculate multi-class metrics
    multiclass_accuracy = np.mean(all_multiclass_preds_flat == all_labels_flat)
    
    # Create confusion matrices
    binary_cm = confusion_matrix(all_binary_labels_flat, all_binary_preds_flat)
    multiclass_cm = confusion_matrix(all_labels_flat, all_multiclass_preds_flat)
    
    # Calculate ROC AUC for binary classification
    binary_auc = roc_auc_score(all_binary_labels_flat, all_binary_probs_flat)
    
    print("\nFinal Evaluation Results:")
    print("Binary Classification Results:")
    print(f"Accuracy: {binary_accuracy:.4f}")
    print(f"F1 Score: {binary_f1:.4f}")
    print(f"ROC AUC: {binary_auc:.4f}")
    print(f"Confusion Matrix:\n{binary_cm}")
    
    print("\nMulti-class Classification Results:")
    print(f"Accuracy: {multiclass_accuracy:.4f}")
    print(f"Confusion Matrix:\n{multiclass_cm}")
    
    # Detailed classification report
    class_names = ['Normal', 'Unnecessary Retx', 'Missing Retx', 'New Data No Retx', 'Max Retx Achieved']
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels_flat, all_multiclass_preds_flat, target_names=class_names))
    
    # Analyze feature attention
    all_numerical_attn = torch.cat(all_numerical_attn, dim=0)
    mean_numerical_attn = torch.mean(all_numerical_attn, dim=(0, 1))  # Average over batch and sequence
    
    print("\nAverage Feature Attention:")
    numerical_features = model.feature_manager.numerical_features
    for i, feature in enumerate(numerical_features):
        print(f"{feature}: {mean_numerical_attn[i]:.4f}")
    
    return {
        'binary_accuracy': binary_accuracy,
        'binary_f1': binary_f1,
        'binary_auc': binary_auc,
        'multiclass_accuracy': multiclass_accuracy,
        'binary_cm': binary_cm,
        'multiclass_cm': multiclass_cm,
        'binary_preds': all_binary_preds,
        'multiclass_preds': all_multiclass_preds,
        'true_labels': all_labels,
        'binary_probs': all_binary_probs,
        'mean_numerical_attn': mean_numerical_attn.numpy(),
        'all_mse': [x.numpy() for x in all_mse],
        'classification_report': classification_report(all_labels_flat, all_multiclass_preds_flat, 
                                                      target_names=class_names, output_dict=True)
    }

if __name__ == "__main__":
    # Import dataset module
    from dataset import create_data_loaders
    
    # Configuration
    config = {
        'train_parquet_path': 'unscaled_pdsch_val.parquet',
        'test_parquet_path': 'unscaled_pdsch_val_min.parquet',
        'numerical_features': ['SFN', 'Slot', 'MCS', 'ReTx'],
        'categorical_features': ['HARQ', 'CRC', 'NDI'],
        'categorical_dims': [16, 2, 2],  # HARQ: 0-15, CRC & NDI: 0-1
        'seq_len': 20,
        'batch_size': 64,
        'num_workers': 0 if torch.cuda.is_available() else min(os.cpu_count(), 4),
        'sample_fraction': 1.0,
        'hidden_dim': 128,
        'embedding_dim': 8,
        'num_layers': 2,
        'dropout': 0.3,
        'model_type': 'bigru',  # 'bigru' or 'transformer'
        'num_heads': 4,  # Only for transformer
        'learning_rate': 0.001,
        'num_epochs': 50,
        'early_stopping_patience': 10,
        'binary_weight': 1.0,
        'multiclass_weight': 0.5,
        'reconstruction_weight': 1.0
    }
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, test_loader, norm_stats = create_data_loaders(
        train_parquet_path=config['train_parquet_path'],
        test_parquet_path=config['test_parquet_path'],
        numerical_features=config['numerical_features'],
        categorical_features=config['categorical_features'],
        categorical_dims=config['categorical_dims'],
        seq_len=config['seq_len'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sample_fraction=config['sample_fraction']
    )
    
    # Get class weights from the training dataset
    class_weights = train_loader.dataset.class_weights.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Create feature manager
    feature_manager = FeatureManager(
        numerical_features=config['numerical_features'],
        categorical_features=config['categorical_features'],
        categorical_dims=config['categorical_dims']
    )
    
    print(f"Feature configuration:\n{feature_manager}")
    
    # Create model
    print(f"Creating {config['model_type']} model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if config['model_type'] == 'bigru':
        model = BiGRUAnomalyDetector(
            numerical_feature_dim=len(config['numerical_features']),
            categorical_dims=config['categorical_dims'],
            feature_manager=feature_manager,
            hidden_dim=config['hidden_dim'],
            embedding_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            device=device
        ).to(device)
    else:  # transformer
        model = TransformerAnomalyDetector(
            numerical_feature_dim=len(config['numerical_features']),
            categorical_dims=config['categorical_dims'],
            feature_manager=feature_manager,
            hidden_dim=config['hidden_dim'],
            embedding_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
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
        class_weights=class_weights,
        binary_weight=config['binary_weight'],
        multiclass_weight=config['multiclass_weight'],
        reconstruction_weight=config['reconstruction_weight']
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
    model_filename = f"anomaly_detector_{config['model_type']}_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'evaluation_results': {k: v for k, v in evaluation_results.items() 
                              if not isinstance(v, np.ndarray) or v.size < 1000}
    }, model_filename)
    
    print(f"Model training and evaluation complete! Model saved as {model_filename}")