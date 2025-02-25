import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class WeightedFocalLoss(nn.Module):
    def __init__(self, weights=None, gamma=2.0, alpha=0.25, reduction='mean'):
        """
        Weighted Focal Loss for handling class imbalance in classification.
        
        Args:
            weights: Class weights tensor [num_classes]
            gamma: Focusing parameter for hard examples (higher value = more focus)
            alpha: Weighting factor for positive vs negative examples
            reduction: 'mean', 'sum', or 'none'
        """
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.weights = weights
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted logits [batch_size, seq_len, num_classes] or [batch_size*seq_len, num_classes]
            targets: Target labels [batch_size, seq_len] or [batch_size*seq_len]
        """
        # Handle different input shapes
        if inputs.dim() == 3:
            batch_size, seq_len, num_classes = inputs.shape
            inputs = inputs.reshape(-1, num_classes)
            targets = targets.reshape(-1)
        
        # Filter out ignored indices
        valid_mask = targets >= 0
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]
        
        if len(targets) == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Apply weights if provided
        if self.weights is not None:
            weights = self.weights.to(inputs.device)
            weights = weights[targets]
        else:
            weights = torch.ones_like(targets, device=inputs.device, dtype=torch.float32)
        
        # Compute softmax probabilities
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        
        # Get probability for target class
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight term
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Apply alpha weighting for positive/negative examples
        alpha_weight = torch.ones_like(targets, device=inputs.device, dtype=torch.float32)
        alpha_weight[targets > 0] = self.alpha  # Anomaly classes
        alpha_weight[targets == 0] = 1 - self.alpha  # Normal class
        
        # Compute final loss
        loss = -weights * alpha_weight * focal_weight * log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class FeatureEmbedding(nn.Module):
    """
    Hybrid feature embedding layer that handles both categorical and numerical features.
    """
    def __init__(self, feature_ranges, embedding_dims=None):
        """
        Args:
            feature_ranges: List of max values for each feature
            embedding_dims: Optional list of embedding dimensions for categorical features
        """
        super(FeatureEmbedding, self).__init__()
        self.feature_ranges = feature_ranges
        
        # Define which features are categorical vs numerical
        # For PDSCH data: [SFN, Slot, HARQ, MCS, CRC, ReTx, NDI]
        self.categorical_indices = [2, 4, 6]  # HARQ, CRC, NDI
        self.numerical_indices = [0, 1, 3, 5]  # SFN, Slot, MCS, ReTx
        
        # Determine embedding dimensions if not provided
        if embedding_dims is None:
            self.embedding_dims = []
            for i, range_val in enumerate(feature_ranges):
                if i in self.categorical_indices:
                    # Rule of thumb: min(50, (cardinality+1)//2)
                    dim = min(50, (range_val + 2) // 2)
                    self.embedding_dims.append(dim)
                else:
                    self.embedding_dims.append(1)  # Numerical features stay 1D
        else:
            self.embedding_dims = embedding_dims
        
        # Create embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(range_val + 1, self.embedding_dims[i])
            for i, range_val in enumerate(feature_ranges)
            if i in self.categorical_indices
        ])
        
        # Track embedding index mapping
        self.embedding_map = {idx: i for i, idx in enumerate(self.categorical_indices)}
        
        # Calculate total dimension after embedding
        self.output_dim = sum(self.embedding_dims[i] if i in self.categorical_indices 
                              else 1 for i in range(len(feature_ranges)))
    
    def forward(self, x):
        """
        Args:
            x: Input features [batch_size, seq_len, num_features]
        
        Returns:
            embedded: Combined embedded features [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, num_features = x.shape
        
        # Process features
        embeddings = []
        
        for i in range(num_features):
            if i in self.categorical_indices:
                # Get feature values and ensure they're valid indices
                feature_vals = torch.clamp(x[:, :, i].long(), 0, self.feature_ranges[i])
                # Apply embedding
                embedded = self.embeddings[self.embedding_map[i]](feature_vals)
                embeddings.append(embedded)
            else:
                # Normalize numerical features
                normalized = (x[:, :, i] / self.feature_ranges[i]).unsqueeze(-1)
                embeddings.append(normalized)
        
        # Concatenate all features
        return torch.cat(embeddings, dim=-1)


class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to input sequences to provide sequence order information.
    """
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        """
        Args:
            d_model: Dimension of the input embeddings
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sinusoidal positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (persistent state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Output with positional encoding added
        """
        # Add positional encoding and apply dropout
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalAttention(nn.Module):
    """
    Self-attention module for capturing temporal dependencies
    """
    def __init__(self, input_dim, dropout=0.1):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(input_dim, dtype=torch.float32))
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
        
        Returns:
            attention_output: Attended features [batch_size, seq_len, input_dim]
        """
        # Compute query, key, value projections
        query = self.query(x)  # [batch_size, seq_len, input_dim]
        key = self.key(x)      # [batch_size, seq_len, input_dim]
        value = self.value(x)  # [batch_size, seq_len, input_dim]
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale  # [batch_size, seq_len, seq_len]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, value)  # [batch_size, seq_len, input_dim]
        
        return attention_output


class SequentialAutoencoder(nn.Module):
    """
    Autoencoder for sequential anomaly detection that preserves temporal dependencies.
    Only trained on normal data.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2, dropout=0.1):
        """
        Args:
            input_dim: Input feature dimension after embedding
            hidden_dim: Hidden dimension size
            latent_dim: Latent space dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(SequentialAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Positional encoding for sequential information
        self.positional_encoding = PositionalEncoding(input_dim, dropout=dropout)
        
        # Encoder network (bidirectional LSTM)
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism to maintain temporal dependencies
        self.attention = TemporalAttention(hidden_dim * 2, dropout=dropout)
        
        # Projection to latent space
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder network
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )
        
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, input_dim)
    
    def encode(self, x):
        """
        Encode input features to latent representation
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
        
        Returns:
            z: Latent representation [batch_size, seq_len, latent_dim]
        """
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Encode with LSTM
        encoder_output, _ = self.encoder_lstm(x)
        
        # Apply attention to capture temporal dependencies
        attended = self.attention(encoder_output)
        
        # Project to latent space
        z = self.to_latent(attended)
        
        return z
    
    def decode(self, z):
        """
        Decode latent representation back to input space
        
        Args:
            z: Latent representation [batch_size, seq_len, latent_dim]
        
        Returns:
            reconstruction: Reconstructed features [batch_size, seq_len, input_dim]
        """
        # Project from latent space
        h = self.from_latent(z)
        
        # Decode with LSTM
        decoder_output, _ = self.decoder_lstm(h)
        
        # Project to output space
        reconstruction = self.output_proj(decoder_output)
        
        return reconstruction
    
    def forward(self, x):
        """
        Forward pass through the autoencoder
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
        
        Returns:
            reconstruction: Reconstructed features [batch_size, seq_len, input_dim]
            z: Latent representation [batch_size, seq_len, latent_dim]
        """
        # Encode
        z = self.encode(x)
        
        # Decode
        reconstruction = self.decode(z)
        
        return reconstruction, z


class HierarchicalClassifier(nn.Module):
    """
    Classifier for telecommunication anomaly detection that directly classifies instances
    into normal or specific anomaly types.
    """
    def __init__(self, input_dim, hidden_dim, num_anomaly_classes=5, dropout=0.1):
        """
        Args:
            input_dim: Input feature dimension (original + reconstructed + errors)
            hidden_dim: Hidden dimension size
            num_anomaly_classes: Number of anomaly classes (including "none of them")
            dropout: Dropout probability
        """
        super(HierarchicalClassifier, self).__init__()
        
        # Number of anomaly classes + 1 for normal
        self.num_classes = num_anomaly_classes + 1
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(input_dim, dropout=dropout)
        
        # Feature extraction network
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention mechanism to focus on important parts of the sequence
        self.attention = TemporalAttention(hidden_dim * 2, dropout=dropout)
        
        # Direct classification head for all classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the classifier
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
                (combined original, reconstructed, and errors)
        
        Returns:
            logits: Classification logits [batch_size, seq_len, num_classes]
        """
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(features)
        
        # Apply attention to focus on relevant parts of the sequence
        attended = self.attention(lstm_out)
        
        # Direct classification into all classes
        logits = self.classifier(attended)
        
        return logits


class TwoPhaseModel(nn.Module):
    """
    Complete two-phase model with autoencoder and hierarchical classifier.
    """
    def __init__(self, feature_ranges, params=None):
        """
        Args:
            feature_ranges: List of max values for each feature
            params: Dictionary of model parameters
        """
        super(TwoPhaseModel, self).__init__()
        
        # Set default parameters if not provided
        if params is None:
            params = {}
        
        # Extract parameters with defaults
        embedding_dim = params.get('embedding_dim', 16)
        hidden_dim = params.get('hidden_dim', 64)
        latent_dim = params.get('latent_dim', 32)
        dropout = params.get('dropout', 0.2)
        num_anomaly_classes = params.get('num_anomaly_classes', 5)  # 4 anomaly types + "none of them"
        
        # Feature embedding layer
        self.feature_embedding = FeatureEmbedding(
            feature_ranges, 
            embedding_dims=[embedding_dim] * len(feature_ranges)
        )
        embedded_dim = self.feature_embedding.output_dim
        
        # Autoencoder for reconstruction
        self.autoencoder = SequentialAutoencoder(
            input_dim=embedded_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout
        )
        
        # Combined feature dimension for classifier (original + reconstructed + errors)
        classifier_input_dim = embedded_dim * 3
        
        # Hierarchical classifier
        self.classifier = HierarchicalClassifier(
            input_dim=classifier_input_dim,
            hidden_dim=hidden_dim,
            num_anomaly_classes=num_anomaly_classes,
            dropout=dropout
        )
        
        # Track which phase is being trained
        self.train_autoencoder = True
        self.train_classifier = False
    
    def get_loss_weights(self, class_counts):
        """
        Compute class weights based on inverse frequency
        
        Args:
            class_counts: Dictionary mapping class indices to counts
        
        Returns:
            weights: Class weights tensor
        """
        total = sum(class_counts.values())
        # Compute inverse frequency and normalize
        weights = torch.zeros(len(class_counts))
        for cls, count in class_counts.items():
            weights[cls] = 1.0 / (count / total)
        
        # Normalize weights to sum to len(class_counts)
        weights = weights * len(class_counts) / weights.sum()
        return weights
    
    def forward(self, x, return_latents=False):
        """
        Forward pass through the entire model
        
        Args:
            x: Input features [batch_size, seq_len, num_features]
            return_latents: Whether to return latent representations
        
        Returns:
            logits: Classification logits [batch_size, seq_len, num_classes]
            (optionally) latents: Latent representations
        """
        # Embed features
        embedded = self.feature_embedding(x)
        
        # Get reconstruction from autoencoder
        reconstructed, latents = self.autoencoder(embedded)
        
        # Compute reconstruction errors per feature
        errors = torch.abs(embedded - reconstructed)
        
        # Combine original features, reconstructed features, and errors
        combined_features = torch.cat([embedded, reconstructed, errors], dim=-1)
        
        # Get classification logits
        logits = self.classifier(combined_features)
        
        if return_latents:
            return logits, latents
        else:
            return logits
    
    def freeze_autoencoder(self):
        """Freeze autoencoder parameters for classifier training phase"""
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.train_autoencoder = False
        self.train_classifier = True
    
    def freeze_classifier(self):
        """Freeze classifier parameters for autoencoder training phase"""
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.train_autoencoder = True
        self.train_classifier = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters for fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True
        self.train_autoencoder = True
        self.train_classifier = True