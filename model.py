import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Constants
EPS = 1e-8

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input embeddings
    """
    def __init__(self, d_model, max_seq_len=200):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate positions
        even_i = torch.arange(0, d_model, 2)
        div_term = torch.exp(even_i.float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd positions that exist
        odd_i = torch.arange(1, d_model, 2)
        if len(odd_i) > 0:
            pe[:, 1::2] = torch.cos(position * div_term[:len(odd_i)])
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [B, T, d_model]
        return x + self.pe[:, :x.size(1), :]


class HybridFeatureEncoder(nn.Module):
    """
    Encodes both categorical and numerical features
    """
    def __init__(self, categorical_dims, embedding_dim=8):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0
        
        # Create embeddings for each categorical feature
        for feature, dim in categorical_dims.items():
            # Ensure minimum embedding dimension based on cardinality
            feat_emb_dim = min(embedding_dim, (dim + 1) // 2 + 1)
            self.embeddings[feature] = nn.Embedding(dim + 1, feat_emb_dim)
            total_embedding_dim += feat_emb_dim
            
        self.output_dim = total_embedding_dim
        
    def forward(self, categorical_features):
        # Process categorical features
        embedded = []
        for feature, embedding in self.embeddings.items():
            if feature in categorical_features:
                feature_tensor = categorical_features[feature]
                # Safety check - clamp to valid range
                feature_tensor = torch.clamp(feature_tensor, 0, embedding.num_embeddings - 1)
                embedded.append(embedding(feature_tensor))
            else:
                # Skip if feature not provided
                pass
                
        # Concatenate all embeddings
        return torch.cat(embedded, dim=2) if embedded else None


class BinaryTransformerEncoder(nn.Module):
    """
    Transformer encoder for binary anomaly detection
    """
    def __init__(self, input_dim, num_heads=4, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(input_dim)
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
    def forward(self, x):
        # Apply positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer encoder
        return self.transformer_encoder(x)


class BinaryAnomalyAttention(nn.Module):
    """
    Attention mechanism to focus on anomalous patterns
    """
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = torch.sqrt(torch.tensor(input_dim, dtype=torch.float32))
        
    def forward(self, x):
        # x shape: [B, T, input_dim]
        q = self.query(x)  # [B, T, input_dim]
        k = self.key(x)    # [B, T, input_dim]
        v = self.value(x)  # [B, T, input_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, T, T]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [B, T, T]
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)  # [B, T, input_dim]
        
        return context, attention_weights


class BinaryAnomalyDetector(nn.Module):
    """
    Binary anomaly detection model with hybrid feature modeling
    """
    def __init__(self, 
                numerical_dim, 
                categorical_dims, 
                embedding_dim=8,
                hidden_dim=64,
                latent_dim=32,
                num_layers=2,
                num_heads=4,
                dropout=0.2):
        super().__init__()
        
        # Feature encoders
        self.categorical_encoder = HybridFeatureEncoder(categorical_dims, embedding_dim)
        categorical_output_dim = self.categorical_encoder.output_dim
        
        # Combined dimension
        self.combined_dim = numerical_dim + categorical_output_dim
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM for sequence modeling with state tracking
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Transformer encoder for long-range dependencies
        self.transformer = BinaryTransformerEncoder(
            hidden_dim * 2,  # bidirectional
            num_heads=num_heads,
            hidden_dim=hidden_dim * 2,
            num_layers=num_layers - 1 if num_layers > 1 else 1,
            dropout=dropout
        )
        
        # Anomaly attention mechanism
        self.anomaly_attention = BinaryAnomalyAttention(hidden_dim * 2)
        
        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, numerical_features, categorical_features, hidden_state=None):
        # Process categorical features
        categorical_embedded = self.categorical_encoder(categorical_features)  # [B, T, cat_emb_dim]
        
        # Combine features
        combined = torch.cat([numerical_features, categorical_embedded], dim=2)  # [B, T, combined_dim]
        
        # Project to hidden dimension
        projected = self.feature_projection(combined)  # [B, T, hidden_dim]
        
        # Process with LSTM
        if hidden_state is not None:
            lstm_out, new_hidden = self.lstm(projected, hidden_state)
        else:
            lstm_out, new_hidden = self.lstm(projected)  # [B, T, hidden_dim*2]
        
        # Process with transformer for long-range dependencies
        transformer_out = self.transformer(lstm_out)  # [B, T, hidden_dim*2]
        
        # Apply anomaly attention
        context, _ = self.anomaly_attention(transformer_out)  # [B, T, hidden_dim*2]
        
        # Binary classification
        logits = self.classifier(context).squeeze(-1)  # [B, T]
        
        return logits, new_hidden
    
    def detect_anomalies(self, numerical_features, categorical_features, threshold=0.5):
        """
        Detect anomalies and return binary predictions
        """
        with torch.no_grad():
            logits, _ = self.forward(numerical_features, categorical_features)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float()
            return predictions, probabilities


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, targets):
        # logits: [B, T] or [B]
        # targets: [B, T] or [B]
        
        # Ensure proper shape
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)  # [B, 1]
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)  # [B, 1]
            
        # Binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        
        # Calculate focal weights
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class TimeSeriesAnomalyDetector(nn.Module):
    """
    Complete binary anomaly detection model with best practices for time series
    """
    def __init__(self,
                numerical_features,
                categorical_dims,
                embedding_dim=8,
                hidden_dim=128,
                latent_dim=64,
                num_layers=2, 
                dropout=0.3,
                alpha=0.75,
                gamma=2.0):
        super().__init__()
        
        # Dimensions
        self.numerical_dim = len(numerical_features)
        self.categorical_dims = categorical_dims
        
        # Core detector
        self.detector = BinaryAnomalyDetector(
            numerical_dim=self.numerical_dim,
            categorical_dims=self.categorical_dims,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Loss function
        self.focal_loss = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        
    def forward(self, numerical_features, categorical_features, hidden_state=None):
        return self.detector(numerical_features, categorical_features, hidden_state)
    
    def compute_loss(self, logits, targets):
        """Compute loss for training"""
        return self.focal_loss(logits, targets)
    
    def predict(self, numerical_features, categorical_features, threshold=0.5):
        """Make binary predictions"""
        logits, _ = self.forward(numerical_features, categorical_features)
        return torch.sigmoid(logits) > threshold
    
    def get_attention_weights(self, numerical_features, categorical_features):
        """Get attention weights for explainability"""
        with torch.no_grad():
            _, attention_weights = self.detector.anomaly_attention(
                self.detector.transformer(
                    self.detector.lstm(
                        self.detector.feature_projection(
                            torch.cat([
                                numerical_features, 
                                self.detector.categorical_encoder(categorical_features)
                            ], dim=2)
                        )
                    )[0]
                )
            )
            return attention_weights