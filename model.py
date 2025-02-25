import torch
from torch import nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based models."""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer to avoid being considered as model parameters
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class FeatureEmbedding(nn.Module):
    """Embeds categorical features into continuous representations."""
    def __init__(self, feature_ranges, embedding_dim=32):
        super().__init__()
        
        # Create separate embeddings for each feature based on its range
        self.embeddings = nn.ModuleList()
        self.feature_ranges = feature_ranges
        self.output_dim = len(feature_ranges) * embedding_dim
        
        for feature_range in feature_ranges:
            self.embeddings.append(nn.Embedding(feature_range + 1, embedding_dim))
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, num_features]
        Returns:
            Tensor, shape [batch_size, seq_len, num_features * embedding_dim]
        """
        # Ensure input is long tensor for embedding lookup
        x = x.long()
        
        # Process each feature through its embedding layer
        embedded_features = []
        for i, embedding in enumerate(self.embeddings):
            feature_val = x[:, :, i]
            # Clamp values to ensure they are within the valid embedding range
            clamped_val = torch.clamp(feature_val, 0, self.feature_ranges[i])
            embedded = embedding(clamped_val)  # [B, T, embedding_dim]
            embedded_features.append(embedded)
        
        # Concatenate all embeddings
        return torch.cat(embedded_features, dim=2)  # [B, T, num_features * embedding_dim]

class CausalSelfAttention(nn.Module):
    """Self-attention layer with causal masking to prevent attending to future tokens."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Single projection for Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            Tensor, shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        
        # Single projection and split into Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Create causal mask (lower triangular)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.reshape(batch_size, seq_len, self.d_model)
        
        # Final projection
        output = self.out_proj(context)
        return output

class FeatureFeatureAttention(nn.Module):
    """Attention mechanism for capturing relationships between features."""
    def __init__(self, d_model, num_features, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_features = num_features
        
        # Calculate feature dimension
        self.feature_dim = d_model // num_features
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.k_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.v_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.out_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            Tensor, shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        
        # Reshape to separate features
        x = x.view(batch_size, seq_len, self.num_features, self.feature_dim)
        
        # Project Q, K, V for each feature
        q = self.q_proj(x)  # [B, T, num_features, feature_dim]
        k = self.k_proj(x)  # [B, T, num_features, feature_dim]
        v = self.v_proj(x)  # [B, T, num_features, feature_dim]
        
        # Compute attention across features - keep feature_dim intact
        # Transpose q to get [B, T, feature_dim, num_features]
        q = q.transpose(2, 3)
        
        # Compute attention scores between features
        # [B, T, feature_dim, num_features] × [B, T, num_features, feature_dim] = [B, T, feature_dim, feature_dim]
        scores = torch.matmul(q, k) / math.sqrt(self.feature_dim)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)  # [B, T, feature_dim, feature_dim]
        attn_weights = self.dropout(attn_weights)
        
        # Transpose v to align with attention weights
        v = v.transpose(2, 3)  # [B, T, feature_dim, num_features]
        
        # Apply attention
        # [B, T, feature_dim, feature_dim] × [B, T, feature_dim, num_features] = [B, T, feature_dim, num_features]
        context = torch.matmul(attn_weights, v)
        
        # Transpose back
        context = context.transpose(2, 3)  # [B, T, num_features, feature_dim]
        
        # Final projection
        output = self.out_proj(context)
        
        # Reshape back to original dimensions
        output = output.reshape(batch_size, seq_len, self.d_model)
        
        return output

class FeedForward(nn.Module):
    """Feed-forward network with residual connection and layer normalization."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            Tensor, shape [batch_size, seq_len, d_model]
        """
        residual = x
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = x + residual
        x = self.norm(x)
        return x

class ClassificationHead(nn.Module):
    """Specialized classification head for each anomaly type."""
    def __init__(self, d_model, anomaly_type, dropout=0.1):
        super().__init__()
        self.anomaly_type = anomaly_type
        
        # Different architectures based on anomaly type
        if anomaly_type == "normal":
            self.layers = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            )
        elif anomaly_type in ["missing_retx", "new_data_no_retx"]:
            # For anomalies that need to track NDI and previous CRC
            self.layers = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 4, 1)
            )
        elif anomaly_type == "max_retx":
            # For anomalies related to high retransmission count
            self.layers = nn.Sequential(
                nn.Linear(d_model, d_model // 3),
                nn.LayerNorm(d_model // 3),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 3, 1)
            )
        elif anomaly_type == "unnecessary_retx":
            # For anomalies that need to check previous CRC
            self.layers = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 4, 1)
            )
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            Tensor, shape [batch_size, seq_len, 1]
        """
        return self.layers(x)

class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss to handle class imbalance
    Focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # weight for each class
        self.gamma = gamma  # focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Tensor of shape [B, C] or [B, T, C] where C is number of classes
            targets: Tensor of shape [B] or [B, T] with class indices
        """
        # Flatten if sequence data
        if inputs.dim() > 2:
            B, T, C = inputs.shape
            inputs = inputs.view(B * T, C)
            targets = targets.view(B * T)
        
        # Compute softmax
        inputs_softmax = F.softmax(inputs, dim=1)
        
        # Pick the values for the target classes
        pt = inputs_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight
        focal_weight = (1 - pt).pow(self.gamma)
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight
        
        # Compute loss
        loss = -focal_weight * torch.log(pt + 1e-8)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class Model(nn.Module):
    """
    Enhanced anomaly detection model for 5G PDSCH telecommunication data
    with embeddings, dual attention mechanism, and specialized classification heads.
    """
    def __init__(self, feature_ranges, hidden_dim=252, num_layers=4, dropout=0.2, num_classes=5):
        """
        Args:
            feature_ranges: List of integers representing the range of each feature
                           [SFN_range, Slot_range, HARQ_range, MCS_range, CRC_range, ReTx_range, NDI_range]
            hidden_dim: Size of hidden dimension
            num_layers: Number of transformer layers
            dropout: Dropout rate
            num_classes: Number of anomaly classes (including normal)
        """
        super().__init__()
        self.num_features = len(feature_ranges)
        self.feature_ranges = feature_ranges
        
        # Embedding layer for categorical features
        self.embedding = FeatureEmbedding(feature_ranges, embedding_dim=32)
        embedding_dim = self.embedding.output_dim
        
        # Projection to hidden dimension
        self.projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer layers with causal masking
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'time_attn': CausalSelfAttention(hidden_dim, num_heads=7, dropout=dropout),
                'feature_attn': FeatureFeatureAttention(hidden_dim, self.num_features, dropout=dropout),
                'ff': FeedForward(hidden_dim, hidden_dim * 4, dropout=dropout)
            }) for _ in range(num_layers)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Specialized classification heads
        self.classification_heads = nn.ModuleList([
            ClassificationHead(hidden_dim, anomaly_type, dropout=dropout)
            for anomaly_type in ["normal", "unnecessary_retx", "missing_retx", "new_data_no_retx", "max_retx"]
        ])
        
        # Final classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # HARQ tracking mechanism - learnable parameters for different HARQ IDs
        self.harq_trackers = nn.Parameter(torch.randn(16, hidden_dim // 8))  # 16 HARQ IDs
        self.harq_proj = nn.Linear(hidden_dim // 8, hidden_dim)
    
    def forward(self, x, return_latents=False):
        """
        Args:
            x: [B, T, num_features] input features.
            return_latents: if True, also return latent representations.
        Returns:
            class_logits: [B, T, num_classes] classification outputs.
            (optionally) latent: latent representations [B, T, hidden_dim].
        """
        batch_size, seq_len, _ = x.size()
        
        # Extract HARQ IDs for tracking
        harq_ids = torch.clamp(x[:, :, 2].long(), 0, 15)  # HARQ is at index 2
        
        # Embed categorical features
        embedded = self.embedding(x)  # [B, T, embedding_dim]
        
        # Project to hidden dimension
        x = self.projection(embedded)  # [B, T, hidden_dim]
        
        # Add positional encoding
        x = self.pos_encoding(x)  # [B, T, hidden_dim]
        
        # Process through transformer layers
        for layer in self.transformer_layers:
            # Time-based causal self-attention
            x_time = layer['time_attn'](x)
            x = x + x_time
            
            # Feature-feature attention
            x_feat = layer['feature_attn'](x)
            x = x + x_feat
            
            # Feed-forward
            x = layer['ff'](x)
        
        # Final layer norm
        latent = self.norm(x)  # [B, T, hidden_dim]
        
        # HARQ tracking - gather embeddings for each HARQ ID
        harq_embeddings = self.harq_trackers[harq_ids]  # [B, T, hidden_dim//8]
        harq_context = self.harq_proj(harq_embeddings)  # [B, T, hidden_dim]
        
        # Add HARQ context to latent representation
        enhanced_latent = latent + harq_context * 0.1  # Small contribution to avoid overwhelming
        
        # Apply specialized classification heads
        head_outputs = [head(enhanced_latent) for head in self.classification_heads]  # list of [B, T, 1]
        head_outputs = torch.cat(head_outputs, dim=2)  # [B, T, 5]
        
        # Apply final classifier
        class_logits = self.classifier(enhanced_latent)  # [B, T, num_classes]
        
        # Combine specialized head outputs with general classifier
        combined_logits = class_logits + head_outputs * 0.5  # [B, T, num_classes]
        
        if return_latents:
            return combined_logits, enhanced_latent
        else:
            return combined_logits
    
    def get_loss_weights(self, counts):
        """
        Calculate class weights inversely proportional to class frequencies,
        with smoothing to avoid extreme weights.
        
        Args:
            counts: Dictionary of class counts {class_idx: count}
        Returns:
            Tensor of class weights
        """
        total = sum(counts.values())
        weights = []
        
        for i in range(len(counts)):
            # Inverse frequency with smoothing
            count = counts.get(i, 1)
            weight = (total / (count + 100)) ** 0.5  # Square root for less extreme weights
            weights.append(weight)
        
        # Normalize weights
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum() * len(weights)
        
        return weights