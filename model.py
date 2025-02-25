import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable

# Constants
EPS = 1e-8

# Update in the FeatureEmbedding class in model.py

class FeatureEmbedding(nn.Module):
    """
    Embeds categorical features into continuous space with safety checks
    """
    def __init__(self, feature_ranges, embedding_dim=8):
        super().__init__()
        self.feature_ranges = feature_ranges
        self.embeddings = nn.ModuleList([
            nn.Embedding(range_size + 1, min(embedding_dim, (range_size + 1) // 2 + 1))
            for range_size in feature_ranges
        ])
        self.output_dim = sum(min(embedding_dim, (range_size + 1) // 2 + 1) for range_size in feature_ranges)
        
    def forward(self, x):
        # x shape: [B, T, F]
        B, T, F = x.shape
        
        # Ensure x is of integer type
        x = x.long()
        
        # Apply embeddings for each feature with safety checks
        embedded_features = []
        for i, embedding in enumerate(self.embeddings):
            feature_values = x[:, :, i]
            
            # Safety check: clamp values to valid range for this feature
            max_valid_index = self.feature_ranges[i]
            feature_values = torch.clamp(feature_values, 0, max_valid_index)
            
            # Get embedding
            embedded = embedding(feature_values)  # [B, T, embedding_dim]
            embedded_features.append(embedded)
            
        # Concatenate all embedded features
        return torch.cat(embedded_features, dim=2)  # [B, T, sum(embedding_dims)]


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
            pe[:, odd_i] = torch.cos(position * div_term[:len(odd_i)])
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [B, T, d_model]
        return x + self.pe[:, :x.size(1), :]


class AutoregressiveTemporalBlock(nn.Module):
    """
    Processes temporal data with bidirectional LSTM and self-attention
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional
            num_heads=4,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2)
        )
        self.output_dim = hidden_dim * 2
        
    def forward(self, x, prev_state=None):
        # x shape: [B, T, input_dim]
        
        # LSTM processing
        if prev_state is not None:
            output, state = self.lstm(x, prev_state)
        else:
            output, state = self.lstm(x)
        
        # Self-attention
        attn_output, _ = self.attention(
            output.transpose(0, 1),  # [T, B, hidden_dim*2]
            output.transpose(0, 1),
            output.transpose(0, 1)
        )
        attn_output = attn_output.transpose(0, 1)  # [B, T, hidden_dim*2]
        
        # Residual connection and normalization
        output = self.norm1(output + attn_output)
        
        # Feed-forward network
        ff_output = self.ff(output)
        output = self.norm2(output + ff_output)
        
        return output, state


class TemporalVAE(nn.Module):
    """
    Variational Autoencoder with temporal components for representation learning
    """
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, num_layers=2, dropout=0.2):
        super().__init__()
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Latent space projections
        self.mu_proj = nn.Linear(hidden_dim * 2, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * 2)
        self.decoder_lstm = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
    def encode(self, x):
        # x shape: [B, T, input_dim]
        output, (h_n, _) = self.encoder_lstm(x)
        
        # Use the hidden state from the last layer
        h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)  # Concatenate forward and backward
        
        # Project to latent space
        mu = self.mu_proj(h_n)
        logvar = self.logvar_proj(h_n)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, seq_len):
        # z shape: [B, latent_dim]
        hidden = self.latent_to_hidden(z)
        
        # Repeat for sequence length
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [B, T, hidden_dim*2]
        
        # Decode
        output, _ = self.decoder_lstm(hidden)
        reconstruction = self.output_proj(output)
        
        return reconstruction
    
    def forward(self, x):
        # x shape: [B, T, input_dim]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, x.size(1))
        
        return reconstruction, mu, logvar, z


class AnomalyClassifier(nn.Module):
    """
    Classifier for anomaly detection that uses both raw features and VAE embeddings
    """
    def __init__(self, input_dim, vae_latent_dim, hidden_dim=64, num_classes=5, num_layers=1, dropout=0.2):
        super().__init__()
        
        # Combine raw features and VAE embeddings
        combined_dim = input_dim + vae_latent_dim
        
        # Temporal processing
        self.temporal_block = AutoregressiveTemporalBlock(
            combined_dim, 
            hidden_dim, 
            num_layers,
            dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.temporal_block.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, vae_embedding):
        # x shape: [B, T, input_dim]
        # vae_embedding shape: [B, vae_latent_dim]
        
        # Repeat VAE embedding for each time step and concatenate with input
        B, T, _ = x.shape
        vae_embedding = vae_embedding.unsqueeze(1).repeat(1, T, 1)  # [B, T, vae_latent_dim]
        combined = torch.cat([x, vae_embedding], dim=2)  # [B, T, input_dim + vae_latent_dim]
        
        # Process with temporal block
        output, _ = self.temporal_block(combined)
        
        # Classify each time step
        logits = self.classifier(output)  # [B, T, num_classes]
        
        return logits
        

class TwoPhaseModel(nn.Module):
    """
    Two-phase model combining VAE representation learning and supervised classification
    """
    def __init__(self, feature_ranges, embedding_dim=8, hidden_dim=64, 
                 latent_dim=32, num_classes=5, num_layers=2, dropout=0.2,
                 alpha=0.5, gamma=2.0, class_weights=None, beta=0.1):
        super().__init__()
        
        # Phase 1: Feature embedding and representation learning
        self.feature_embedding = FeatureEmbedding(feature_ranges, embedding_dim)
        self.positional_encoding = PositionalEncoding(self.feature_embedding.output_dim)
        
        self.vae = TemporalVAE(
            input_dim=self.feature_embedding.output_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Phase 2: Anomaly classification
        self.classifier = AnomalyClassifier(
            input_dim=self.feature_embedding.output_dim,
            vae_latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Parameters for Focal Loss
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.beta = beta  # Weight for VAE loss versus classification loss
        
    def forward(self, x, return_latents=False):
        # x shape: [B, T, F]
        
        # Phase 1: Embedding and VAE
        embedded = self.feature_embedding(x)
        embedded = self.positional_encoding(embedded)
        reconstruction, mu, logvar, z = self.vae(embedded)
        
        # Phase 2: Classification
        class_logits = self.classifier(embedded, z)
        
        if return_latents:
            return class_logits, z
        return class_logits
    
    def compute_vae_loss(self, x, reconstruction, mu, logvar):
        """
        Compute VAE loss (reconstruction + KL divergence)
        """
        # Reconstruction loss
        embedded = self.feature_embedding(x)
        embedded = self.positional_encoding(embedded)
        recon_loss = F.mse_loss(reconstruction, embedded)
        
        # KL divergence
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kld_loss
    
    def compute_classification_loss(self, logits, targets):
        """
        Compute focal loss for classification
        """
        return self.focal_loss(logits, targets)
    
    def focal_loss(self, logits, targets):
        """
        Compute the focal loss between logits and targets
        """
        # Get weights for each class
        weights = self.get_loss_weights(None) if self.class_weights is None else self.class_weights
        weights = weights.to(logits.device)
        
        # Reshape for loss calculation
        B, T, C = logits.shape
        logits = logits.view(-1, C)  # [B*T, C]
        targets = targets.view(-1)    # [B*T]
        
        # Focal loss calculation
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss)
        
        # Apply class weights
        class_weights = weights[targets]
        weighted_focal_loss = focal_loss * class_weights
        
        return weighted_focal_loss.mean()
    
    def get_loss_weights(self, counts):
        """
        Compute class weights based on class frequencies
        """
        if counts is None:
            # Default weights if counts not provided (equal weighting)
            return torch.ones(5)
        
        # Convert counts to tensor if it's a dictionary
        if isinstance(counts, dict):
            count_tensor = torch.zeros(len(counts))
            for class_id, count in counts.items():
                count_tensor[class_id] = count
            counts = count_tensor
        
        # Calculate inverse frequency weights and normalize
        total_samples = torch.sum(counts)
        class_weights = total_samples / (counts * len(counts) + EPS)
        
        # Normalize weights to sum to number of classes
        class_weights = class_weights * len(counts) / torch.sum(class_weights)
        return class_weights
    
    def inference(self, x, prev_states=None):
        """
        Forward pass with state tracking for inference over longer sequences
        """
        # Embedding
        embedded = self.feature_embedding(x)
        embedded = self.positional_encoding(embedded)
        
        # VAE encoding
        mu, logvar = self.vae.encode(embedded)
        z = self.vae.reparameterize(mu, logvar)
        
        # Classification with previous state if available
        if prev_states is not None:
            combined = torch.cat([embedded, z.unsqueeze(1).repeat(1, embedded.size(1), 1)], dim=2)
            output, new_states = self.temporal_block(combined, prev_states)
            logits = self.classifier(output)
            return logits, new_states
        else:
            # Standard forward pass
            return self.forward(x), None

    def get_anomaly_score(self, x):
        """
        Compute anomaly score based on reconstruction error and classification
        """
        # Get embeddings
        embedded = self.feature_embedding(x)
        embedded = self.positional_encoding(embedded)
        
        # VAE reconstruction
        reconstruction, mu, logvar, z = self.vae(embedded)
        
        # Reconstruction error as anomaly signal
        recon_error = F.mse_loss(reconstruction, embedded, reduction='none')
        recon_error = recon_error.mean(dim=2)  # Average across feature dimension
        
        # Classification probabilities
        class_logits = self.classifier(embedded, z)
        class_probs = F.softmax(class_logits, dim=2)
        
        # Probability of normal class (class 0)
        normal_probs = class_probs[:, :, 0]
        
        # Combined anomaly score: high reconstruction error or low normal probability
        combined_score = recon_error - torch.log(normal_probs + EPS)
        
        return combined_score, class_logits


class WeightedFocalLoss(nn.Module):
    """
    Focal loss with adjustable alpha and gamma parameters
    """
    def __init__(self, weights=None, alpha=0.5, gamma=2.0):
        super().__init__()
        self.weights = weights
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, T, C] logits
            targets: [B, T] class indices
        """
        # Reshape inputs and targets
        B, T, C = inputs.shape
        inputs = inputs.view(-1, C)  # [B*T, C]
        targets = targets.view(-1)    # [B*T]
        
        # Cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Focal loss calculation
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss)
        
        # Apply class weights if provided
        if self.weights is not None:
            weights = self.weights.to(focal_loss.device)
            class_weights = weights[targets]
            focal_loss = focal_loss * class_weights
        
        return focal_loss.mean()