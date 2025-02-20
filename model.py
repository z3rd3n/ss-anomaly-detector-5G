import torch
from torch import nn

class NumericalAutoencoder(nn.Module):
    def __init__(self, num_features, hidden_dim, dropout=0.1, proj_dim=32):
        """
        A CNN + Self-Attention based sequence autoencoder that operates on normalized numerical features.
        In addition to reconstruction (via one output head per feature), it projects the internal representation
        into a latent space for contrastive learning.
        
        Args:
            num_features: int, number of input features.
            hidden_dim: hidden dimension for the convolution and attention layers.
            dropout: dropout probability.
            proj_dim: dimension of the projected latent space.
        """
        super().__init__()
        self.num_features = num_features
        
        self.conv3 = nn.Conv1d(in_channels=num_features, out_channels=hidden_dim // 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=num_features, out_channels=hidden_dim // 2, kernel_size=5, padding=2)
        self.conv_combine = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
        
        self.attention = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dropout=dropout, batch_first=True)
        
        # One output head per feature.
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_features)
        ])
        # Projection head for contrastive learning.
        self.contrast_head = nn.Linear(hidden_dim, proj_dim)
    
    def forward(self, x, return_latents=False):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape [B, T, num_features] (normalized values).
            return_latents: If True, also returns the projected latent representations (per timestamp).
        
        Returns:
            outputs: List of predictions for each feature, each of shape [B, T, 1].
            If return_latents is True, also returns proj_latents: Tensor of shape [B, T, proj_dim].
        """
        batch_size, seq_len, _ = x.size()
        x_conv = x.transpose(1, 2)  # [B, num_features, T]
        conv_out_3 = self.conv3(x_conv)
        conv_out_5 = self.conv5(x_conv)
        conv_cat = torch.cat([conv_out_3, conv_out_5], dim=1)
        combined = self.conv_combine(conv_cat)
        combined = combined.transpose(1, 2)  # [B, T, hidden_dim]
        
        attn_output = self.attention(combined)  # [B, T, hidden_dim]
        
        outputs = [head(attn_output) for head in self.output_heads]  # list of [B, T, 1]
        
        if return_latents:
            # Project per-timestep latent representations.
            proj_latents = self.contrast_head(attn_output)  # [B, T, proj_dim]
            return outputs, proj_latents
        return outputs
