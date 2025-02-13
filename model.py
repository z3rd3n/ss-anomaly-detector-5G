import torch
from torch import nn

class NumericalAutoencoder(nn.Module):
    def __init__(self, num_features, hidden_dim, dropout=0.1):
        """
        A CNN + Self-Attention based sequence autoencoder that operates on normalized numerical features.
        
        Args:
            num_features: int, number of input features (e.g. 7).
            hidden_dim: hidden dimension for the convolution and attention layers.
            dropout: dropout probability.
        
        Input:
            x: Tensor of shape [batch, seq_len, num_features] (normalized fp16 values)
        
        Output:
            outputs: List of tensors (one per feature), each of shape [B, T, 1]
            (if return_latents=True, also returns latents: Tensor of shape [B, hidden_dim] from the last timestep)
        
        Tip for improvement: You might add positional encodings or layer normalization before the attention layer
        to capture subtler temporal relationships without increasing the parameter count.
        """
        super().__init__()
        self.num_features = num_features
        
        # Since we no longer use embeddings, the input channel count is just num_features.
        self.conv3 = nn.Conv1d(in_channels=num_features, out_channels=hidden_dim // 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=num_features, out_channels=hidden_dim // 2, kernel_size=5, padding=2)
        self.conv_combine = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
        
        self.attention = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dropout=dropout, batch_first=True)
        
        # One output head per feature; each head predicts a single normalized value.
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_features)
        ])
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape [B, T, num_features] (normalized fp16 values).
            return_latents: If True, also returns the latent representation from the last timestep.
        
        Returns:
            outputs: List of predictions for each feature. Each tensor is of shape [B, T, 1].
            latents (optional): Tensor of shape [B, hidden_dim] from the last timestep.
        """
        batch_size, seq_len, _ = x.size()
        # Transpose to [B, num_features, T] for convolution.
        x_conv = x.transpose(1, 2)
        conv_out_3 = self.conv3(x_conv)
        conv_out_5 = self.conv5(x_conv)
        conv_cat = torch.cat([conv_out_3, conv_out_5], dim=1)
        combined = self.conv_combine(conv_cat)
        combined = combined.transpose(1, 2)  # back to [B, T, hidden_dim]
        
        attn_output = self.attention(combined)
        
        outputs = [head(attn_output) for head in self.output_heads]
        return outputs
