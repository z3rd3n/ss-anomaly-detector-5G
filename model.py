
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, num_features, hidden_dim, dropout=0.1, num_classes=5):
        """
        A CNN+Self-Attention autoencoder for numerical features that includes:
         - Reconstruction via per-feature output heads.
         - A latent space from the encoder.
         - A classification head that outputs 5 classes:
             4 for rule–based anomalies,
             1 for not anomaly (normal),
        """
        super().__init__()
        self.num_features = num_features
        
        self.conv3 = nn.Conv1d(in_channels=num_features, out_channels=hidden_dim // 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=num_features, out_channels=hidden_dim // 2, kernel_size=5, padding=2)
        self.conv_combine = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
        
        self.attention = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dropout=dropout, batch_first=True)
        
        # Reconstruction heads: one per feature.
        self.output_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_features)])
        
        # Classification head on the latent space.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),  
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, return_latents=False):
        """
        Args:
            x: [B, T, num_features] normalized input.
            return_latents: if True, also return latent representations.
        Returns:
            outputs: list of [B, T, 1] reconstructions (one per feature).
            class_logits: [B, T, num_classes] classification outputs.
            (optionally) attn_output: latent representations [B, T, hidden_dim].
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
        class_logits = self.classifier(attn_output)  # [B, T, num_classes]
        
        if return_latents:
            return outputs, class_logits, attn_output
        else:
            return outputs, class_logits
