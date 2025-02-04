import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalContrastiveEncoder(nn.Module):
    def __init__(self, params):
        """
        feature_cardinalities: list of int, one per feature (e.g. [1024, 30, 16, 32, 2, 10, 2])
        embed_dim: embedding dimension for each discrete feature
        d_model: the Transformer model dimension
        num_heads: number of attention heads
        num_layers: number of Transformer layers
        """
        super().__init__()
        self.num_features = len(params.feature_cardinalities)
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, params.embed_dim) for card in params.feature_cardinalities
        ])
        # Project concatenated embeddings into d_model:
        self.input_linear = nn.Linear(self.num_features * params.embed_dim, params.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=params.d_model, nhead=params.num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=params.num_layers)
    
    def forward(self, x):
        """
        x: tensor of shape (batch_size, seq_len, num_features) with dtype=torch.long
        returns: latent representations of shape (batch_size, seq_len, d_model)
        """
        # Compute embeddings for each feature (result: list of (batch, seq_len, embed_dim))
        embed_list = [emb(x[:,:,i]) for i, emb in enumerate(self.embeddings)]
        # Concatenate along the feature dimension: (batch, seq_len, num_features*embed_dim)
        x_emb = torch.cat(embed_list, dim=-1)
        x_proj = self.input_linear(x_emb)  # (batch, seq_len, d_model)
        encoded = self.transformer_encoder(x_proj) # (batch, seq_len, d_model)
        return encoded
    
    def encode(self, x):
        return self.forward(x)
