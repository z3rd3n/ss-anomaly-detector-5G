import torch
import torch.nn as nn
import math

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x: [B, S, d_model]
        return x + self.pe[:, :x.size(1)]


class MultiFeatureEmbedding(nn.Module):
    """
    Embeds each of the 7 discrete features and concatenates them.
    """
    def __init__(self, cardinalities, embed_dims):
        super().__init__()
        assert len(cardinalities) == len(embed_dims), "cardinalities and embed_dims must match in length."
        self.embeddings = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cardinalities, embed_dims)])
        self.total_dim = sum(embed_dims)
        self.positional_embedding = SinusoidalPositionalEmbedding(self.total_dim)
    def forward(self, x):
        # x: [B, S, 7]
        embs = []
        for i, emb_layer in enumerate(self.embeddings):
            emb_i = emb_layer(x[..., i])
            embs.append(emb_i)
        concatenated = torch.cat(embs, dim=-1)  # [B, S, total_dim]
        return self.positional_embedding(concatenated)


class MemoryModule(nn.Module):
    """
    A small MLP that maps the key to a value estimate.
    """
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model)
        )
    def forward(self, k):
        return self.net(k)

class Reconstructor(nn.Module):
    """
    Uses a multihead attention block followed by an FFN and projects to produce logits for all features.
    """
    def __init__(self, d_model, cardinalities):
        super().__init__()
        self.d_model = d_model
        self.cardinalities = cardinalities
        self.total_classes = sum(cardinalities)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.proj_out = nn.Linear(d_model, self.total_classes)
    def forward(self, x_up, memory_module):
        # x_up: [B, S, d_model]
        Q = self.W_q(x_up)
        K = self.W_k(x_up)
        # Get V_hat from memory (detached so that reconstructor loss does not update memory_module)
        V_hat = memory_module(K).detach()
        attn_out, _ = self.attn(Q, K, V_hat)
        ffn_out = self.ffn(attn_out)
        logits = self.proj_out(ffn_out)  # [B, S, total_classes]
        return logits
    

class SurpriseGate(nn.Module):
    """
    A learnable gating mechanism that returns 1 if surprise is below a learnable threshold,
    and 0 otherwise. (In other words, if the surprise is too large the instance is ignored.)
    """
    def __init__(self, init_threshold=0.1):
        super().__init__()
        # Learnable threshold (if surprise is greater than this, gate is 0)
        self.threshold = nn.Parameter(torch.tensor(init_threshold, dtype=torch.float32))
    
    def forward(self, surprise):
        # Binary gate: if surprise is less than or equal to threshold -> 1, else 0.
        gate = (surprise <= self.threshold).float()
        return gate
    

class SurpriseTransformer(nn.Module):
    """
    Combined model that embeds the multi-feature input, applies a memory module,
    and reconstructs the input via attention.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_multi = MultiFeatureEmbedding(config.cardinalities, config.embed_dims)
        d_model = self.embed_multi.total_dim
        self.pos_embedding = SinusoidalPositionalEmbedding(d_model)
        self.memory_module = MemoryModule(d_model)
        self.reconstructor = Reconstructor(d_model, config.cardinalities)
        self.surprise_gate = SurpriseGate(init_threshold=config.surprise_threshold)
    def forward(self, x):
        # x: [B, S, 7]
        x_up = self.embed_multi(x)
        x_up = self.pos_embedding(x_up)
        v_hat = self.memory_module(x_up)
        logits = self.reconstructor(x_up, self.memory_module)
        return x_up, v_hat, logits
