import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        layers = []
        in_channels = c_in
        while in_channels * 4 < d_model:
            out_channels = in_channels * 4
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=padding,
                    padding_mode='circular',
                    bias=False
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        # Final layer to reach d_model
        layers.append(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=d_model,
                kernel_size=3,
                padding=padding,
                padding_mode='circular',
                bias=False
            )
        )
        layers.append(nn.BatchNorm1d(d_model))
        layers.append(nn.ReLU())
        self.tokenConv = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return x

class MemoryModule(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # For example, a small MLP to map k -> v_hat
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model//4),
            nn.ReLU(),
            nn.Linear(d_model//4, d_model),
        )

    def forward(self, k):
        # Returns predicted value for the given key
        v_hat = self.net(k)
        return v_hat

class Reconstructor(nn.Module):
    def __init__(self, d_model, d_input):
        super().__init__()
        # e.g. some multi-head attention block
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        # ...
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
        )
        self.proj_out = nn.Linear(d_model, d_input)  # final down-project

    def forward(self, x_up, memory_module):
        # x_up shape: [B, S, d_model]
        # We'll treat them as queries & keys (toy example)

        Q = self.W_q(x_up)  # [B, S, d_model]
        K = self.W_k(x_up)  # [B, S, d_model]
        # For each position, get V from memory
        V_hat = memory_module(K).detach()  # shape [B, S, d_model]

        attn_out, _ = self.attn(Q, K, V_hat)

        # pass through feed-forward
        ffn_out = self.ffn(attn_out)       # [B, S, d_model]
        x_recon_up = ffn_out        # a typical residual connection, # I want it to be reconstructed, but surprise shouldn't be affected
        x_recon = self.proj_out(x_recon_up)  # [B, S, d_input]
        return x_recon
    

# Gating function example:
class SurpriseGate(nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(10.0))  # Steeper sigmoid
        self.beta = nn.Parameter(torch.tensor(-1.0))

    def forward(self, surprise):
        # More aggressive gating based on a fixed threshold
        gate = torch.sigmoid(-self.alpha * surprise + self.beta)
        return gate
    

class SurpriseTransformer(nn.Module):
    """
    Single combined model that wraps:
      - DataEmbedding
      - MemoryModule
      - Reconstructor
      - SurpriseGate

    This allows saving/loading *one* model checkpoint easily.
    """
    def __init__(self, d_in, d_model):
        """
        Args:
            d_in (int): Input feature dimension (e.g. len(feature_columns))
            d_model (int): Dimension for internal embedding/transformer feed-forward
        """
        super().__init__()
        self.embed = DataEmbedding(d_in, d_model)
        self.memory_module = MemoryModule(d_model)
        self.reconstructor = Reconstructor(d_model, d_in)
        self.surprise_gate = SurpriseGate()

    def forward(self, x):
        """
        A simple forward pass example that does:
          1) embed -> x_up
          2) memory -> v_hat
          3) reconstruct -> x_recon
        and returns (x_up, v_hat, x_recon)

        In practice, your training code might do extra steps
        (like gating, computing gradient-based surprise, etc.).
        """
        x_up = self.embed(x)                     # [B, S, d_model]
        v_hat = self.memory_module(x_up)         # [B, S, d_model]
        x_recon = self.reconstructor(x_up, self.memory_module)  # [B, S, d_in]
        return x_up, v_hat, x_recon
