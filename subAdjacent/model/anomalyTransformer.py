import torch
import torch.nn as nn
import torch.nn.functional as F

from subAdjacent.model.attentionsLayer import LinearAnomalyAttention, AttentionLayer
from subAdjacent.model.dataEmbedding import DataEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, d_model, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        print("d_ff = 512, hard coded for debugging, will be changed later!")
        d_ff = 512
        self.attention_layer = attention_layer
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x, queries, keys = self.attention_layer(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), queries, keys


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        # x [B, L, D]
        queries_list = []
        keys_list = []
        for attn_layer in self.attn_layers:
            x, queries, keys = attn_layer(x)
            queries_list.append(queries)
            keys_list.append(keys)

        if self.norm is not None:
            x = self.norm(x)

        return x, queries_list, keys_list


class AnomalyTransformer(nn.Module):
    def __init__(self, enc_in, c_out, d_model=512, n_heads=8, e_layers=3,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        attention_layers = [
            EncoderLayer(
            AttentionLayer(
                LinearAnomalyAttention(
                dropout=dropout, 
                output_attention=output_attention, 
                ),
                d_model, 
                n_heads
            ),
            d_model,
            dropout=dropout,
            activation=activation
            ) 
            for _ in range(e_layers)
        ]
        
        self.encoder = Encoder(
            attn_layers=attention_layers,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, queries_list, keys_list = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, queries_list, keys_list
        else:
            return enc_out  # [B, L, D]
        
    def compute_sub_adj_contrib(queries, keys, span=None, one_side=True):
        """
        Same as your SACon function, returning shape [B, L].
        Minimally renamed here to 'compute_sub_adj_contrib'.
        """
        L = queries.shape[1]
        if span is None:
            span = [20, 30]

        # queries, keys: [B, L, H, D]
        attnMatrix = torch.einsum("b l h e, b s h e -> b h l s", queries, keys)
        attnMatrix = attnMatrix / (attnMatrix.sum(dim=-1, keepdim=True) + 1e-6)

        lossMat = None
        for k in range(-span[1], span[1] + 1):
            if one_side and k < span[0]:
                continue
            diag1 = torch.diagonal(attnMatrix, offset=k, dim1=-2, dim2=-1)
            if k >= 0:
                diag1 = F.pad(diag1, (k, 0))
            else:
                diag1 = F.pad(diag1, (0, abs(k)))
            if lossMat is None:
                lossMat = diag1
            else:
                lossMat += diag1

        # mean over heads => [B, L]
        lossMat = torch.mean(lossMat, dim=1)
        return lossMat  # shape: [B, L]