import torch
import torch.nn as nn
import torch.nn.functional as F

from approaches.subAdjacentEmbed.model.attentionsLayer import LinearAnomalyAttention, AttentionLayer
from approaches.subAdjacentEmbed.model.dataEmbedding import EnhancedDataEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, d_model, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = 4 * d_model
        self.attention_layer = attention_layer
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        # x: [B, L, D]
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


class CategoricalDecoder(nn.Module):
    """
    Projects each token embedding (dimension 'd_model') 
    to a distribution over the possible categories for each feature.
    """
    def __init__(self, feature_config, d_model):
        super(CategoricalDecoder, self).__init__()
        self.feature_config = feature_config
        self.feature_heads = nn.ModuleDict()
        for feat_name, config in feature_config.items():
            n_categories = len(config["value_to_index"])
            self.feature_heads[feat_name] = nn.Linear(d_model, n_categories)

    def forward(self, x):
        """
        x: [B, L, D_model]
        Return dict: { feat_name: [B, L, n_categories] }, after softmax
        """
        predictions = {}
        for feat_name, head in self.feature_heads.items():
            logits = head(x)  # shape [B, L, n_categories]
            predictions[feat_name] = logits
        return predictions


class AnomalyTransformer(nn.Module):
    def __init__(self, feature_config, c_out, d_model=512, n_heads=8, e_layers=3,
                 dropout=0.0, activation='gelu', output_attention=True, negative_qk=False):
        """
        c_out is unused here if we're returning a dictionary (unless you need it for something else).
        """
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # 1) Data embedding
        self.embedding = EnhancedDataEmbedding(feature_config, d_model, dropout)

        # 2) Attention layers
        attention_layers = [
            EncoderLayer(
                AttentionLayer(
                    LinearAnomalyAttention(
                        dropout=dropout,
                        output_attention=output_attention,
                        negative_qk=negative_qk
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

        # 3) Categorical Decoder
        self.decoder = CategoricalDecoder(feature_config, d_model)
        
        # If you no longer need a single projection, you can remove it:
        # self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        """
        x is expected to be a dict of LongTensors (per-feature), 
        shape [B, L] each, as handled by EnhancedDataEmbedding.
        
        Returns:
          - predictions: dict of {feat_name: [B, L, n_classes_for_that_feat]}
          - queries_list, keys_list: attention for each layer if output_attention=True
        """
        # 1) Embed
        enc_out = self.embedding(x)  # [B, L, d_model]

        # 2) Encoder
        enc_out, queries_list, keys_list = self.encoder(enc_out)  # [B, L, d_model]

        # 3) Decode to a dictionary of feature distributions
        preds = self.decoder(enc_out)  # { feat_name: [B, L, n_categories] }

        if self.output_attention:
            return preds, queries_list, keys_list
        else:
            return preds

    def compute_sub_adj_contrib(self, q, k, span, one_side):
        """
        Same as your SACon function, returning shape [B, L].
        Minimally renamed here to 'compute_sub_adj_contrib'.
        """
        L = q.shape[1]
        assert L >= span[1] >= span[0] >= 0

        # compute attention matrix: shape [B, n_heads, L, L]
        attnMatrix = torch.einsum("b l h e, b s h e -> b h l s", q, k)
        den = attnMatrix.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        attnMatrix = attnMatrix / den

        lossMat = None
        for offset in range(-span[1], span[1] + 1):
            # handle one-side skip
            if one_side and offset < span[0]:
                continue
            if not one_side and abs(offset) < span[0]:
                continue

            diag1 = torch.diagonal(attnMatrix, offset=offset, dim1=-2, dim2=-1)
            if offset > 0:
                p1d = (offset, 0)
            else:
                p1d = (0, abs(offset))
            diag1 = F.pad(diag1, p1d)

            if lossMat is None:
                lossMat = diag1
            else:
                lossMat += diag1

            # second diagonal pass
            if offset > 0:
                offset_k = -(L - offset)
            else:
                offset_k = L + offset
            diag1 = torch.diagonal(attnMatrix, offset=offset_k, dim1=-2, dim2=-1)
            if offset_k > 0:
                p1d = (offset_k, 0)
            else:
                p1d = (0, abs(offset_k))
            diag1 = F.pad(diag1, p1d)

            lossMat += diag1

        # shape is [B, n_heads, L]; average over n_heads => [B, L]
        lossMat = torch.mean(lossMat, dim=-2)
        return lossMat
