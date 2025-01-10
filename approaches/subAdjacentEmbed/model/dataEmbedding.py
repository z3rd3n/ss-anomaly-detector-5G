# # subAdjacentEmbed/model/dataEmbedding.py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


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
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return x
    

class CategoricalFeatureEmbedding(nn.Module):
    def __init__(self, feature_config, d_model):
        super(CategoricalFeatureEmbedding, self).__init__()
        
        self.feature_embeddings = nn.ModuleDict()
        self.feature_norms = nn.ModuleDict()
        self.feature_config = feature_config
        
        for feat_name, config in feature_config.items():
            n_categories = len(config["value_to_index"])
            # Calculate embedding size based on cardinality
            embed_dim = min(4, (n_categories + 1) // 2)
            
            self.feature_embeddings[feat_name] = nn.Embedding(
                num_embeddings=n_categories,
                embedding_dim=embed_dim
            )
            
            # Layer norm per feature
            self.feature_norms[feat_name] = nn.LayerNorm(embed_dim)
            
        # Project all embeddings to d_model dimension
        total_embed_dim = sum(embed.embedding_dim for embed in self.feature_embeddings.values())
        self.projection = nn.Sequential(
            nn.Linear(total_embed_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, x_dict):
        embedded_features = []
        
        for feat_name, embedding in self.feature_embeddings.items():
            # Get feature input
            feat_input = x_dict[feat_name]
            
            # Embed the feature
            feat_embed = embedding(feat_input)
            
            # Apply layer norm
            feat_embed = self.feature_norms[feat_name](feat_embed)
            
            embedded_features.append(feat_embed)
        
        # Concatenate all embeddings
        x = torch.cat(embedded_features, dim=-1)
        
        # Project to d_model dimension
        return self.projection(x)

class EnhancedDataEmbedding(nn.Module):
    def __init__(self, feature_config, d_model, dropout=0.1):
        super(EnhancedDataEmbedding, self).__init__()
        
        self.categorical_embedding = CategoricalFeatureEmbedding(feature_config, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Get categorical embeddings
        cat_embed = self.categorical_embedding(x)
        
        # Add positional encoding
        x = cat_embed + self.position_embedding(cat_embed)
        
        return self.dropout(x)

def compute_categorical_reconstruction_loss(predictions, targets, feature_config):
    """
    Compute reconstruction loss considering categorical nature of features
    while preserving the structure needed for sub-adjacent attention
    """
    total_loss = 0
    for feat_name in feature_config:
        pred = predictions[..., feature_config[feat_name]["feature_idx"]]
        true = targets[..., feature_config[feat_name]["feature_idx"]]
        
        # Get class weights for this feature
        weights = torch.tensor([
            feature_config[feat_name]["class_weights"].get(str(i), 1.0)
            for i in range(len(feature_config[feat_name]["value_to_index"]))
        ]).to(predictions.device)
        
        # Compute weighted cross entropy for this feature
        loss = F.cross_entropy(
            pred.reshape(-1, len(feature_config[feat_name]["value_to_index"])),
            true.reshape(-1).long(),
            weight=weights,
            reduction='none'
        )
        
        total_loss += loss.reshape(predictions.shape[0], -1)
    
    return total_loss