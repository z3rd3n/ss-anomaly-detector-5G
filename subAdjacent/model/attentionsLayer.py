from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAnomalyAttention(nn.Module):    
    def __init__(
        self,
        dropout: float = 0.0,
        output_attention: bool = False,
    ):
        super().__init__()
        self.output_attention = output_attention
        self.dropout = nn.Dropout(dropout)
        
        self.softmax = nn.Softmax(dim=-1)
        self.delta1 = nn.Parameter(torch.tensor(1.0))

    def _apply_mapping(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        queries[queries < 0] = -100
        keys[keys < 0] = -100
        delta = nn.Softplus()(self.delta1)
        queries = self.softmax(queries / delta)
        keys = self.softmax(keys / delta)

        return queries, keys

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of the linear attention mechanism."""
        
        queries, keys = self._apply_mapping(queries, keys)
        
        kv = torch.einsum("b e h l, b l h f -> b h e f", keys.transpose(1, 3), values)

        z = 1 / (torch.einsum("b l h e, b h e -> b l h", queries, keys.sum(dim=1)) + 1e-6)
        output = torch.einsum("b l h e, b h e e, b l h -> b l h e", queries, kv, z)
        
        if self.output_attention:
            return output.contiguous(), (queries, keys)
        return output.contiguous(), None


class AttentionLayer(nn.Module): 
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None
    ):
        super().__init__()
        
        self.d_keys = d_keys or (d_model // n_heads)
        self.d_values = d_values or (d_model // n_heads)
        self.n_heads = n_heads
        
        # Layer components
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        
        # Projection layers
        self.query_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, self.d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(self.d_values * n_heads, d_model)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size, seq_len, _ = queries.shape
        _, source_len, _ = keys.shape
        
        # Project inputs to multi-head representations
        queries = self.query_projection(queries).view(batch_size, seq_len, self.n_heads, -1)
        keys = self.key_projection(keys).view(batch_size, source_len, self.n_heads, -1)
        values = self.value_projection(values).view(batch_size, source_len, self.n_heads, -1)
        
        # Apply attention mechanism
        output, (query_weights, key_weights) = self.inner_attention(
            queries,
            keys,
            values
        )
        
        # Reshape and project output
        output = output.view(batch_size, seq_len, -1)
        output = self.out_projection(output)
        
        return output, query_weights, key_weights