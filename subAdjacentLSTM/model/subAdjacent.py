
# subAdjacentLSTM/model/subAdjacentLSTM.py
import torch
import torch.nn as nn
from subAdjacentLSTM.model.dataEmbedding import DataEmbedding

class ForcedSubAdjacentLSTM(nn.Module):
    def __init__(self, params):
        super(ForcedSubAdjacentLSTM, self).__init__()
        
        self.input_size = len(params.feature_columns)
        self.model_dim = params.model_dim
        self.seq_len = params.seq_len

        # Data embedding layer
        self.embedding = DataEmbedding(c_in=self.input_size, d_model=self.model_dim)
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=self.model_dim,
            hidden_size=params.latent_dim,
            num_layers=params.e_layers,
            batch_first=True,
            dropout=params.dropout
        )
        

        # Decoder LSTM (only uses attended features)
        self.decoder = nn.LSTM(
            input_size=params.latent_dim,  # Only takes attended features
            hidden_size=params.model_dim,
            num_layers=params.e_layers,
            batch_first=True,
            dropout=params.dropout
        )
        

        self.output_proj  =  nn.Linear(params.model_dim, self.input_size)      
        self.stored_states = None
        
    def forward(self, x):
        x = self.embedding(x)

        # Encoder forward pass
        encoder_out, (h_n, c_n) = self.encoder(x)
        
        # Store states for loss computation
        self.hidden_states = encoder_out
        
        # Decoder forward pass
        decoder_out, _ = self.decoder(encoder_out)
        
        # Project to original dimension
        output = self.output_proj(decoder_out)
        
        return output
    
    def compute_anomaly_scores(self, x, recon, params):
        """
        Compute anomaly scores based on reconstruction error and temporal dependencies
        """
        batch_size, seq_len, _ = x.size()
        
        # Reconstruction error component
        rec_error = torch.mean((x - recon) ** 2, dim=-1)  # [batch_size, seq_len]
        
        # Temporal dependency component using stored states
        h_states = self.hidden_states  # [batch_size, seq_len, hidden_size]
        
        # Calculate temporal similarities
        temp_sim = torch.bmm(h_states, h_states.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        
        row_sums = temp_sim.sum(dim=2, keepdim=True) + 1e-8
        temporal_sim = temp_sim / row_sums

        mask = torch.zeros(seq_len, seq_len, device=x.device)
        for i in range(seq_len):
            for j in range(seq_len):
                dist = abs(i - j)
                if params.k1 <= dist <= params.k2:
                    mask[i, j] = 1.0
                else:
                    mask[i, j] = params.alpha
        
        # Calculate temporal anomaly score
        temp_sim = temp_sim * mask
        temp_score = 1.0 - torch.mean(temp_sim, dim=-1)  # [batch_size, seq_len]
        
        # Combine scores (higher score = more anomalous)
        anomaly_scores = rec_error * (1 + temp_score)
        
        return anomaly_scores