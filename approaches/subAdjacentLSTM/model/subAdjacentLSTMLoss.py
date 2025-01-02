
# subAdjacentLSTM/model/subAdjacentLSTMLoss.py
import torch
import torch.nn as nn


class SubAdjacentLSTMLoss(nn.Module):
    def __init__(self, k1, k2, alpha=0.1, lamda_rec = 2, lamda_sacon = 10):
        super().__init__()
        self.k1 = k1  # Minimum distance for sub-adjacent region
        self.k2 = k2  # Maximum distance for sub-adjacent region
        self.alpha = alpha  # Weight for non-sub-adjacent regions
        self.lamda_rec = lamda_rec
        self.lamda_sacon = lamda_sacon
        self.mse = nn.MSELoss(reduction='none')
    
    def create_temporal_weight_mask(self, seq_len, device):
        mask = torch.ones(seq_len, seq_len, device=device) * self.alpha
        for i in range(seq_len):
            for j in range(seq_len):
                dist = abs(i - j)
                if self.k1 <= dist <= self.k2:
                    mask[i, j] = 1.0
                else:
                    mask[i, j] = 0
        return mask
    
    def forward(self, x_input, x_recon, hidden_state):
        _, seq_len, _ = x_input.size()
        
        # Basic reconstruction loss
        rec_loss = self.mse(x_input, x_recon).mean(dim=-1)  # [batch_size, seq_len]
        
        # Create temporal weight mask
        temp_mask = self.create_temporal_weight_mask(seq_len, x_input.device)
        
        temporal_sim = torch.bmm(hidden_state, hidden_state.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        
        # Apply temporal mask to emphasize sub-adjacent dependencies
        temporal_sim = temporal_sim * temp_mask

        #row_sums = temporal_sim.sum(dim=2, keepdim=True) + 1e-8
        #temporal_sim = temporal_sim / row_sums
        
        # Calculate temporal dependency loss
        temp_loss = -torch.mean(temporal_sim, dim=(1, 2))  # [batch_size]
        
        # Combine losses
        total_loss = self.lamda_rec * rec_loss.mean(dim=1) + self.lamda_sacon * temp_loss
        
        return total_loss.mean(), rec_loss.mean(), temp_loss.mean()