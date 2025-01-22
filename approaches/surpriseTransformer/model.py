import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveSurpriseThreshold(nn.Module):
    def __init__(self, input_dim, initial_threshold=0.7):
        super().__init__()
        # Initialize with a relatively high threshold (0.7) to focus on normal patterns
        self.base_threshold = nn.Parameter(torch.tensor(initial_threshold))
        self.adaptive_component = nn.Linear(input_dim, 1)
        
    def forward(self, x_t):
        # x_t is shape [B, Seq, d_in] or [B, d_in]
        # Combine base threshold with input-dependent adjustment
        # We apply tanh(...) * 0.3 for ±0.3 range
        adjustment = torch.tanh(self.adaptive_component(x_t)) * 0.3
        # base_threshold is shape [], so broadcasting works
        return torch.sigmoid(self.base_threshold + adjustment)

class DimensionProjection(nn.Module):
    """Projects input features to higher dimension and back."""
    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        self.project_up = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.ReLU()
        )
        self.project_down = nn.Sequential(
            nn.Linear(d_model, d_in)
        )
    
    def forward_up(self, x: torch.Tensor) -> torch.Tensor:
        return self.project_up(x)
    
    def forward_down(self, x: torch.Tensor) -> torch.Tensor:
        return self.project_down(x)

class SurpriseMemory(nn.Module):
    def __init__(self, d_in, d_model=128, d_ff=512, n_heads=4,
                 mem_layers=2, learnable_surprise=True):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model

        # Dimension projection
        self.dim_proj = DimensionProjection(d_in, d_model)

        # Key and Value projections (in higher dimensions)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Memory module (MLP)
        memory_layers = []
        memory_layers.append(nn.Linear(d_model, d_model))
        memory_layers.append(nn.LayerNorm(d_model))
        memory_layers.append(nn.ReLU())
        for _ in range(mem_layers - 1):
            memory_layers.append(nn.Linear(d_model, d_model))
            memory_layers.append(nn.LayerNorm(d_model))
            memory_layers.append(nn.ReLU())
        self.memory_network = nn.Sequential(*memory_layers)

        # Parameters for surprise mechanism
        self.eta   = nn.Linear(d_in, 1)  # surprise decay
        self.theta = nn.Linear(d_in, 1)  # learning rate
        self.alpha = nn.Linear(d_in, 1)  # forgetting rate

        self.learnable_surprise = learnable_surprise
        if learnable_surprise:
            self.surprise_threshold = AdaptiveSurpriseThreshold(d_in)
        else:
            self.surprise_threshold = 0.7  # Fixed threshold

        # Memory update buffer
        #  self.S shape: [d_model, d_model]
        self.register_buffer('S', torch.zeros(d_model, d_model))

        # Surprise history for normalization
        self.register_buffer('surprise_history', torch.zeros(10000))
        self.surprise_idx = 0

        # Attention mechanism
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        # Simple reconstructor
        self.reconstructor = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.ReLU(),
        )

    def compute_reconstruction_loss(self, x_input, x_recon):
        """Compute reconstruction loss"""
        return F.mse_loss(x_recon, x_input)
    
    def compute_surprise_score(self, grads):
        """Compute normalized surprise score from the network gradients."""
        # 'grads' is a tuple of Tensors for each param
        all_grads = []
        for g in grads:
            if g is not None:
                all_grads.append(g.reshape(-1))
        if len(all_grads) == 0:
            # If for some reason we have no grads
            return torch.tensor(0.0, device=grads[0].device if grads[0] is not None else 'cpu')
        all_grads = torch.cat(all_grads, dim=0)

        # L2 norm
        surprise_magnitude = torch.norm(all_grads, p=2)

        # Normalize with running stats
        if self.surprise_idx > 0:
            mean = self.surprise_history[:self.surprise_idx].mean()
            std = self.surprise_history[:self.surprise_idx].std() + 1e-6
            normalized_surprise = (surprise_magnitude - mean) / std
        else:
            normalized_surprise = surprise_magnitude

        # Update buffer
        self.surprise_history[self.surprise_idx] = surprise_magnitude.detach()
        self.surprise_idx = (self.surprise_idx + 1) % len(self.surprise_history)

        return torch.sigmoid(normalized_surprise)  # [0,1] range
    
    def forward(self, x_t):
        """
        Forward pass: encode + reconstruct
        x_t shape expected: [B, Seq, d_model]
        Returns: x_recon shape [B, Seq, d_in]
        """
        q_t = self.W_Q(x_t)
        k_t = self.W_K(x_t)
        v_t = self.W_V(x_t)

        # Pass Q through memory network to get retrieval
        retrieval = self.memory_network(q_t)

        # Multi-head attention
        attn_outs, _ = self.cross_attn(
            query=retrieval,  # [B, Seq, d_model]
            key=k_t,
            value=v_t
        )
        # Reconstruct in d_model space
        x_recon = self.reconstructor(attn_outs)
        # Project back down to d_in
        x_recon = self.dim_proj.forward_down(x_recon)
        return x_recon

    def update_memory(self, x_t, is_training=True):
        """
        x_t shape: [B, Seq, d_in]
        Returns:
            x_recon, recon_loss, assoc_loss, surprise_score
        """
        # 1) Project x_t up to d_model
        x_up = self.dim_proj.forward_up(x_t)  # [B, Seq, d_model]

        # 2) Create k_t, v_t
        k_t = self.W_K(x_up)  # [B, Seq, d_model]
        v_t = self.W_V(x_up)

        # 3) Compute per-sample scalars (currently shape [B, Seq, 1])
        eta_t   = torch.sigmoid(self.eta(x_t))    # shape [B, Seq, 1]
        theta_t = F.softplus(self.theta(x_t))     # shape [B, Seq, 1]
        alpha_t = torch.sigmoid(self.alpha(x_t))  # shape [B, Seq, 1]

        # 4) If learnable threshold, we get shape [B, Seq, 1] => reduce to scalar
        if self.learnable_surprise:
            threshold_tensor = self.surprise_threshold(x_t)  # [B, Seq, 1]
            threshold_val = threshold_tensor.mean().item()
        else:
            threshold_val = float(self.surprise_threshold)  # e.g. 0.7

        # 5) Forward pass through memory to get v_t_hat
        v_t_hat = self.memory_network(k_t)  # [B, Seq, d_model]

        # 6) Reconstruct the original x
        x_recon = self.forward(x_up)  # [B, Seq, d_in]

        # 7) Compute losses
        assoc_loss = self.compute_reconstruction_loss(v_t_hat, v_t)
        recon_loss = self.compute_reconstruction_loss(x_t, x_recon)

        # 8) Compute gradient w.r.t. memory network
        gradients = torch.autograd.grad(
            assoc_loss,
            self.memory_network.parameters(),
            create_graph=True
        )

        # 9) Surprise score (0-dim tensor)
        surprise_score = self.compute_surprise_score(gradients)

        # 10) Memory update (only if training and surprise is below threshold)
        if is_training:
            if surprise_score.item() < threshold_val:
                # (a) reduce the 3D scalars to single scalars for the entire batch
                eta_val   = eta_t.mean()   # shape [] in PyTorch
                theta_val = theta_t.mean()
                alpha_val = alpha_t.mean()

                # (b) S_new = eta_val * S
                #     Then add -theta_val * gradient for each param with dim=2
                S_new = eta_val * self.S

                for param, grad in zip(self.memory_network.parameters(), gradients):
                    if grad is None:
                        continue
                    # We only update memory-related parameters that are [d_model, d_model]
                    if param.dim() == 2 and param.shape == (self.d_model, self.d_model):
                        S_new = S_new + (-theta_val * grad)

                self.S = S_new

                # (c) Apply forgetting to each param in memory_network
                with torch.no_grad():
                    for param in self.memory_network.parameters():
                        if param.dim() == 2 and param.shape == (self.d_model, self.d_model):
                            param.data = (1.0 - alpha_val) * param.data + self.S

        return x_recon, recon_loss, assoc_loss, surprise_score
