
# subAdjacentLSTM/configClass.py
import os

class Config:
    def __init__(self):
        import torch
        import numpy as np
        self.seed=42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.train = True
        self.detect = False
        self.seq_len = 16
        self.stride = None
        self.batch_size = 64
        self.num_epochs = 15
        self.validation_ratio=0.2
        self.learning_rate = 9.2e-3

        self.optimizer_name = 'AdamW'
        self.weight_decay = 9e-3

        self.feature_columns = [
            'SFN', 'Slot', 'CC', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI',
        ]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = None  # Path to load checkpoint from

        # Directories
        self.parquet_path = "data/scaled_pdsch.parquet"
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        self.output_dir = results_dir

        # Model architecture params
        self.model_dim = 512       # d_model
        self.latent_dim = 128       # d_latent
        self.n_heads = 4          # number of attention heads
        self.e_layers = 2         # number of encoder layers
        self.lamda_sacon = 10        # trade-off parameter for loss
        self.lamda_rec = 2
        self.dropout = 0.15
        self.k1 = 4
        self.k2 = 8
        self.alpha = 0.1

        # Training specific params
        self.shuffle_files = True
        self.output_attention = True

        # EVT params
        self.p = 95 # percentile
        self.q = 0.99 # quantile

        # System params
        self.num_workers = 0
        self.pin_memory = True

    def build_model(self):
        from subAdjacentLSTM.model.subAdjacent import ForcedSubAdjacentLSTM
        model = ForcedSubAdjacentLSTM(self).to(self.device)
        return model


