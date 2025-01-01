# subAdjacent/configClass.py
import os

class Config:
    def __init__(self):
        import torch
        import numpy as np
        self.seed=42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.train = True
        self.detect = True
        self.seq_len = 32
        self.stride = None
        self.batch_size = 128
        self.num_epochs = 10
        self.validation_ratio=0.2
        self.learning_rate = 1e-4

        self.optimizer_name = 'Adam'
        self.weight_decay = 6e-5

        self.feature_columns = [
            'SFN', 'Slot', 'CC', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI',
        ]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = 'subAdjacent/results/checkpoint_epoch_5.pt'  # Path to load checkpoint from

        # Directories
        self.parquet_path = "data/scaled_pdsch.parquet"
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        self.output_dir = results_dir

        # Model architecture params
        self.model_dim = 512       # d_model
        self.n_heads = 12          # number of attention heads
        self.e_layers = 4         # number of encoder layers
        self.activation = 'gelu'  # activation function
        self.k_value = 2        # trade-off parameter for loss
        self.dropout = 0.15
        self.span = [4,12]
        self.one_side = False
        self.max_grad_norm = 5.0

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
        from subAdjacent.model.anomalyTransformer import AnomalyTransformer
        model = AnomalyTransformer(
            enc_in=len(self.feature_columns),
            c_out=len(self.feature_columns),
            d_model=self.model_dim,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout,
            activation=self.activation,
            output_attention=self.output_attention,
        ).to(self.device)
        return model