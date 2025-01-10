# subAdjacent/configClass.py
import os
import json

class Config:
    def __init__(self):
        import torch
        import numpy as np
        self.seed=42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.train = True
        self.detect = True
        self.seq_len = 25
        self.stride = None
        self.batch_size = 32
        self.num_epochs = 15
        self.validation_ratio=0.2
        self.learning_rate = 1e-4

        self.optimizer_name = 'AdamW'
        self.weight_decay = 1e-4

        self.feature_columns = [
            'SFN', 'Slot', 'CC', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI',
        ]
        with open('/workspaces/thesis/approaches/subAdjacentEmbed/model/feature_mappings.json', 'r') as file:
            self.feature_config = json.load(file)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = None  # Path to load checkpoint from

        # Directories
        self.parquet_path = "/workspaces/thesis/approaches/subAdjacentEmbed/data/unscaled_pdsch.parquet"
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        self.output_dir = results_dir

        # Model architecture params
        self.model_dim = 512       # d_model
        self.n_heads = 8          # number of attention heads
        self.e_layers = 2         # number of encoder layers
        self.activation = 'gelu'  # activation function
        self.k_value = 10        # trade-off parameter for loss
        self.lamda_rec = 2
        self.negative_qk = False
        self.dropout = 0.4
        self.span = [12, 18]
        self.one_side = False
        self.max_grad_norm = 5.0

        # Training specific params
        self.shuffle_files = True
        self.output_attention = True

        # EVT params
        self.p = 95 # percentile
        self.q = 0.95 # quantile

        # System params
        self.num_workers = 0
        self.pin_memory = True

    def build_model(self):
        from approaches.subAdjacentEmbed.model.anomalyTransformer import AnomalyTransformer
        model = AnomalyTransformer(
            feature_config=self.feature_config,
            c_out=len(self.feature_columns),
            d_model=self.model_dim,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout,
            activation=self.activation,
            output_attention=self.output_attention,
            negative_qk=self.negative_qk
        ).to(self.device)
        return model


