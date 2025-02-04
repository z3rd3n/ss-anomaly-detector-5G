import torch
import numpy as np
from approaches.sslgad.model import TemporalContrastiveEncoder

class Config:
    def __init__(self):
        # Model parameters

        self.seed=42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Fill these with your desired hyperparams or read them from args
        self.parquet_path = "data/pdsch_data_romes_clean/processed/unscaled_pdsch.parquet"
        self.output_dir = "/workspaces/thesis/approaches/sslgad/output"

        self.validation_csv_path = "/workspaces/thesis/data/pdsch_data_romes_clean/test/test.csv"
        self.ground_truth_csv_path = "/workspaces/thesis/data/pdsch_data_romes_clean/test/test_gt.csv"

        self.feature_columns = ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"]
        self.validation_ratio = 0.0
        self.seed = 42
        self.num_workers = 0
        self.pin_memory = False

        # Model / training hyperparams
        self.num_epochs = 1
        self.seq_len = 100
        self.window_size = 50
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.lr_memory = 1e-4
        self.weight_decay = 1e-5
        self.optimizer_name = 'AdamW'

        self.early_stopping_patience = 5
        self.early_stopping_min_delta = 1e-4

        self.train = True
        self.detect = False

        self.feature_cardinalities = [1024, 30, 16, 32, 2, 10, 2]
        self.embed_dim = 8
        self.d_model = 32
        self.num_heads = 4
        self.num_layers = 2
        
        # Training parameters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.temperature = 0.1
        self.gmm_components = 5
        self.beta = 0.5
        self.anomaly_threshold_std = 3.0

        

    def build_model(self):
            return TemporalContrastiveEncoder(self)
