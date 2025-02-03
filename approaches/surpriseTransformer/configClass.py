# surpriseTransformer/configClass.py
import os
import torch
import numpy as np
from approaches.surpriseTransformer.model import SurpriseTransformer

class Config:
    def __init__(self):

        self.seed=42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Fill these with your desired hyperparams or read them from args
        self.parquet_path = "data/pdsch_data_romes_clean/processed/unscaled_pdsch.parquet"
        self.output_dir = "/workspaces/thesis/approaches/surpriseTransformer/output"

        self.validation_csv_path = "/workspaces/thesis/data/pdsch_data_romes_clean/test/test.csv"
        self.ground_truth_csv_path = "/workspaces/thesis/data/pdsch_data_romes_clean/test/test_gt.csv"


        self.feature_columns = ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"]
        self.validation_ratio = 0.0
        self.seed = 42
        self.num_workers = 0
        self.pin_memory = False

        # Model / training hyperparams
        self.num_epochs = 1
        self.seq_len = 50
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.lr_memory = 1e-4
        self.weight_decay = 1e-5
        self.optimizer_name = 'AdamW'

        self.early_stopping_patience = 5
        self.early_stopping_min_delta = 1e-4

        self.lambda_surprise = 10.0
        self.lambda_rules = 100.0
        self.assoc_loss_weight = 0.1
        self.surprise_threshold = 0.05
        self.gate_alpha = 10.0  # scale the grads
        self.gate_beta = -1.0 # shift the sigmoid

        self.feature_columns = ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"]
        self.cardinalities = [1024, 30, 16, 32, 2, 10, 2]
        self.embed_dims = [12, 4, 4, 4, 2, 4, 2]  # sums to 32
        self.feature_weights = {
            "SFN": 0.1,
            "Slot": 0.1,
            "HARQ": 0.1,
            "MCS": 0.2,
            "CRC": 0.2,
            "ReTx": 0.2,
            "NDI": 0.1
        }
        self.alpha_surprise_detect = 100.0  # weighting surprise in anomaly score
        self.max_retx_normal = 4

        self.train = True
        self.detect = False

    def build_model(self):
            return SurpriseTransformer(self)


