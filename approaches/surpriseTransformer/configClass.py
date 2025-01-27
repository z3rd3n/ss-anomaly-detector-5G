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
        self.parquet_path = "data/scaled_pdsch.parquet"
        self.output_dir = "approaches/surpriseTransformer/output"


        self.feature_columns = ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"]
        self.seq_len = 32
        self.validation_ratio = 0.2
        self.seed = 42
        self.batch_size = 64
        self.num_workers = 0
        self.pin_memory = False

        # Model / training hyperparams
        self.num_epochs = 30
        self.learning_rate = 1e-3
        self.lr_memory = 1e-4
        self.weight_decay = 1e-5
        self.optimizer_name = 'AdamW'

        self.early_stopping_patience = 5
        self.early_stopping_min_delta = 1e-4

        self.lambda_surprise = 100.0
        self.surprise_threshold = 0.05
        self.gate_alpha = 10.0  # scale the grads
        self.gate_beta = -1.0 # shift the sigmoid
        self.d_model = 128

        self.feature_columns = ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"]
        self.cardinalities = [1024, 30, 16, 32, 2, 10, 2]
        self.embed_dims = [12, 4, 4, 4, 2, 4, 2]  # sums to 32
        self.detect_feature_weights = [0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0]
        self.alpha_surprise_detect = 100.0  # weighting surprise in anomaly score

        self.train = True
        self.detect = True

    def build_model(self):
            return SurpriseTransformer(d_in=len(self.feature_columns), d_model=self.d_model)


