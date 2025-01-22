# surpriseTransformer/configClass.py
import os
import torch
import numpy as np
from approaches.surpriseTransformer.model import SurpriseMemory

class Config:
    def __init__(self):

        self.seed=42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Fill these with your desired hyperparams or read them from args
        self.parquet_path = "data/scaled_pdsch.parquet"
        self.feature_columns = ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"]
        self.seq_len = 32
        self.validation_ratio = 0.2
        self.seed = 42
        self.batch_size = 64
        self.num_workers = 0
        self.pin_memory = False

        self.num_epochs = 10
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.optimizer_name = 'AdamW'

        self.early_stopping_patience = 5
        self.early_stopping_min_delta = 1e-4

        self.d_hidden_mem = 64
        self.momentum_beta = 0.9
        self.lr_memory = 1e-3
        self.gamma = 2
        self.n_heads = 2
        self.d_model = 128
        self.d_ff = 512
        self.num_layers = 2

        self.lambda_assoc = 1.0
        self.lambda_mse = 1.0
        self.lambda_surprise = 1

        self.train = True
        self.detect = True

    def build_model(self):
            d_in = len(self.feature_columns)
            model = SurpriseMemory(
                d_in=d_in,
                d_model=self.d_model,
                d_ff=self.d_ff,
                n_heads=self.n_heads,
                mem_layers=self.num_layers,
                learnable_surprise=True
            )
            return model


