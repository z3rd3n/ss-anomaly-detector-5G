import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from dataset import ParquetSequenceDataset, custom_collate_fn
from torch.utils.data import DataLoader

import logging
from utils import save_checkpoint, start_logging, validate_csv, compute_features_statistics, log_model_size
from model import NumericalAutoencoder

def train_anomaly_detector(model, train_loader, optimizer, device, params, means, variances):
    num_epochs = params.get('num_epochs', 10)
    best_f1 = 0.0
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs}", unit="batch") as pbar:
            for batch in train_loader:
                optimizer.zero_grad()
                features = batch['features'].to(device)  # [B, T, num_features]
                outputs = model(features)
                batch_loss = 0.0
                # Compute MSE loss for each feature head.
                for i in range(model.num_features):
                    pred = outputs[i]  # [B, T, 1]
                    target = features[:,:,i:i+1]  # [B, T, 1]
                    loss_i = F.mse_loss(pred, target, reduction='mean')
                    batch_loss += loss_i
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
                pbar.set_postfix(loss=f"{batch_loss.item():.4f}")
                pbar.update(1)
        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch}/{num_epochs}: Average Loss = {avg_loss:.4f}")
        
        # Perform CSV validation at the end of each epoch.
        precision, recall, f1 = validate_csv(model, params, means, variances)
        logging.info(f"Epoch {epoch}: Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'params': params,
            }
            filename = os.path.join(params['output_dir'], 'best_model.pt')
            save_checkpoint(checkpoint_state, filename)
    return model, best_f1

def main_training_pipeline(params, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create the training dataset. Note that the dataset now will skip anomalous sequences
    # and perform normalization (using the provided statistics).
    train_dataset = ParquetSequenceDataset(
        parquet_path=params['parquet_path'],
        feature_columns=params['feature_columns'],
        seq_len=params['seq_len'],
        stride=params['stride'],
        split='train',
        validation_ratio=params['validation_ratio'],
        seed=params['seed'],
        skip_anomalies=params['skip_anomalies'],
        normalization_stats=None  # Initially, we load raw data for computing stats.
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    # Compute features statistics from the raw training data.
    num_features = len(params['feature_columns'])
    means, variances = compute_features_statistics(train_loader, params, num_features)
    
    # Now, reinitialize the dataset with normalization enabled.
    train_dataset.normalization_stats = {'means': means, 'variances': variances}
    
    log_model_size(model, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 1e-3))
    model, best_f1 = train_anomaly_detector(model, train_loader, optimizer, device, params, means, variances)
    logging.info(f"Training complete. Best validation F1: {best_f1:.4f}")
    return model, params

if __name__ == '__main__':
    # Hyperparameters and settings.
    params = {
        'parquet_path': 'unscaled_pdsch.parquet',
        'validation_parquet_path': 'unscaled_pdsch_val.parquet', 
        'output_dir': './output',
        'feature_columns': ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"],
        'seq_len': 20,
        'stride': 10,
        'validation_ratio': 0.0,
        'seed': 42,
        'batch_size': 64,
        'num_epochs': 3,
        'lr': 5e-4,
        'hidden_dim': 128,
        'percentile': 95,
        'dropout': 0.5,
        'pca_n_components': 2,
        'features_stats_json': 'features_stats.json',
        'skip_anomalies': True
    }

    start_logging(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = len(params['feature_columns'])
    model = NumericalAutoencoder(num_features=num_features, hidden_dim=params['hidden_dim'], dropout=params['dropout'])
    model.to(device)
    model, params = main_training_pipeline(params, model)
