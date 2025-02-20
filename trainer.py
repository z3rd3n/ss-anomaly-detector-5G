import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from dataset import ParquetSequenceDataset, custom_collate_fn
from torch.utils.data import DataLoader

import logging
from utils import (save_checkpoint, start_logging, validate_csv, 
                   compute_features_statistics, log_model_size, EPS,
                   center_margin_loss)
from model import NumericalAutoencoder

def train_anomaly_detector(model, train_loader, optimizer, device, params, means, variances):
    num_epochs = params.get('num_epochs', 10)
    best_f1 = 0.0
    contrast_lambda = params.get('contrast_lambda', 1.0)  # weight for contrastive loss

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs}", unit="batch") as pbar:
            for batch in train_loader:
                optimizer.zero_grad()
                # features shape: [B, T, num_features]
                features = batch['features'].to(device)
                # rule_flags: [B, T] (bool tensor indicating rule-based anomaly at each timestamp)
                rule_flags = batch['rule_flags'].to(device)
                
                # Forward pass with latent extraction.
                outputs, proj_latents = model(features, return_latents=True)
                # Concatenate outputs along feature dim to get shape [B, T, num_features]
                outputs_concat = torch.cat(outputs, dim=-1)
                
                # Reconstruction loss per feature (weighted by inverse variance)
                loss = F.mse_loss(outputs_concat, features, reduction='none')  # [B, T, num_features]
                weights = torch.tensor(
                    [1.0 / (variances[i].item() + EPS) for i in range(model.num_features)],
                    device=device
                )  # shape: [num_features]
                weighted_loss = loss * weights  # broadcasting over B and T
                rec_loss = weighted_loss.mean()
                
                # Compute contrastive (center margin) loss on the projected latents.
                contrast_loss = center_margin_loss(proj_latents, rule_flags, margin=params.get('contrast_margin', 1.0))
                
                total_loss = rec_loss + contrast_lambda * contrast_loss
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
                pbar.set_postfix(loss=f"{total_loss.item():.4f}", rec_loss=f"{rec_loss.item():.4f}", contrast_loss=f"{contrast_lambda * contrast_loss.item():.4f}")
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

    # Create the training dataset (now including subtle anomalies).
    train_dataset = ParquetSequenceDataset(
        parquet_path=params['parquet_path'],
        feature_columns=params['feature_columns'],
        seq_len=params['seq_len'],
        stride=params['stride'],
        ratio=params['train_ratio'],
        seed=params['seed'],
        skip_anomalies=False,  # Include subtle anomalies during training.
        normalization_stats=None  # Initially load raw data for computing stats.
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    # Compute feature statistics from the raw training data.
    num_features = len(params['feature_columns'])
    means, variances = compute_features_statistics(train_loader, params, num_features)
    
    # Reinitialize the dataset with normalization enabled.
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
        'validation_parquet_path': 'unscaled_pdsch_val_min.parquet', 
        'output_dir': './output',
        'feature_columns': ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"],
        'seq_len': 20,
        'stride': 10,
        'train_ratio': 0.1,
        'val_ratio': 1.0,
        'seed': 42,
        'batch_size': 64,
        'num_epochs': 3,
        'lr': 5e-4,
        'hidden_dim': 128,  # Reduced to enforce a stronger bottleneck.
        'percentile': 95,
        'dropout': 0.5,
        'pca_n_components': 2,
        'features_stats_json': 'features_stats.json',
        'skip_anomalies': False,  # Do not skip rule-based anomalies
        'contrast_lambda': 1.0,   # Weight for contrastive loss term
        'contrast_margin': 2.0,   # Margin for the contrastive loss
        'contrast_beta': 0.5,     # Weight for adding contrastive anomaly score at inference
    }

    start_logging(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = len(params['feature_columns'])
    model = NumericalAutoencoder(num_features=num_features, hidden_dim=params['hidden_dim'], dropout=params['dropout'])
    model.to(device)
    model, params = main_training_pipeline(params, model)
