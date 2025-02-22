# trainer.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import ParquetSequenceDataset, custom_collate_fn
from torch.utils.data import DataLoader
import json
import logging

from utils import (save_checkpoint, start_logging, validate_csv, 
                   compute_features_statistics, log_model_size, EPS, load_checkpoint)
from model import Model

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Args:
            gamma: Focusing parameter.
            alpha: Tensor of shape [num_classes] with per-class weights.
                   For your case, set alpha[0] = 0 (or a very small number) 
                   and alpha for anomalies based on inverse frequency.
            reduction: 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # if None, defaults to 1 for all classes
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate the log probabilities.
        logpt = -F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(logpt)

        # Get the alpha for each sample: if alpha is provided, index it by targets.
        if self.alpha is not None:
            at = self.alpha[targets]
        else:
            at = 1.0

        loss = -at * ((1 - pt) ** self.gamma) * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def train_anomaly_detector(model, train_loader, optimizer, device, params, means, variances):
    num_epochs = params.get('num_epochs', 10)
    best_val_class_acc = 0.0

    counts_dataset = train_loader.dataset.counts
    counts = torch.tensor([counts_dataset[i] for i in range(len(counts_dataset))], dtype=torch.float)
    logging.info(f"Class counts: {counts.tolist()}")

    # Compute inverse frequencies (add a small epsilon if needed).
    inv_freq = 1.0 / (counts.float() + 1e-8)

    # Normalize the weights so they sum to 1 (or scale them appropriately).
    alpha = inv_freq / inv_freq.sum()

    # Move alpha to the device.
    alpha = alpha.to(device)

    # Then initialize your loss:
    focal_loss_fn = FocalLoss(gamma=2.0, alpha=alpha, reduction='mean')

    
    mse_loss_fn = nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss_epoch = 0.0
        total_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            features = batch['features'].to(device)    # shape: [B, T, num_features]
            true_labels = batch['labels'].to(device)     # shape: [B, T]

            # Forward pass.
            recon, class_logits = model(features)  # recon: [B, T, num_features], logits: [B, T, num_classes]
            recon = torch.cat(recon, dim=-1)  # [B, T, num_features]
            
            # Reconstruction loss.
            recon_loss = mse_loss_fn(recon, features)
            
            # Classification loss computed per time step.
            clf_loss = focal_loss_fn(class_logits.view(-1, len(counts)), true_labels.view(-1))
            
            # Total loss is the sum (you may weight each term as needed).
            total_loss = recon_loss + clf_loss
            total_loss.backward()
            optimizer.step()
            
            total_loss_epoch += total_loss.item()
            total_batches += 1
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}",
                              "Recon": f"{recon_loss.item():.4f}",
                              "Clf": f"{clf_loss.item():.4f}"})
            
        avg_loss = total_loss_epoch / total_batches
        logging.info(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
        # Validate and report per–instance metrics.
        val_class_acc = validate_csv(model, params, means, variances, device)
        logging.info(f"Validation Classification Accuracy: {val_class_acc*100:.2f}%")
        
        if val_class_acc > best_val_class_acc:
            best_val_class_acc = val_class_acc
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_class_acc': best_val_class_acc,
                'params': params,
            }
            filename = os.path.join(params['output_dir'], 'best_model.pt')
            save_checkpoint(checkpoint_state, filename)
    return model, best_val_class_acc

def main_training_pipeline(params, model, train=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if train:
        # Create the training dataset (including anomalies and per–timestamp labels).
        train_dataset = ParquetSequenceDataset(
            parquet_path=params['parquet_path'],
            feature_columns=params['feature_columns'],
            seq_len=params['seq_len'],
            stride=params['stride'],
            ratio=params['train_ratio'],
            seed=params['seed'],
            skip_anomalies=False,  # include all instances for training
            normalization_stats=None  # initially unnormalized to compute stats
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        # Compute per–feature statistics.
        num_features = len(params['feature_columns'])
        means, variances = compute_features_statistics(train_loader, params, num_features)
        # Reinitialize dataset with normalization.
        train_dataset.normalization_stats = {'means': means, 'variances': variances}
        
        log_model_size(model, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 1e-3))
        model, best_acc = train_anomaly_detector(model, train_loader, optimizer, device, params, means, variances)
        logging.info(f"Training complete. Best Validation Classification Accuracy: {best_acc:.4f}")
    else:
        # Load the model from the checkpoint.
        checkpoint_path = os.path.join(params['output_dir'], 'best_model.pt')
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 1e-3))
        _, best_acc, params = load_checkpoint(checkpoint_path, model, optimizer)
        logging.info(f"Loaded model for validation. Best Validation Accuracy: {best_acc:.4f}")

        # Load normalization statistics.
        stats_file = params['features_stats_json']
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        means = torch.tensor(stats['means'], dtype=torch.float32)
        variances = torch.tensor(stats['variances'], dtype=torch.float32)

    # Perform validation.
    val_class_acc = validate_csv(model, params, means, variances, device)
    logging.info(f"Final Classification Accuracy: {val_class_acc*100:.2f}%")
    return model, params

if __name__ == '__main__':
    # Hyperparameters and settings.
    params = {
        'parquet_path': 'unscaled_pdsch_val.parquet',
        'validation_parquet_path': 'unscaled_pdsch_val_min.parquet', 
        'output_dir': './output',
        'feature_columns': ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"],
        'seq_len': 20,
        'stride': 5,
        'train_ratio': 0.01, # less than 0.2, not enough class
        'val_ratio': 1.0,
        'seed': 42,
        'batch_size': 64,
        'num_epochs': 3,
        'lr': 5e-4,
        'hidden_dim': 64,
        'dropout': 0.5,
        'pca_n_components': 2,
        'features_stats_json': 'features_stats.json',
        'skip_anomalies': False,
        'train': True,
    }

    start_logging(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = len(params['feature_columns'])
    model = Model(num_features=num_features, hidden_dim=params['hidden_dim'], dropout=params['dropout'])
    model.to(device)

    model, params = main_training_pipeline(params, model, params['train'])
    logging.info("Training complete.")
