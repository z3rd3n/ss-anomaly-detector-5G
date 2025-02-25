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
from model import Model, WeightedFocalLoss

def train_classifier(model, train_loader, optimizer, device, params, means, variances):
    num_epochs = params.get('num_epochs', 10)
    best_val_class_acc = 0.0

    # Use class counts from the dataset to compute inverse frequencies.
    counts_dataset = train_loader.dataset.counts
    logging.info(f"Class counts: {counts_dataset}")

    weights = model.get_loss_weights(counts_dataset).to(device)

    focal_loss = WeightedFocalLoss(weights)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss_epoch = 0.0
        total_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            features = batch['features'].to(device)    # shape: [B, T, num_features]
            true_labels = batch['labels'].to(device)     # shape: [B, T]
            
            class_logits = model(features)
            total_loss = focal_loss(class_logits, true_labels)
            
            total_loss.backward()
            optimizer.step()
            
            # Compute batch-level accuracy metrics.
            pred_labels = torch.argmax(class_logits, dim=-1)
            
            normal_mask = (true_labels == 0)
            anomaly_mask = (true_labels != 0)
            
            normal_total = normal_mask.sum().item()
            anomaly_total = anomaly_mask.sum().item()
            normal_correct = (pred_labels[normal_mask] == true_labels[normal_mask]).sum().item() if normal_total > 0 else 0
            anomaly_correct = (pred_labels[anomaly_mask] == true_labels[anomaly_mask]).sum().item() if anomaly_total > 0 else 0
            
            normal_acc = normal_correct / normal_total if normal_total > 0 else 0.0
            anomaly_acc = anomaly_correct / anomaly_total if anomaly_total > 0 else 0.0
            
            total_loss_epoch += total_loss.item()
            total_batches += 1
            pbar.set_postfix({
                "Loss": f"{total_loss.item():.4f}",
                "Acc_Normal": f"{normal_acc*100:.2f}%",
                "Acc_Anomaly": f"{anomaly_acc*100:.2f}%"
            })
            
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
        # Create the training dataset (map-style dataset with oversampling).
        train_dataset = ParquetSequenceDataset(
            parquet_path=params['parquet_path'],
            feature_columns=params['feature_columns'],
            seq_len=params['seq_len'],
            stride=params['stride'],
            ratio=params['train_ratio'],
            seed=params['seed'],
            skip_anomalies=False,
            normalization_stats=None
        )

        # Compute per–feature statistics.
        num_features = len(params['feature_columns'])
        means, variances = compute_features_statistics(train_dataset, params, num_features)
        # Reinitialize dataset with normalization.
        #balanced_dataset.normalization_stats = {'means': means, 'variances': variances}
        
        log_model_size(model, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 1e-3))

        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            collate_fn=custom_collate_fn,
        )
        
        model, best_acc = train_classifier(model, train_loader, optimizer, device, params, means, variances)
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

    return model, params

if __name__ == '__main__':
    # Hyperparameters and settings.
    params = {
        'parquet_path': 'unscaled_pdsch_val.parquet',
        'validation_parquet_path': 'unscaled_pdsch_val_min.parquet', 
        'output_dir': './output',
        'feature_columns': ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"],
        'seq_len': 100,
        'stride': 100,
        'train_ratio': 1.0,
        'val_ratio': 1.0,
        'seed': 42,
        'batch_size': 16,
        'num_epochs': 1,
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
    feature_ranges = [1023, 30, 15, 32, 1, 8, 1]

    model = Model(feature_ranges)
    model.to(device)

    model, params = main_training_pipeline(params, model, params['train'])
    logging.info("Training complete.")
