import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report

from torch.utils.data import DataLoader
import json
import logging
from datetime import datetime

from dataset import ParquetSequenceDataset, OverSampledSequenceDataset, custom_collate_fn
from utils import (save_checkpoint, start_logging, validate_csv, 
                  compute_features_statistics, log_model_size, EPS, load_checkpoint)
from model import TwoPhaseModel, WeightedFocalLoss

def train_autoencoder(model, train_loader, optimizer, device, params, epoch, num_epochs):
    """
    Train the autoencoder phase of the model
    
    Args:
        model: TwoPhaseModel instance
        train_loader: DataLoader for training data (normal instances only)
        optimizer: Optimizer instance
        device: Device to train on
        params: Parameter dictionary
        epoch: Current epoch
        num_epochs: Total number of epochs
    
    Returns:
        avg_loss: Average loss for this epoch
    """
    model.train()
    model.freeze_classifier()  # Only train autoencoder
    
    total_loss_epoch = 0.0
    total_batches = 0
    
    pbar = tqdm(train_loader, desc=f"AE Epoch {epoch}/{num_epochs}", unit="batch")
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        # Get features
        features = batch['features'].to(device)  # shape: [B, T, num_features]
        
        # Embed features
        embedded = model.feature_embedding(features)
        
        # Get reconstruction
        reconstructed, _ = model.autoencoder(embedded)
        
        # Compute reconstruction loss
        recon_loss = F.mse_loss(reconstructed, embedded)
        
        # Add regularization if needed
        if params.get('use_l1_reg', False):
            l1_lambda = params.get('l1_lambda', 1e-5)
            l1_reg = sum(p.abs().sum() for p in model.autoencoder.parameters())
            recon_loss += l1_lambda * l1_reg
        
        recon_loss.backward()
        
        # Gradient clipping
        if params.get('clip_grad', True):
            max_grad_norm = params.get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # Update progress bar
        total_loss_epoch += recon_loss.item()
        total_batches += 1
        pbar.set_postfix({
            "Loss": f"{recon_loss.item():.4f}",
        })
    
    avg_loss = total_loss_epoch / total_batches
    logging.info(f"Autoencoder Epoch {epoch} Average Loss: {avg_loss:.6f}")
    
    return avg_loss

def train_classifier(model, train_loader, optimizer, device, params, epoch, num_epochs, means, variances):
    """
    Train the classifier phase of the model
    
    Args:
        model: TwoPhaseModel instance
        train_loader: DataLoader for training data
        optimizer: Optimizer instance
        device: Device to train on
        params: Parameter dictionary
        epoch: Current epoch
        num_epochs: Total number of epochs
        means: Feature means for normalization
        variances: Feature variances for normalization
    
    Returns:
        avg_loss: Average loss for this epoch
    """
    model.train()
    model.freeze_autoencoder()  # Only train classifier
    
    total_loss_epoch = 0.0
    total_batches = 0
    all_preds = []
    all_labels = []
    
    # Use class counts from the dataset to compute class weights
    counts_dataset = train_loader.dataset.counts
    logging.info(f"Class counts: {counts_dataset}")
    
    # Get class weights and create loss function
    weights = model.get_loss_weights(counts_dataset).to(device)
    
    # Use focal loss for better handling of class imbalance
    gamma = params.get('focal_gamma', 2.0)
    alpha = params.get('focal_alpha', 0.25)
    focal_loss = WeightedFocalLoss(weights, gamma=gamma, alpha=alpha)
    
    pbar = tqdm(train_loader, desc=f"Classifier Epoch {epoch}/{num_epochs}", unit="batch")
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        # Get features and labels
        features = batch['features'].to(device)    # shape: [B, T, num_features]
        true_labels = batch['labels'].to(device)   # shape: [B, T]
        
        # Forward pass
        class_logits = model(features)
        
        # Compute loss
        loss = focal_loss(class_logits, true_labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if params.get('clip_grad', True):
            max_grad_norm = params.get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # Compute batch metrics
        pred_labels = torch.argmax(class_logits, dim=-1)
        all_preds.append(pred_labels.detach().cpu())
        all_labels.append(true_labels.detach().cpu())
        
        # Compute and display accuracy metrics
        normal_mask = (true_labels == 0)
        anomaly_mask = (true_labels != 0)
        
        normal_total = normal_mask.sum().item()
        anomaly_total = anomaly_mask.sum().item()
        normal_correct = (pred_labels[normal_mask] == true_labels[normal_mask]).sum().item() if normal_total > 0 else 0
        anomaly_correct = (pred_labels[anomaly_mask] == true_labels[anomaly_mask]).sum().item() if anomaly_total > 0 else 0
        
        normal_acc = normal_correct / normal_total if normal_total > 0 else 0.0
        anomaly_acc = anomaly_correct / anomaly_total if anomaly_total > 0 else 0.0
        
        total_loss_epoch += loss.item()
        total_batches += 1
        
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc_Normal": f"{normal_acc*100:.1f}%",
            "Acc_Anomaly": f"{anomaly_acc*100:.1f}%"
        })
    
    # Compute overall metrics
    all_preds = torch.cat(all_preds, dim=0).view(-1)
    all_labels = torch.cat(all_labels, dim=0).view(-1)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())
    logging.info(f"Confusion Matrix:\n{cm}")
    
    # Compute per-class metrics
    report = classification_report(
        all_labels.numpy(), all_preds.numpy(), 
        labels=list(range(model.classifier.num_classes)),
        zero_division=0
    )
    logging.info(f"Classification Report:\n{report}")
    
    avg_loss = total_loss_epoch / total_batches
    logging.info(f"Classifier Epoch {epoch} Average Loss: {avg_loss:.6f}")
    
    # Validate and report metrics
    val_metrics = validate_model(model, params, means, variances, device)
    logging.info(f"Validation Metrics: {val_metrics}")
    
    return avg_loss, val_metrics.get('balanced_acc', 0.0)

def validate_model(model, params, means, variances, device):
    """
    Validate the model on validation data
    
    Args:
        model: TwoPhaseModel instance
        params: Parameter dictionary
        means: Feature means for normalization
        variances: Feature variances for normalization
        device: Device to validate on
    
    Returns:
        metrics: Dictionary of validation metrics
    """
    logging.info("Starting validation procedure...")
    model.eval()
    
    # Create validation dataset
    val_dataset = ParquetSequenceDataset(
        parquet_path=params['validation_parquet_path'],
        feature_columns=params['feature_columns'],
        seq_len=params['seq_len'],
        stride=params['val_stride'],
        ratio=params['val_ratio'],
        seed=params['seed'],
        skip_anomalies=False,
        normalization_stats={'means': means, 'variances': variances}
    )
    
    # Create validation dataloader
    val_loader = DataLoader(
        val_dataset, 
        batch_size=params['batch_size'], 
        shuffle=False, 
        collate_fn=custom_collate_fn
    )
    
    # Track predictions by timestamp for evaluation
    pred_by_timestamp = {}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", unit="batch"):
            features = batch['features'].to(device)  # [B, T, num_features]
            true_labels = batch['labels']            # [B, T]
            
            # Get predictions
            class_logits = model(features)
            pred_labels = torch.argmax(class_logits, dim=-1).cpu()  # [B, T]
            
            # Store predictions by timestamp
            for ts_list, t_labels, p_labels in zip(batch['timestamps'], true_labels, pred_labels):
                for ts, t_label, p_label in zip(ts_list, t_labels.tolist(), p_labels.tolist()):
                    pred_by_timestamp[ts] = (t_label, p_label)
    
    if not pred_by_timestamp:
        logging.warning("No predictions made during validation!")
        return {"balanced_acc": 0.0}
    
    # Compute metrics
    all_true = []
    all_pred = []
    for t_label, p_label in pred_by_timestamp.values():
        all_true.append(t_label)
        all_pred.append(p_label)
    
    all_true = torch.tensor(all_true)
    all_pred = torch.tensor(all_pred)
    
    # Compute per-class metrics
    class_labels = torch.unique(all_true)
    class_metrics = {}
    
    for label in class_labels:
        mask = (all_true == label)
        total = mask.sum().item()
        correct = (all_true[mask] == all_pred[mask]).sum().item()
        accuracy = correct / total if total > 0 else 0.0
        class_metrics[label.item()] = {"correct": correct, "total": total, "accuracy": accuracy}
    
    # Log per-class metrics
    anomaly_mapping = {v: k for k, v in val_dataset.anomaly_mapping.items()}
    anomaly_mapping[0] = "normal"
    
    for label, metrics in class_metrics.items():
        class_name = anomaly_mapping.get(label, f"Class {label}")
        logging.info(f"{class_name}: {metrics['correct']}/{metrics['total']} ({metrics['accuracy']*100:.2f}%)")
    
    # Compute normal vs anomaly metrics
    normal_mask = (all_true == 0)
    anomaly_mask = (all_true != 0)
    
    normal_total = normal_mask.sum().item()
    anomaly_total = anomaly_mask.sum().item()
    
    normal_correct = (all_true[normal_mask] == all_pred[normal_mask]).sum().item() if normal_total > 0 else 0
    anomaly_correct = (all_true[anomaly_mask] == all_pred[anomaly_mask]).sum().item() if anomaly_total > 0 else 0
    
    normal_acc = normal_correct / normal_total if normal_total > 0 else 0.0
    anomaly_acc = anomaly_correct / anomaly_total if anomaly_total > 0 else 0.0
    balanced_acc = (normal_acc + anomaly_acc) / 2.0
    
    logging.info(f"Normal Instances: {normal_correct}/{normal_total} ({normal_acc*100:.2f}%)")
    logging.info(f"Anomaly Instances: {anomaly_correct}/{anomaly_total} ({anomaly_acc*100:.2f}%)")
    logging.info(f"Balanced Accuracy: {balanced_acc*100:.2f}%")
    
    # Return metrics
    return {
        "normal_acc": normal_acc,
        "anomaly_acc": anomaly_acc,
        "balanced_acc": balanced_acc,
        "class_metrics": class_metrics
    }

def train_two_phase_model(model, params, means=None, variances=None):
    """
    Complete training pipeline for the two-phase model
    
    Args:
        model: TwoPhaseModel instance
        params: Parameter dictionary
        means: Optional feature means for normalization
        variances: Optional feature variances for normalization
    
    Returns:
        model: Trained model
        best_metrics: Best validation metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() and params.get('use_cuda', True) else "cpu")
    logging.info(f"Using device: {device}")
    model.to(device)
    
    # Log model size
    log_model_size(model)
    
    # Compute or load feature statistics if not provided
    if means is None or variances is None:
        # Create dataset for computing statistics
        stats_dataset = ParquetSequenceDataset(
            parquet_path=params['parquet_path'],
            feature_columns=params['feature_columns'],
            seq_len=params['seq_len'],
            stride=params['stride'],
            ratio=params.get('stats_ratio', 0.1),  # Use subset for efficiency
            seed=params['seed'],
            skip_anomalies=False,
            normalization_stats=None,
            pre_normalize=False
        )
        
        # Compute statistics
        num_features = len(params['feature_columns'])
        means, variances = compute_features_statistics(stats_dataset, params, num_features)
        
        # Save statistics
        stats_file = params['features_stats_json']
        with open(stats_file, 'w') as f:
            json.dump({
                'means': means.tolist(),
                'variances': variances.tolist()
            }, f)
        
        logging.info(f"Feature statistics saved to {stats_file}")
    
    # Create dataset for autoencoder training (normal instances only)
    logging.info("Creating autoencoder training dataset (normal instances only)...")
    ae_dataset = ParquetSequenceDataset(
        parquet_path=params['parquet_path'],
        feature_columns=params['feature_columns'],
        seq_len=params['seq_len'],
        stride=params['stride'],
        ratio=params['train_ratio'],
        seed=params['seed'],
        skip_anomalies=True,  # Only normal instances
        normalization_stats={'means': means, 'variances': variances}
    )
    
    ae_loader = DataLoader(
        ae_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=params.get('num_workers', 0)
    )
    
    # Create optimizer for autoencoder phase
    ae_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.get('ae_lr', 1e-3),
        weight_decay=params.get('ae_weight_decay', 1e-6)
    )
    
    # Train autoencoder phase
    logging.info("Starting autoencoder training phase...")
    ae_epochs = params.get('ae_epochs', 10)
    best_ae_loss = float('inf')
    
    for epoch in range(1, ae_epochs + 1):
        avg_loss = train_autoencoder(model, ae_loader, ae_optimizer, device, params, epoch, ae_epochs)
        
        if avg_loss < best_ae_loss:
            best_ae_loss = avg_loss
            # Save autoencoder checkpoint
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': ae_optimizer.state_dict(),
                'best_ae_loss': best_ae_loss,
                'params': params,
            }
            ae_checkpoint_path = os.path.join(params['output_dir'], 'best_autoencoder.pt')
            save_checkpoint(checkpoint_state, ae_checkpoint_path)
    
    # Load best autoencoder checkpoint
    ae_checkpoint_path = os.path.join(params['output_dir'], 'best_autoencoder.pt')
    _, best_ae_loss, _ = load_checkpoint(ae_checkpoint_path, model, ae_optimizer)
    logging.info(f"Loaded best autoencoder model with loss: {best_ae_loss:.6f}")
    
    # Create dataset for classifier training (with oversampling)
    logging.info("Creating classifier training dataset with oversampling...")
    
    if params.get('use_oversampling', True):
        # Use oversampled dataset
        cls_dataset = OverSampledSequenceDataset(
            parquet_path=params['parquet_path'],
            feature_columns=params['feature_columns'],
            seq_len=params['seq_len'],
            stride=params['stride'],
            ratio=params['train_ratio'],
            seed=params['seed'],
            normalization_stats={'means': means, 'variances': variances},
            oversample_factors=params.get('oversample_factors', {1: 2.0, 2: 10.0, 3: 2.0, 4: 10.0})
        )
    else:
        # Use regular dataset
        cls_dataset = ParquetSequenceDataset(
            parquet_path=params['parquet_path'],
            feature_columns=params['feature_columns'],
            seq_len=params['seq_len'],
            stride=params['stride'],
            ratio=params['train_ratio'],
            seed=params['seed'],
            skip_anomalies=False,
            normalization_stats={'means': means, 'variances': variances}
        )
    
    cls_loader = DataLoader(
        cls_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=params.get('num_workers', 0)
    )
    
    # Create optimizer for classifier phase
    cls_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.get('cls_lr', 5e-4),
        weight_decay=params.get('cls_weight_decay', 1e-6)
    )
    
    # Train classifier phase
    logging.info("Starting classifier training phase...")
    cls_epochs = params.get('cls_epochs', 20)
    best_val_acc = 0.0
    best_metrics = None
    
    # Learning rate scheduler
    lr_scheduler = None
    if params.get('use_lr_scheduler', True):
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            cls_optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3, 
            verbose=True
        )
    
    for epoch in range(1, cls_epochs + 1):
        avg_loss, val_acc = train_classifier(
            model, cls_loader, cls_optimizer, device, params, 
            epoch, cls_epochs, means, variances
        )
        
        # Update learning rate if using scheduler
        if lr_scheduler is not None:
            lr_scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Validate to get detailed metrics
            best_metrics = validate_model(model, params, means, variances, device)
            
            # Save classifier checkpoint
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': cls_optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'best_metrics': best_metrics,
                'params': params,
            }
            cls_checkpoint_path = os.path.join(params['output_dir'], 'best_classifier.pt')
            save_checkpoint(checkpoint_state, cls_checkpoint_path)
    
    # Final joint fine-tuning (optional)
    if params.get('do_fine_tuning', False):
        logging.info("Starting fine-tuning phase (joint training)...")
        
        # Unfreeze all parameters
        model.unfreeze_all()
        
        # Create optimizer for fine-tuning
        ft_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params.get('ft_lr', 1e-5),
            weight_decay=params.get('ft_weight_decay', 1e-6)
        )
        
        # Fine-tuning epochs
        ft_epochs = params.get('ft_epochs', 5)
        
        for epoch in range(1, ft_epochs + 1):
            # Train with classifier loss
            avg_loss, val_acc = train_classifier(
                model, cls_loader, ft_optimizer, device, params,
                epoch, ft_epochs, means, variances
            )
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Validate to get detailed metrics
                best_metrics = validate_model(model, params, means, variances, device)
                
                # Save fine-tuned checkpoint
                checkpoint_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': ft_optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'best_metrics': best_metrics,
                    'params': params,
                }
                ft_checkpoint_path = os.path.join(params['output_dir'], 'best_fine_tuned.pt')
                save_checkpoint(checkpoint_state, ft_checkpoint_path)
    
    # Load best model
    if params.get('do_fine_tuning', False):
        best_checkpoint_path = os.path.join(params['output_dir'], 'best_fine_tuned.pt')
    else:
        best_checkpoint_path = os.path.join(params['output_dir'], 'best_classifier.pt')
    
    # Create dummy optimizer for loading
    dummy_optimizer = torch.optim.Adam(model.parameters())
    _, best_val_acc, _ = load_checkpoint(best_checkpoint_path, model, dummy_optimizer)
    logging.info(f"Loaded best model with validation accuracy: {best_val_acc:.6f}")
    
    return model, best_metrics

def run_training_pipeline(params):
    """
    Complete training pipeline from start to finish
    
    Args:
        params: Parameter dictionary
    
    Returns:
        model: Trained model
        best_metrics: Best validation metrics
    """
    # Initialize logging
    start_logging(params)
    
    # Set random seeds for reproducibility
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(params['seed'])
    
    # Create directories
    os.makedirs(params['output_dir'], exist_ok=True)
    
    # Save parameters
    params_file = os.path.join(params['output_dir'], 'params.json')
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)
    
    # Feature ranges
    feature_ranges = params.get('feature_ranges', [1023, 30, 15, 32, 1, 8, 1])
    
    # Create model
    model = TwoPhaseModel(
        feature_ranges=feature_ranges,
        params=params
    )
    
    # Train model
    if params.get('train', True):
        model, best_metrics = train_two_phase_model(model, params)
        return model, best_metrics
    else:
        # Load model from checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() and params.get('use_cuda', True) else "cpu")
        model.to(device)
        
        # Load stats
        stats_file = params['features_stats_json']
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        means = torch.tensor(stats['means'], dtype=torch.float32)
        variances = torch.tensor(stats['variances'], dtype=torch.float32)
        
        # Load best model
        best_checkpoint_path = os.path.join(params['output_dir'], 'best_classifier.pt')
        optimizer = torch.optim.Adam(model.parameters())
        _, best_val_acc, _ = load_checkpoint(best_checkpoint_path, model, optimizer)
        
        # Validate
        metrics = validate_model(model, params, means, variances, device)
        
        return model, metrics

if __name__ == '__main__':
    # Hyperparameters and settings
    params = {
        # Data paths
        'parquet_path': 'unscaled_pdsch_val.parquet',
        'validation_parquet_path': 'unscaled_pdsch_val_min.parquet',
        'output_dir': './output',
        'features_stats_json': 'features_stats.json',
        
        # Feature configuration
        'feature_columns': ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"],
        'feature_ranges': [1023, 30, 15, 32, 1, 8, 1],
        
        # Dataset parameters
        'seq_len': 100,
        'stride': 50,  # Overlapping sequences to catch boundary anomalies
        'val_stride': 100,
        'train_ratio': 0.1,
        'val_ratio': 1.0,
        'seed': 42,
        'batch_size': 32,
        'num_workers': 4,
        
        # Oversampling configuration
        'use_oversampling': True,
        'oversample_factors': {
            1: 5.0,   # unnecessary_retx: moderate oversampling
            2: 20.0,  # missing_retx: heavy oversampling (very rare)
            3: 5.0,   # new_data_no_retx: moderate oversampling
            4: 20.0,  # max_retx_achieved: heavy oversampling (very rare)
        },
        
        # Model parameters
        'embedding_dim': 16,
        'hidden_dim': 128,
        'latent_dim': 64,
        'dropout': 0.3,
        'num_anomaly_classes': 5,  # 4 types + "none of them"
        
        # Autoencoder training parameters
        'ae_epochs': 1,
        'ae_lr': 1e-3,
        'ae_weight_decay': 1e-6,
        'use_l1_reg': True,
        'l1_lambda': 1e-5,
        
        # Classifier training parameters
        'cls_epochs': 3,
        'cls_lr': 5e-4,
        'cls_weight_decay': 1e-6,
        'focal_gamma': 2.5,
        'focal_alpha': 0.25,
        
        # Fine-tuning parameters
        'do_fine_tuning': False,
        'ft_epochs': 5,
        'ft_lr': 1e-5,
        'ft_weight_decay': 1e-6,
        
        # Training control
        'use_lr_scheduler': True,
        'clip_grad': True,
        'max_grad_norm': 1.0,
        'train': True,
        'use_cuda': True,
    }
    
    # Run training pipeline
    model, best_metrics = run_training_pipeline(params)
    
    # Log final results
    logging.info("Training complete!")
    logging.info(f"Best validation metrics: {best_metrics}")