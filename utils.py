import os
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
import torch.nn.functional as F
import pandas as pd

EPS = 1e-8  # Small constant to avoid division by zero

def start_logging(params=None):
    """
    Set up logging to both file and console
    
    Args:
        params: Optional parameter dictionary
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directories
    output_dir = params['output_dir'] if params and 'output_dir' in params else 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up log file
    log_file = os.path.join(log_dir, f'training_{current_time}.log')
    
    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    
    # Add console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    # Log parameters if provided
    if params:
        logging.info("Hyperparameters and settings:")
        for key, value in params.items():
            if key != 'feature_ranges' and key != 'oversample_factors':
                logging.info(f"  {key}: {value}")
            else:
                # Format these more compactly
                logging.info(f"  {key}: {value}")

def save_checkpoint(state: dict, filename: str):
    """
    Save model checkpoint to file
    
    Args:
        state: Dictionary containing model state and metadata
        filename: File path to save to
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    logging.info(f"Checkpoint saved to {filename}")

def load_checkpoint(filename: str, model: nn.Module, optimizer: torch.optim.Optimizer):
    """
    Load model checkpoint from file
    
    Args:
        filename: File path to load from
        model: Model to load state into
        optimizer: Optimizer to load state into
    
    Returns:
        epoch: Epoch number from checkpoint
        best_metric: Best metric from checkpoint
        params: Parameters from checkpoint
    """
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint and optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                logging.warning("Could not load optimizer state - continuing without it")
        
        epoch = checkpoint.get('epoch', 0)
        
        # Try to get best metric from various keys
        if 'best_val_class_acc' in checkpoint:
            best_metric = checkpoint['best_val_class_acc']
        elif 'best_val_acc' in checkpoint:
            best_metric = checkpoint['best_val_acc']
        elif 'best_ae_loss' in checkpoint:
            best_metric = checkpoint['best_ae_loss']
        else:
            best_metric = 0.0
        
        params = checkpoint.get('params', {})
        
        logging.info(f"Loaded checkpoint '{filename}' (epoch {epoch}) with best metric: {best_metric:.4f}")
        return epoch, best_metric, params
    else:
        logging.warning(f"No checkpoint found at '{filename}'")
        return 0, 0.0, {}

def compute_features_statistics(dataset, params, num_features):
    """
    Compute mean and variance of features for normalization
    
    Args:
        dataset: Dataset to compute statistics from
        params: Parameter dictionary
        num_features: Number of features
    
    Returns:
        means: Feature means tensor
        variances: Feature variances tensor
    """
    stats_file = params['features_stats_json']
    
    # Check if stats already exist
    if os.path.exists(stats_file):
        logging.info(f"Loading features statistics from {stats_file}")
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        means = torch.tensor(stats['means'], dtype=torch.float32)
        variances = torch.tensor(stats['variances'], dtype=torch.float32)
        return means, variances
    
    logging.info("Computing features statistics from training data...")
    
    # Create data loader for computing statistics
    from torch.utils.data import DataLoader
    from dataset import custom_collate_fn
    
    data_loader = DataLoader(
        dataset, 
        batch_size=params.get('batch_size', 32),
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    # Compute statistics
    total_sum = torch.zeros(num_features, dtype=torch.float64)
    total_sum_sq = torch.zeros(num_features, dtype=torch.float64)
    count = 0
    
    for batch in tqdm(data_loader, desc="Computing statistics", unit="batch"):
        features = batch['features'].to(torch.float64)  # [B, T, F]
        
        # Handle variable sequence lengths (in case of padding)
        if 'mask' in batch:
            mask = batch['mask']
            B, T, F = features.shape
            features_flat = features.reshape(-1, F)[mask.reshape(-1)]
            count += features_flat.shape[0]
            total_sum += features_flat.sum(dim=0)
            total_sum_sq += (features_flat**2).sum(dim=0)
        else:
            B, T, F = features.shape
            count += B * T
            total_sum += features.sum(dim=(0,1))
            total_sum_sq += (features**2).sum(dim=(0,1))
    
    # Calculate mean and variance
    means = (total_sum / count).to(torch.float32)
    variances = ((total_sum_sq / count) - (means.double()**2)).to(torch.float32)
    
    # Ensure positive variances
    variances = torch.clamp(variances, min=EPS)
    
    # Save statistics
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    stats = {'means': means.tolist(), 'variances': variances.tolist()}
    with open(stats_file, 'w') as f:
        json.dump(stats, f)
    
    logging.info(f"Features statistics saved to {stats_file}")
    return means, variances

def unnormalize_features(normalized_tensor, means, variances):
    """
    Unnormalize features (inverse of normalization)
    
    Args:
        normalized_tensor: Normalized features tensor
        means: Feature means tensor
        variances: Feature variances tensor
    
    Returns:
        unnormalized_tensor: Unnormalized features tensor
    """
    std = torch.sqrt(variances + EPS)
    unnorm = normalized_tensor.float() * std + means
    return torch.clamp(torch.round(unnorm), min=0).to(torch.int64)

def validate_csv(model, params, means, variances, device):
    """
    Validate model on validation data and compute metrics
    
    Args:
        model: Model to validate
        params: Parameter dictionary
        means: Feature means tensor
        variances: Feature variances tensor
        device: Device to validate on
    
    Returns:
        balanced_acc: Balanced accuracy (average of normal and anomaly accuracies)
    """
    logging.info("Starting validation procedure...")
    model.eval()
    
    # Import necessary modules
    from dataset import ParquetSequenceDataset, custom_collate_fn
    from torch.utils.data import DataLoader
    
    # Create validation dataset
    val_dataset = ParquetSequenceDataset(
        parquet_path=params['validation_parquet_path'],
        feature_columns=params['feature_columns'],
        seq_len=params['seq_len'],
        stride=params.get('val_stride', params['seq_len']),
        ratio=params.get('val_ratio', 1.0),
        seed=params['seed'],
        skip_anomalies=False,
        normalization_stats={'means': means, 'variances': variances}
    )
    
    # Create validation loader
    val_loader = DataLoader(
        val_dataset, 
        batch_size=params.get('batch_size', 32), 
        shuffle=False, 
        collate_fn=custom_collate_fn
    )
    
    # Collect predictions and true labels
    pred_by_timestamp = {}
    all_reconstruction_errors = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", unit="batch"):
            features = batch['features'].to(device)  # [B, T, num_features]
            true_labels = batch['labels']            # [B, T]
            
            # Get predictions from model
            if hasattr(model, 'train_autoencoder') and not model.train_classifier:
                # Autoencoder-only mode
                embedded = model.feature_embedding(features)
                reconstructed, _ = model.autoencoder(embedded)
                recon_error = torch.mean(torch.abs(embedded - reconstructed), dim=-1)
                
                # Use reconstruction error threshold for anomaly detection
                threshold = params.get('recon_threshold', 0.1)
                pred_labels = (recon_error > threshold).long().cpu()
                
                # Save reconstruction errors for analysis
                all_reconstruction_errors.append(recon_error.cpu())
            else:
                # Full model with classifier
                class_logits = model(features)
                pred_labels = torch.argmax(class_logits, dim=-1).cpu()  # [B, T]
            
            # Store predictions by timestamp
            for ts_list, t_labels, p_labels in zip(batch['timestamps'], true_labels, pred_labels):
                for ts, t_label, p_label in zip(ts_list, t_labels.tolist(), p_labels.tolist()):
                    if ts:  # Skip empty timestamps (padding)
                        pred_by_timestamp[ts] = (t_label, p_label)
    
    if not pred_by_timestamp:
        logging.warning("No predictions made during validation!")
        return 0.0
    
    # Collate predictions and ground truth
    all_true = []
    all_pred = []
    for t_label, p_label in pred_by_timestamp.values():
        all_true.append(t_label)
        all_pred.append(p_label)
    
    all_true = torch.tensor(all_true)
    all_pred = torch.tensor(all_pred)
    
    # Compute per-class metrics
    class_labels = torch.unique(all_true)
    class_accuracies = {}
    
    for label in class_labels:
        mask = (all_true == label)
        total = mask.sum().item()
        correct = (all_true[mask] == all_pred[mask]).sum().item()
        accuracy = correct / total if total > 0 else 0.0
        class_accuracies[label.item()] = (correct, total, accuracy)
    
    # Get anomaly class names
    anomaly_mapping = {v: k for k, v in val_dataset.anomaly_mapping.items()}
    anomaly_mapping[0] = "normal"
    
    # Log per-class metrics
    for label, (correct, total, accuracy) in class_accuracies.items():
        class_name = anomaly_mapping.get(label, f"Class {label}")
        logging.info(f"{class_name}: {correct}/{total} ({accuracy*100:.2f}%)")
    
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
    
    # Compute F1 scores
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true.numpy(), all_pred.numpy(), 
        average=None, 
        labels=list(range(len(anomaly_mapping))),
        zero_division=0
    )
    
    # Log precision, recall, F1
    logging.info("Precision, Recall, F1 by class:")
    for i in range(len(precision)):
        class_name = anomaly_mapping.get(i, f"Class {i}")
        logging.info(f"{class_name}: P={precision[i]:.4f}, R={recall[i]:.4f}, F1={f1[i]:.4f}")
    
    # Compute confusion matrix
    cm = confusion_matrix(all_true.numpy(), all_pred.numpy())
    logging.info(f"Confusion Matrix:\n{cm}")
    
    # Plot and save confusion matrix if matplotlib is available
    try:
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = [anomaly_mapping.get(i, f"Class {i}") for i in range(len(anomaly_mapping))]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save figure
        cm_plot_path = os.path.join(params['output_dir'], 'confusion_matrix.png')
        plt.savefig(cm_plot_path)
        plt.close()
        logging.info(f"Confusion matrix saved to {cm_plot_path}")
    except Exception as e:
        logging.warning(f"Could not plot confusion matrix: {e}")
    
    # If we collected reconstruction errors, analyze them
    if all_reconstruction_errors:
        try:
            all_errors = torch.cat(all_reconstruction_errors).numpy()
            
            # Plot histogram of reconstruction errors
            plt.figure(figsize=(10, 6))
            plt.hist(all_errors, bins=50, alpha=0.7)
            plt.title('Reconstruction Error Distribution')
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Count')
            
            # Save figure
            error_plot_path = os.path.join(params['output_dir'], 'reconstruction_errors.png')
            plt.savefig(error_plot_path)
            plt.close()
            logging.info(f"Reconstruction error distribution saved to {error_plot_path}")
        except Exception as e:
            logging.warning(f"Could not plot reconstruction errors: {e}")
    
    return balanced_acc

def log_model_size(model: torch.nn.Module, device: torch.device = None) -> None:
    """
    Log model size and parameter counts
    
    Args:
        model: Model to analyze
        device: Optional device to move model to for analysis
    """
    if device is not None:
        model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Log parameter counts
    if total_params >= 1e6:
        logging.info(
            f"Model parameters: Total: {total_params/1e6:.2f}M, "
            f"Trainable: {trainable_params/1e6:.2f}M, "
            f"Non-trainable: {non_trainable_params/1e6:.2f}M"
        )
    else:
        logging.info(
            f"Model parameters: Total: {total_params/1e3:.2f}K, "
            f"Trainable: {trainable_params/1e3:.2f}K, "
            f"Non-trainable: {non_trainable_params/1e3:.2f}K"
        )
    
    # Estimate model size in memory
    bytes_per_param = 4  # assuming float32
    total_bytes = total_params * bytes_per_param
    size_mb = total_bytes / (1024**2)
    logging.info(f"Approximate model size: {size_mb:.2f} MB (assuming fp32)")
    
    # Log model architecture
    max_line_length = 80
    model_str = str(model)
    model_lines = model_str.split('\n')
    
    logging.info("Model architecture:")
    for line in model_lines:
        if len(line) > max_line_length:
            # Truncate long lines
            logging.info(f"  {line[:max_line_length-3]}...")
        else:
            logging.info(f"  {line}")

def plot_learning_curves(train_losses, val_metrics, output_dir):
    """
    Plot and save learning curves
    
    Args:
        train_losses: List of training losses
        val_metrics: List of validation metrics
        output_dir: Directory to save plots to
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, 'b-', label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        loss_plot_path = os.path.join(output_dir, 'training_loss.png')
        plt.savefig(loss_plot_path)
        plt.close()
        
        # Plot validation metrics
        if val_metrics:
            plt.figure(figsize=(10, 6))
            for metric_name, values in val_metrics.items():
                plt.plot(values, label=metric_name)
            
            plt.title('Validation Metrics')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save figure
            metrics_plot_path = os.path.join(output_dir, 'validation_metrics.png')
            plt.savefig(metrics_plot_path)
            plt.close()
            
        logging.info(f"Learning curves saved to {output_dir}")
    except Exception as e:
        logging.warning(f"Could not plot learning curves: {e}")

def plot_latent_space(model, val_loader, params, device=None):
    """
    Plot latent space representation using PCA or t-SNE
    
    Args:
        model: Model to extract latent representations from
        val_loader: DataLoader for validation data
        params: Parameter dictionary
        device: Device to run model on
    """
    try:
        if device is None:
            device = next(model.parameters()).device
        
        model.eval()
        all_latents = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Extracting latent representations", unit="batch"):
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)
                
                # Get latent representations
                if hasattr(model, 'train_autoencoder') and hasattr(model, 'autoencoder'):
                    # For TwoPhaseModel
                    embedded = model.feature_embedding(features)
                    _, latents = model.autoencoder(embedded)
                elif hasattr(model, 'encode'):
                    # For standalone autoencoder
                    latents = model.encode(features)
                else:
                    # Use forward with return_latents flag if available
                    if 'return_latents' in model.forward.__code__.co_varnames:
                        _, latents = model(features, return_latents=True)
                    else:
                        logging.warning("Could not extract latent representations from model")
                        return
                
                # Collect latents and labels
                all_latents.append(latents.cpu().numpy().reshape(latents.shape[0], -1))
                all_labels.append(labels.cpu().numpy().reshape(-1))
        
        if not all_latents:
            logging.warning("No latent representations found for plotting")
            return
        
        # Concatenate all latents and labels
        all_latents = np.concatenate(all_latents, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Filter out padding (-1) labels
        valid_mask = all_labels >= 0
        all_latents = all_latents[valid_mask]
        all_labels = all_labels[valid_mask]
        
        # Apply PCA
        n_components = params.get('pca_n_components', 2)
        pca = PCA(n_components=min(n_components, all_latents.shape[1]))
        latent_pca = pca.fit_transform(all_latents)
        
        # Get explained variance
        explained_variance = pca.explained_variance_ratio_
        
        # Create colormap for classes
        unique_labels = np.unique(all_labels)
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
        
        # Plot 2D or 3D visualization
        if n_components >= 3 and latent_pca.shape[1] >= 3:
            # 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            for i, label in enumerate(unique_labels):
                mask = all_labels == label
                ax.scatter(
                    latent_pca[mask, 0],
                    latent_pca[mask, 1],
                    latent_pca[mask, 2],
                    c=[colors[i]],
                    label=f'Class {label}',
                    alpha=0.7
                )
            
            ax.set_title('Latent Space (PCA, 3 Components)')
            ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)')
            ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)')
            ax.set_zlabel(f'PC3 ({explained_variance[2]*100:.1f}% variance)')
            ax.legend()
        else:
            # 2D plot
            fig, ax = plt.subplots(figsize=(12, 10))
            
            for i, label in enumerate(unique_labels):
                mask = all_labels == label
                ax.scatter(
                    latent_pca[mask, 0],
                    latent_pca[mask, 1] if latent_pca.shape[1] > 1 else np.zeros_like(latent_pca[mask, 0]),
                    c=[colors[i]],
                    label=f'Class {label}',
                    alpha=0.7
                )
            
            ax.set_title('Latent Space (PCA, 2 Components)')
            ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)')
            if latent_pca.shape[1] > 1:
                ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)')
            ax.legend()
        
        # Save figure
        latent_plot_path = os.path.join(params['output_dir'], 'latent_space.png')
        plt.savefig(latent_plot_path)
        plt.close()
        
        logging.info(f"Latent space visualization saved to {latent_plot_path}")
    except Exception as e:
        logging.warning(f"Could not plot latent space: {e}")

def diagnose_anomalies(model, val_loader, params, device=None, top_k=10):
    """
    Diagnose anomalies by analyzing examples with highest reconstruction errors
    
    Args:
        model: Model to use for diagnosis
        val_loader: DataLoader for validation data
        params: Parameter dictionary
        device: Device to run model on
        top_k: Number of top examples to analyze
    """
    try:
        if device is None:
            device = next(model.parameters()).device
        
        model.eval()
        errors_by_timestamp = {}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Diagnosing anomalies", unit="batch"):
                features = batch['features'].to(device)
                timestamps = batch['timestamps']
                
                # Get reconstruction from autoencoder
                if hasattr(model, 'train_autoencoder') and hasattr(model, 'autoencoder'):
                    # For TwoPhaseModel
                    embedded = model.feature_embedding(features)
                    reconstructed, _ = model.autoencoder(embedded)
                    recon_error = torch.mean(torch.abs(embedded - reconstructed), dim=-1)
                elif hasattr(model, 'decode'):
                    # For standalone autoencoder
                    reconstructed = model(features)
                    recon_error = torch.mean(torch.abs(features - reconstructed), dim=-1)
                else:
                    logging.warning("Could not compute reconstruction errors from model")
                    return
                
                # Store errors by timestamp
                for ts_list, errors in zip(timestamps, recon_error):
                    for ts, error in zip(ts_list, errors.cpu().numpy()):
                        if ts:  # Skip empty timestamps (padding)
                            errors_by_timestamp[ts] = error
        
        if not errors_by_timestamp:
            logging.warning("No reconstruction errors found for diagnosis")
            return
        
        # Sort timestamps by error
        sorted_timestamps = sorted(
            errors_by_timestamp.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Log top_k highest error examples
        logging.info(f"Top {top_k} anomalies by reconstruction error:")
        for i, (ts, error) in enumerate(sorted_timestamps[:top_k]):
            logging.info(f"{i+1}. Timestamp: {ts}, Error: {error:.6f}")
        
        # Save top anomalies to CSV
        anomalies_csv_path = os.path.join(params['output_dir'], 'top_anomalies.csv')
        with open(anomalies_csv_path, 'w') as f:
            f.write("Rank,Timestamp,ReconstructionError\n")
            for i, (ts, error) in enumerate(sorted_timestamps[:top_k*10]):  # Save more for analysis
                f.write(f"{i+1},{ts},{error:.6f}\n")
        
        logging.info(f"Top anomalies saved to {anomalies_csv_path}")
    except Exception as e:
        logging.warning(f"Could not diagnose anomalies: {e}")