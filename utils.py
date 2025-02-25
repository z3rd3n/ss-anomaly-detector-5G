import os
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
from collections import defaultdict
import torch.nn.functional as F

# Constants
EPS = 1e-8

def start_logging(params=None):
    """
    Initialize logging.
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = params['output_dir'] if params and 'output_dir' in params else 'output'
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'log_training_{current_time}.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    if params:
        logging.info("Hyperparameters and settings:")
        for key, value in params.items():
            logging.info(f"{key}: {value}")

def save_checkpoint(state: dict, filename: str):
    """
    Save model checkpoint.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    logging.info(f"Checkpoint saved to {filename}")

def load_checkpoint(filename: str, model: nn.Module, optimizer: torch.optim.Optimizer):
    """
    Load model checkpoint.
    """
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        best_metrics = checkpoint.get('best_val_metrics', {'balanced_acc': 0.0})
        params = checkpoint.get('params', {})
        logging.info(f"Loaded checkpoint '{filename}' (epoch {epoch})")
        return epoch, best_metrics, params
    else:
        logging.warning(f"No checkpoint found at '{filename}'")
        return 0, {'balanced_acc': 0.0}, {}

def compute_features_statistics(dataset, params, num_features):
    """
    Compute mean and variance for each feature.
    """
    stats_file = params['features_stats_json']
    if os.path.exists(stats_file):
        logging.info(f"Loading features statistics from {stats_file}")
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        means = torch.tensor(stats['means'], dtype=torch.float32)
        variances = torch.tensor(stats['variances'], dtype=torch.float32)
        return means, variances
    
    logging.info("Computing features statistics from training data...")
    data_loader = DataLoader(
        dataset, 
        batch_size=params.get('batch_size', 128), 
        shuffle=False
    )
    
    total_sum = torch.zeros(num_features, dtype=torch.float64)
    total_sum_sq = torch.zeros(num_features, dtype=torch.float64)
    count = 0
    
    for batch in tqdm(data_loader, desc="Computing features statistics", unit="batch"):
        features = batch['features'].to(torch.float64)
        B, T, F = features.shape
        count += B * T
        total_sum += features.sum(dim=(0,1))
        total_sum_sq += (features**2).sum(dim=(0,1))
    
    means = (total_sum / count).to(torch.float32)
    variances = ((total_sum_sq / count) - (means.double()**2)).to(torch.float32)
    
    # Ensure positive variances
    variances = torch.clamp(variances, min=EPS)
    
    stats = {'means': means.tolist(), 'variances': variances.tolist()}
    
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, 'w') as f:
        json.dump(stats, f)
    
    logging.info(f"Features statistics saved to {stats_file}")
    return means, variances

def unnormalize_features(normalized_tensor, means, variances):
    """
    Unnormalize features using mean and variance.
    """
    std = torch.sqrt(variances + EPS)
    unnorm = normalized_tensor.float() * std + means
    return torch.clamp(torch.round(unnorm), min=0).to(torch.int64)

class MetricsTracker:
    """
    Track and compute metrics for anomaly detection.
    """
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.true_labels = []
        self.pred_labels = []
        self.pred_probs = []
        self.total_samples = 0
        self.correct = 0
        
        # Per-class metrics
        self.class_totals = torch.zeros(self.num_classes)
        self.class_correct = torch.zeros(self.num_classes)
        
    def update(self, true_labels, pred_labels, pred_probs=None):
        """
        Update metrics with batch results.
        
        Args:
            true_labels: Tensor of true class labels
            pred_labels: Tensor of predicted class labels
            pred_probs: Tensor of class probabilities (optional)
        """
        # Flatten if needed
        if true_labels.ndim > 1:
            true_labels = true_labels.view(-1)
        if pred_labels.ndim > 1:
            pred_labels = pred_labels.view(-1)
        if pred_probs is not None and pred_probs.ndim > 2:
            B, T, C = pred_probs.shape
            pred_probs = pred_probs.view(B*T, C)
        
        # Update counts
        self.true_labels.append(true_labels.cpu())
        self.pred_labels.append(pred_labels.cpu())
        if pred_probs is not None:
            self.pred_probs.append(pred_probs.cpu())
        
        # Update accuracy metrics
        correct = (true_labels == pred_labels).float()
        self.total_samples += true_labels.size(0)
        self.correct += correct.sum().item()
        
        # Update per-class metrics
        for c in range(self.num_classes):
            class_mask = (true_labels == c)
            if class_mask.sum() > 0:
                self.class_totals[c] += class_mask.sum().item()
                self.class_correct[c] += (correct * class_mask.float()).sum().item()
    
    def compute_metrics(self):
        """
        Compute all metrics.
        
        Returns:
            dict: Dictionary of metrics
        """
        # Concatenate all batches
        all_true = torch.cat(self.true_labels, dim=0).numpy()
        all_pred = torch.cat(self.pred_labels, dim=0).numpy()
        
        # Overall accuracy
        accuracy = self.correct / self.total_samples if self.total_samples > 0 else 0.0
        
        # Per-class accuracy
        class_accuracy = torch.zeros(self.num_classes)
        for c in range(self.num_classes):
            if self.class_totals[c] > 0:
                class_accuracy[c] = self.class_correct[c] / self.class_totals[c]
        
        # Normal vs anomaly accuracy
        normal_mask = (all_true == 0)
        anomaly_mask = (all_true != 0)
        
        normal_total = normal_mask.sum()
        anomaly_total = anomaly_mask.sum()
        
        normal_correct = ((all_true == all_pred) & normal_mask).sum()
        anomaly_correct = ((all_true == all_pred) & anomaly_mask).sum()
        
        normal_acc = normal_correct / normal_total if normal_total > 0 else 0.0
        anomaly_acc = anomaly_correct / anomaly_total if anomaly_total > 0 else 0.0
        
        # Balanced accuracy
        balanced_acc = (normal_acc + anomaly_acc) / 2.0
        
        # Precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            all_true, all_pred, average=None)
        
        weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
            all_true, all_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(all_true, all_pred)
        
        return {
            'accuracy': accuracy,
            'class_accuracy': class_accuracy.tolist(),
            'normal_acc': float(normal_acc),
            'anomaly_acc': float(anomaly_acc),
            'balanced_acc': float(balanced_acc),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist(),
            'weighted_precision': weighted_prec,
            'weighted_recall': weighted_rec,
            'weighted_f1': weighted_f1,
            'confusion_matrix': cm.tolist()
        }
    
    def log_metrics(self, prefix="", class_names=None):
        """
        Log metrics to the logging system.
        
        Args:
            prefix: String prefix for the log messages
            class_names: List of class names for prettier logging
        """
        metrics = self.compute_metrics()
        
        if not class_names:
            class_names = [f"Class {i}" for i in range(self.num_classes)]
            
        # Log overall metrics
        logging.info(f"{prefix} Overall Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"{prefix} Balanced Accuracy: {metrics['balanced_acc']:.4f}")
        logging.info(f"{prefix} Normal Accuracy: {metrics['normal_acc']:.4f}")
        logging.info(f"{prefix} Anomaly Accuracy: {metrics['anomaly_acc']:.4f}")
        logging.info(f"{prefix} Weighted Precision: {metrics['weighted_precision']:.4f}")
        logging.info(f"{prefix} Weighted Recall: {metrics['weighted_recall']:.4f}")
        logging.info(f"{prefix} Weighted F1: {metrics['weighted_f1']:.4f}")
        
        # Log per-class metrics
        logging.info(f"{prefix} Per-class metrics:")
        for i, (name, acc, prec, rec, f1, sup) in enumerate(zip(
                class_names, 
                metrics['class_accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1'],
                metrics['support'])):
            logging.info(f"  {name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, "
                         f"F1={f1:.4f}, Support={sup}")
            
        return metrics

def validate_csv(model, params, means, variances, device):
    """
    Validate model on validation set.
    """
    logging.info("Starting validation procedure...")
    
    from dataset import SequenceStateCacheDataset, custom_collate_fn
    
    val_dataset = SequenceStateCacheDataset(
        parquet_path=params['validation_parquet_path'],
        feature_columns=params['feature_columns'],
        seq_len=params['seq_len'],
        stride=params.get('stride', None),
        ratio=params.get('val_ratio', 1.0),
        seed=params['seed'],
        skip_anomalies=False,
        normalization_stats={'means': means, 'variances': variances},
        use_state_cache=params.get('use_state_cache', True),
        overlap_ratio=params.get('overlap_ratio', 0.5)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=params['batch_size'], 
        shuffle=False, 
        collate_fn=custom_collate_fn
    )

    model.eval()
    tracker = MetricsTracker(num_classes=params.get('num_classes', 5))
    
    # Mapping for prettier logging
    anomaly_mapping = {v: k for k, v in val_dataset.anomaly_mapping.items()}
    anomaly_mapping[0] = "normal"
    class_names = [anomaly_mapping.get(i, f"Class {i}") for i in range(params.get('num_classes', 5))]
    
    # Evaluate sequences
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", unit="batch"):
            features = batch['features'].to(device)
            true_labels = batch['labels'].to(device)
            
            # Forward pass
            class_logits = model(features)
            
            # Get predictions
            pred_labels = torch.argmax(class_logits, dim=-1)
            class_probs = F.softmax(class_logits, dim=-1)
            
            # Update metrics
            tracker.update(true_labels, pred_labels, class_probs)
    
    # Calculate and log metrics
    metrics = tracker.log_metrics(prefix="Validation", class_names=class_names)
    
    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, class_names, params['output_dir'], "validation_confusion_matrix.png")
    
    return metrics['balanced_acc']

def plot_confusion_matrix(cm, class_names, output_dir, filename):
    """
    Plot and save confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def log_model_size(model: torch.nn.Module, device: torch.device = None) -> None:
    """
    Log model size and parameter count.
    """
    if device is not None:
        model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

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

    bytes_per_param = 4
    total_bytes = total_params * bytes_per_param
    size_mb = total_bytes / (1024**2)
    logging.info(f"Approximate model size: {size_mb:.2f} MB (assuming fp32)")

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    def __init__(self, patience=5, delta=0.001, mode='max', verbose=False):
        """
        Args:
            patience: How many epochs to wait before stopping after best
            delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' depending on whether lower or higher values are better
            verbose: Whether to print info about early stopping
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, epoch, score):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
            
        if self.mode == 'min':
            improvement = self.best_score - score > self.delta
        else:
            improvement = score - self.best_score > self.delta
            
        if improvement:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            if self.verbose:
                logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

def plot_learning_curves(train_metrics, val_metrics, output_dir, filename_prefix="learning_curves"):
    """
    Plot and save learning curves.
    
    Args:
        train_metrics: List of dictionaries with training metrics
        val_metrics: List of dictionaries with validation metrics
        output_dir: Directory to save plots
        filename_prefix: Prefix for filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract epochs
    epochs = range(1, len(train_metrics) + 1)
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [m['loss'] for m in train_metrics], 'b-', label='Training Loss')
    if 'loss' in val_metrics[0]:
        plt.plot(epochs, [m['loss'] for m in val_metrics], 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_loss.png"))
    plt.close()
    
    # Plot accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [m['balanced_acc'] for m in train_metrics], 'b-', label='Training Balanced Acc')
    plt.plot(epochs, [m['balanced_acc'] for m in val_metrics], 'r-', label='Validation Balanced Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Balanced Accuracy')
    plt.legend()
    plt.title('Training and Validation Balanced Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_balanced_acc.png"))
    plt.close()
    
    # Plot normal vs anomaly accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [m['normal_acc'] for m in train_metrics], 'b-', label='Train Normal Acc')
    plt.plot(epochs, [m['anomaly_acc'] for m in train_metrics], 'g-', label='Train Anomaly Acc')
    plt.plot(epochs, [m['normal_acc'] for m in val_metrics], 'r-', label='Val Normal Acc')
    plt.plot(epochs, [m['anomaly_acc'] for m in val_metrics], 'm-', label='Val Anomaly Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Normal vs Anomaly Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_normal_anomaly_acc.png"))
    plt.close()
    
    # Plot F1 scores
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [m['f1_score'] for m in train_metrics], 'b-', label='Training F1')
    plt.plot(epochs, [m['f1_score'] for m in val_metrics], 'r-', label='Validation F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Training and Validation F1 Score')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_f1.png"))
    plt.close()