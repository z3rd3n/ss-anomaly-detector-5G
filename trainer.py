import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score

from dataset import BinaryAnomalyDataset, create_binary_dataloader, binary_collate_fn
from model import TimeSeriesAnomalyDetector
from datetime import datetime

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
    return log_file

def log_model_size(model):
    """
    Log model size and number of parameters.
    """
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
        
    # Estimate model size in memory
    bytes_per_param = 4  # assuming float32
    model_size_bytes = total_params * bytes_per_param
    model_size_mb = model_size_bytes / (1024 * 1024)
    logging.info(f"Approximate model size in memory: {model_size_mb:.2f} MB")

# Constants
EPS = 1e-8

class BinaryMetricsTracker:
    """
    Track metrics for binary anomaly detection
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.true_binary = []
        self.pred_binary = []
        self.pred_probs = []
        self.true_original = []  # Original multi-class labels for reporting
        
    def update(self, true_binary, pred_binary, pred_probs, true_original=None):
        """
        Update metrics with batch results
        
        Args:
            true_binary: Tensor of true binary labels (0: normal, 1: anomaly)
            pred_binary: Tensor of predicted binary labels
            pred_probs: Tensor of anomaly probabilities
            true_original: Tensor of original multi-class labels
        """
        # Flatten if needed
        if true_binary.ndim > 1:
            true_binary = true_binary.view(-1)
        if pred_binary.ndim > 1:
            pred_binary = pred_binary.view(-1)
        if pred_probs.ndim > 1:
            pred_probs = pred_probs.view(-1)
            
        # Save for later computation
        self.true_binary.append(true_binary.cpu())
        self.pred_binary.append(pred_binary.cpu())
        self.pred_probs.append(pred_probs.cpu())
        
        if true_original is not None:
            if true_original.ndim > 1:
                true_original = true_original.view(-1)
            self.true_original.append(true_original.cpu())
    
    def compute_metrics(self):
        """
        Compute all binary classification metrics
        """
        # Concatenate all batches
        all_true_binary = torch.cat(self.true_binary, dim=0).numpy()
        all_pred_binary = torch.cat(self.pred_binary, dim=0).numpy()
        all_pred_probs = torch.cat(self.pred_probs, dim=0).numpy()
        
        # Calculate overall metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_binary, all_pred_binary, average='binary')
        
        # Handle edge case for AUC calculation
        if len(np.unique(all_true_binary)) == 1:
            auc = 0.0  # Only one class present
        else:
            auc = roc_auc_score(all_true_binary, all_pred_probs)
            
        # Calculate precision-recall AUC
        average_precision = average_precision_score(all_true_binary, all_pred_probs)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(all_true_binary, all_pred_binary).ravel()
        
        # Derived metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = recall  # Same as recall
        
        # Return all metrics
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'auc': auc,
            'average_precision': average_precision,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
    
    def compute_anomaly_type_breakdown(self, inverse_mapping):
        """
        Compute breakdown of detection performance by original anomaly type
        
        Args:
            inverse_mapping: Dictionary mapping class indices to anomaly names
        """
        if not self.true_original:
            return {}
            
        all_true_binary = torch.cat(self.true_binary, dim=0).numpy()
        all_pred_binary = torch.cat(self.pred_binary, dim=0).numpy()
        all_true_original = torch.cat(self.true_original, dim=0).numpy()
        
        # For each original class, compute how well we detected it
        results = {}
        
        for class_idx, class_name in inverse_mapping.items():
            # Find samples of this class
            mask = (all_true_original == class_idx)
            if not np.any(mask):
                continue
                
            # Count how many were correctly identified as anomalies (except for normal class)
            if class_idx == 0:  # Normal class
                correct = np.sum((all_true_original == 0) & (all_pred_binary == 0))
                total = np.sum(all_true_original == 0)
                detection_rate = correct / total if total > 0 else 0.0
                results[class_name] = {
                    'class_idx': class_idx,
                    'total_samples': total,
                    'correctly_classified': correct,
                    'detection_rate': detection_rate
                }
            else:  # Anomaly classes
                correct = np.sum((all_true_original == class_idx) & (all_pred_binary == 1))
                total = np.sum(all_true_original == class_idx)
                detection_rate = correct / total if total > 0 else 0.0
                results[class_name] = {
                    'class_idx': class_idx,
                    'total_samples': total,
                    'correctly_classified': correct,
                    'detection_rate': detection_rate
                }
                
        return results
        
    def log_metrics(self, prefix=""):
        """
        Log binary classification metrics
        """
        metrics = self.compute_metrics()
        
        logging.info(f"{prefix} Binary Classification Metrics:")
        logging.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"  Precision: {metrics['precision']:.4f}")
        logging.info(f"  Recall (Sensitivity): {metrics['recall']:.4f}")
        logging.info(f"  Specificity: {metrics['specificity']:.4f}")
        logging.info(f"  F1 Score: {metrics['f1']:.4f}")
        logging.info(f"  AUC: {metrics['auc']:.4f}")
        logging.info(f"  Average Precision: {metrics['average_precision']:.4f}")
        logging.info(f"  Confusion Matrix: TP={metrics['true_positives']}, "
                     f"FP={metrics['false_positives']}, "
                     f"TN={metrics['true_negatives']}, "
                     f"FN={metrics['false_negatives']}")
        
        return metrics
        
    def log_anomaly_breakdown(self, inverse_mapping, prefix=""):
        """
        Log detection performance by anomaly type
        """
        breakdown = self.compute_anomaly_type_breakdown(inverse_mapping)
        
        logging.info(f"{prefix} Anomaly Type Detection Breakdown:")
        for class_name, stats in breakdown.items():
            logging.info(f"  {class_name}: {stats['detection_rate']:.4f} "
                         f"({stats['correctly_classified']}/{stats['total_samples']})")
            
        return breakdown


class BinaryAnomalyTrainer:
    """
    Trainer for binary anomaly detection
    """
    def __init__(self, model, params, device):
        self.model = model
        self.params = params
        self.device = device
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=params.get('lr', 1e-3),
            weight_decay=params.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=2,
            verbose=True
        )
        
        # Load feature statistics from file
        stats_file = params.get('features_stats_json', 'feature_stats.json')
        if os.path.exists(stats_file):
            logging.info(f"Loading feature statistics from {stats_file}")
            with open(stats_file, 'r') as f:
                self.feature_stats = json.load(f)
        else:
            # Set default stats to avoid errors
            self.feature_stats = {
                'means': [0.0] * len(params.get('numerical_features', ['SFN', 'Slot', 'MCS', 'ReTx'])),
                'stds': [1.0] * len(params.get('numerical_features', ['SFN', 'Slot', 'MCS', 'ReTx']))
            }
            logging.warning(f"Feature statistics file {stats_file} not found, using default values")
        
        # Create datasets and dataloaders
        self._setup_datasets()
        self._setup_dataloaders()
        
        # Best metrics tracking
        self.best_val_metrics = {
            'f1': 0.0,
            'auc': 0.0,
            'average_precision': 0.0
        }
        
    def _setup_datasets(self):
        """
        Create and prepare datasets with optimized loading
        """
        # Create training dataset with pre-loaded statistics
        self.train_dataset = BinaryAnomalyDataset(
            parquet_path=self.params['parquet_path'],
            feature_columns=self.params['feature_columns'],
            seq_len=self.params['seq_len'],
            stride=self.params.get('stride', None),
            ratio=self.params.get('train_ratio', 1.0),
            seed=self.params['seed'],
            normalization_stats=self.feature_stats,  # Use pre-loaded stats
            pre_normalize=self.params.get('pre_normalize', True),
            use_state_cache=self.params.get('use_state_cache', True),
            overlap_ratio=self.params.get('overlap_ratio', 0.5),
            keep_original_labels=True
        )
        
        # Create validation dataset if path provided
        if 'validation_parquet_path' in self.params and self.params['validation_parquet_path']:
            self.val_dataset = BinaryAnomalyDataset(
                parquet_path=self.params['validation_parquet_path'],
                feature_columns=self.params['feature_columns'],
                seq_len=self.params['seq_len'],
                stride=self.params.get('stride', None),
                ratio=self.params.get('val_ratio', 1.0),
                seed=self.params['seed'],
                normalization_stats=self.feature_stats,
                pre_normalize=self.params.get('pre_normalize', True),
                use_state_cache=self.params.get('use_state_cache', True),
                overlap_ratio=self.params.get('overlap_ratio', 0.5),
                keep_original_labels=True
            )
        else:
            self.val_dataset = None
        
    def _setup_dataloaders(self):
        """
        Create dataloaders for training and validation
        """
        # Create balanced training dataloader
        self.train_loader = create_binary_dataloader(
            self.train_dataset,
            batch_size=self.params['batch_size'],
            shuffle=True,
            balance=self.params.get('use_balanced_sampling', True),
            num_workers=self.params.get('num_workers', 4)
        )
        
        # Create validation dataloader if dataset exists
        if self.val_dataset:
            self.val_loader = create_binary_dataloader(
                self.val_dataset,
                batch_size=self.params['batch_size'],
                shuffle=False,
                balance=False,  # No balancing for validation
                num_workers=self.params.get('num_workers', 4)
            )
        else:
            self.val_loader = None
    
    def train(self):
        """
        Train the binary anomaly detection model
        """
        num_epochs = self.params.get('num_epochs', 10)
        output_dir = self.params['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Training loop
        train_metrics_history = []
        val_metrics_history = []
        
        for epoch in range(1, num_epochs + 1):
            # Train one epoch
            train_metrics = self._train_epoch(epoch, num_epochs)
            train_metrics_history.append(train_metrics)
            
            # Validate
            if self.val_loader:
                val_metrics = self._validate()
                val_metrics_history.append(val_metrics)
                
                # Update learning rate
                self.scheduler.step(val_metrics['f1'])
                
                # Save best model
                if val_metrics['f1'] > self.best_val_metrics['f1']:
                    self.best_val_metrics = val_metrics
                    self._save_checkpoint(epoch, is_best=True)
                    logging.info(f"Saved best model at epoch {epoch} with F1: {val_metrics['f1']:.4f}")
            
            # Save checkpoint periodically
            if epoch % self.params.get('checkpoint_interval', 5) == 0:
                self._save_checkpoint(epoch)
            
        # Return best validation metrics
        return self.best_val_metrics
    
    def _train_epoch(self, epoch, num_epochs):
        """
        Train for one epoch
        """
        self.model.train()
        metrics_tracker = BinaryMetricsTracker()
        total_loss = 0.0
        total_batches = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")
        
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            # Get batch data
            numerical_features = batch['numerical_features'].to(self.device)
            categorical_features = {
                k: v.to(self.device) for k, v in batch['categorical_features'].items()
            }
            binary_labels = batch['binary_labels'].to(self.device)
            
            # Optional: get original labels for reporting
            original_labels = batch.get('original_labels')
            if original_labels is not None:
                original_labels = original_labels.to(self.device)
            
            # Forward pass
            logits, _ = self.model(numerical_features, categorical_features)
            
            # Calculate loss
            loss = self.model.compute_loss(logits, binary_labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.params.get('grad_clip', 1.0)
            )
            
            # Update weights
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > 0.5).float()
                
                # Update metrics
                metrics_tracker.update(
                    binary_labels, 
                    predictions, 
                    probabilities,
                    original_labels
                )
            
            # Update running statistics
            total_loss += loss.item()
            total_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        # Calculate epoch metrics
        metrics = metrics_tracker.log_metrics(prefix=f"Epoch {epoch} Train")
        
        # Log anomaly type breakdown
        if hasattr(self.train_dataset, 'inverse_anomaly_mapping'):
            metrics_tracker.log_anomaly_breakdown(
                self.train_dataset.inverse_anomaly_mapping,
                prefix=f"Epoch {epoch} Train"
            )
        
        # Add loss to metrics
        metrics['loss'] = total_loss / total_batches
        
        return metrics
    
    def _validate(self):
        """
        Validate the model
        """
        self.model.eval()
        metrics_tracker = BinaryMetricsTracker()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", unit="batch"):
                # Get batch data
                numerical_features = batch['numerical_features'].to(self.device)
                categorical_features = {
                    k: v.to(self.device) for k, v in batch['categorical_features'].items()
                }
                binary_labels = batch['binary_labels'].to(self.device)
                
                # Optional: get original labels for reporting
                original_labels = batch.get('original_labels')
                if original_labels is not None:
                    original_labels = original_labels.to(self.device)
                
                # Forward pass
                logits, _ = self.model(numerical_features, categorical_features)
                
                # Calculate metrics
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > 0.5).float()
                
                # Update metrics
                metrics_tracker.update(
                    binary_labels, 
                    predictions, 
                    probabilities,
                    original_labels
                )
        
        # Calculate and log metrics
        metrics = metrics_tracker.log_metrics(prefix="Validation")
        
        # Log anomaly type breakdown
        if hasattr(self.val_dataset, 'inverse_anomaly_mapping'):
            breakdown = metrics_tracker.log_anomaly_breakdown(
                self.val_dataset.inverse_anomaly_mapping,
                prefix="Validation"
            )
            
            # Store breakdown in metrics
            metrics['anomaly_breakdown'] = breakdown
        
        return metrics
    
    def _save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint
        """
        output_dir = self.params['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'feature_stats': self.feature_stats,
            'params': self.params,
            'best_metrics': self.best_val_metrics
        }
        
        # Save checkpoint
        if is_best:
            checkpoint_path = os.path.join(output_dir, 'best_model.pt')
        else:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
            
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")
    
    def predict(self, dataset, batch_size=32, save_path=None, anomaly_threshold=0.5):
        """
        Make predictions on a dataset and optionally save results
        
        Returns:
            DataFrame with predictions and their original anomaly types
        """
        self.model.eval()
        dataloader = create_binary_dataloader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            balance=False
        )
        
        # Store results
        all_timestamps = []
        all_binary_preds = []
        all_binary_probs = []
        all_original_labels = []
        all_harq_ids = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
                # Get batch data
                numerical_features = batch['numerical_features'].to(self.device)
                categorical_features = {
                    k: v.to(self.device) for k, v in batch['categorical_features'].items()
                }
                
                # Original labels and timestamps for reporting
                if 'original_labels' in batch:
                    original_labels = batch['original_labels'].cpu().numpy()
                    all_original_labels.append(original_labels)
                    
                all_timestamps.extend(batch['timestamps'])
                if 'harq_ids' in batch:
                    all_harq_ids.append(batch['harq_ids'].cpu().numpy())
                
                # Forward pass
                logits, _ = self.model(numerical_features, categorical_features)
                probabilities = torch.sigmoid(logits).cpu().numpy()
                predictions = (probabilities > anomaly_threshold).astype(float)
                
                # Store predictions
                all_binary_preds.append(predictions)
                all_binary_probs.append(probabilities)
        
        # Concatenate results
        binary_preds = np.concatenate(all_binary_preds).flatten()
        binary_probs = np.concatenate(all_binary_probs).flatten()
        
        # Prepare timestamps (flatten nested lists)
        flat_timestamps = []
        for batch_timestamps in all_timestamps:
            flat_timestamps.extend(batch_timestamps)
            
        # Create result DataFrame
        results = {
            'timestamp': flat_timestamps,
            'anomaly_probability': binary_probs,
            'is_anomaly': binary_preds
        }
        
        # Add original labels if available
        if all_original_labels:
            original_labels = np.concatenate(all_original_labels).flatten()
            results['original_label'] = original_labels
            
            # Map original labels to names
            if hasattr(dataset, 'inverse_anomaly_mapping'):
                results['anomaly_type'] = [
                    dataset.inverse_anomaly_mapping.get(label, "unknown")
                    for label in original_labels
                ]
                
        # Add HARQ IDs if available
        if all_harq_ids:
            harq_ids = np.concatenate(all_harq_ids).flatten()
            results['harq_id'] = harq_ids
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Save to file if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            results_df.to_csv(save_path, index=False)
            logging.info(f"Saved predictions to {save_path}")
            
            # Also save unidentified anomalies
            if 'anomaly_type' in results_df.columns:
                # Find anomalies detected by model but not in predefined categories
                unidentified = results_df[
                    (results_df['is_anomaly'] == 1) & 
                    (results_df['original_label'] == 0)
                ]
                
                if len(unidentified) > 0:
                    unidentified_path = save_path.replace('.csv', '_unidentified.csv')
                    unidentified.to_csv(unidentified_path, index=False)
                    logging.info(f"Saved {len(unidentified)} unidentified anomalies to {unidentified_path}")
        
        return results_df
    
    def evaluate_model(self, dataset=None, save_dir=None, anomaly_threshold=0.5):
        """
        Comprehensive evaluation of the model
        """
        if dataset is None:
            if self.val_dataset:
                dataset = self.val_dataset
            else:
                dataset = self.train_dataset
                
        if save_dir is None:
            save_dir = self.params['output_dir']
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Make predictions
        predictions_path = os.path.join(save_dir, 'predictions.csv')
        predictions = self.predict(
            dataset, 
            batch_size=self.params['batch_size'],
            save_path=predictions_path,
            anomaly_threshold=anomaly_threshold
        )
        
        return predictions


def train_binary_model(params):
    """
    Train a binary anomaly detection model with optimized workflow
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Get categorical dimensions
    categorical_dims = {}
    categorical_features = params.get('categorical_features', ['HARQ', 'CRC', 'NDI'])
    for feature in categorical_features:
        if feature == 'HARQ':
            categorical_dims[feature] = 15  # 16 possible HARQ IDs (0-15)
        elif feature == 'CRC':
            categorical_dims[feature] = 1   # Binary (0-1)
        elif feature == 'NDI':
            categorical_dims[feature] = 1   # Binary (0-1)
    
    # Initialize model
    model = TimeSeriesAnomalyDetector(
        numerical_features=params.get('numerical_features', ['SFN', 'Slot', 'MCS', 'ReTx']),
        categorical_dims=categorical_dims,
        embedding_dim=params.get('embedding_dim', 8),
        hidden_dim=params.get('hidden_dim', 128),
        latent_dim=params.get('latent_dim', 64),
        num_layers=params.get('num_layers', 2),
        dropout=params.get('dropout', 0.3),
        alpha=params.get('focal_alpha', 0.75),
        gamma=params.get('focal_gamma', 2.0)
    )
    model.to(device)
    
    log_model_size(model)
    
    # Create trainer
    trainer = BinaryAnomalyTrainer(model, params, device)
    
    # Train model
    best_metrics = trainer.train()
    logging.info(f"Training complete. Best validation metrics: {best_metrics}")
    
    # Evaluate model
    if params.get('evaluate_after_training', True):
        logging.info("Evaluating model on validation set...")
        evaluation_results = trainer.evaluate_model()
        logging.info("Evaluation complete.")
    
    return model, trainer, best_metrics


def load_and_evaluate_model(model_path, params, device=None):
    """
    Load a trained model and evaluate it
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Update params with those from checkpoint
    for key, value in checkpoint.get('params', {}).items():
        if key not in params:
            params[key] = value
    
    # Get feature statistics from checkpoint
    feature_stats = checkpoint.get('feature_stats')
    if feature_stats is None:
        stats_file = params.get('features_stats_json', 'feature_stats.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                feature_stats = json.load(f)
        else:
            numerical_features = params.get('numerical_features', ['SFN', 'Slot', 'MCS', 'ReTx'])
            feature_stats = {
                'means': [0.0] * len(numerical_features),
                'stds': [1.0] * len(numerical_features)
            }
    
    # Get categorical dimensions
    categorical_dims = {}
    categorical_features = params.get('categorical_features', ['HARQ', 'CRC', 'NDI'])
    for feature in categorical_features:
        if feature == 'HARQ':
            categorical_dims[feature] = 15  # 16 possible HARQ IDs (0-15)
        elif feature == 'CRC':
            categorical_dims[feature] = 1   # Binary (0-1)
        elif feature == 'NDI':
            categorical_dims[feature] = 1   # Binary (0-1)
    
    # Initialize model
    model = TimeSeriesAnomalyDetector(
        numerical_features=params.get('numerical_features', ['SFN', 'Slot', 'MCS', 'ReTx']),
        categorical_dims=categorical_dims,
        embedding_dim=params.get('embedding_dim', 8),
        hidden_dim=params.get('hidden_dim', 128),
        latent_dim=params.get('latent_dim', 64),
        num_layers=params.get('num_layers', 2),
        dropout=params.get('dropout', 0.3),
        alpha=params.get('focal_alpha', 0.75),
        gamma=params.get('focal_gamma', 2.0)
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Create trainer
    trainer = BinaryAnomalyTrainer(model, params, device)
    
    # Evaluate model
    logging.info("Evaluating model...")
    evaluation_results = trainer.evaluate_model()
    logging.info("Evaluation complete.")
    
    return model, trainer, evaluation_results


def main():
    """Main execution function"""
    
    # Get default parameters and update with arguments
    params = {
        'parquet_path': 'unscaled_pdsch_val.parquet',
        'validation_parquet_path': 'unscaled_pdsch_val_min.parquet',
        'output_dir': 'output',
        'feature_columns': ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"],
        'numerical_features': ['SFN', 'Slot', 'MCS', 'ReTx'],
        'categorical_features': ['HARQ', 'CRC', 'NDI'],
        'seq_len': 100,
        'stride': 50,
        'overlap_ratio': 0.5,
        'train_ratio': 0.2,
        'val_ratio': 1.0,
        'seed': 42,
        'batch_size': 64,
        'num_epochs': 1,
        'lr': 5e-4,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'embedding_dim': 8,
        'hidden_dim': 128,
        'latent_dim': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'focal_alpha': 0.75,
        'focal_gamma': 2.0,
        'use_balanced_sampling': True,
        'use_state_cache': True,
        'pre_normalize': True,
        'num_workers': 4,
        'checkpoint_interval': 5,
        'features_stats_json': 'features_stats.json',
        'anomaly_threshold': 0.5,
        'max_samples': None
    }
    
    # Set up logging
    start_logging(params)
    
    # Set random seeds for reproducibility
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    
    # Train a new model
    logging.info("Training new model...")
    _, _, best_metrics = train_binary_model(params)
    logging.info(f"Training complete. Best metrics: {best_metrics}")
    
    logging.info("Process complete!")

if __name__ == "__main__":
    main()