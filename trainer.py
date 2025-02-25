import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import logging
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from dataset import (
    SequenceStateCacheDataset, 
    BalancedSequenceSampler, 
    NeighborPreservingSampler,
    custom_collate_fn
)
from utils import (
    save_checkpoint, 
    start_logging, 
    compute_features_statistics, 
    log_model_size, 
    load_checkpoint,
    validate_csv, 
    EPS
)
from model import TwoPhaseModel, WeightedFocalLoss

class TwoPhaseTrainer:
    """
    Trainer for two-phase anomaly detection model
    """
    def __init__(self, model, params, device):
        self.model = model
        self.params = params
        self.device = device
        
        # Create optimizers
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
        
        # Create dataset object
        self.train_dataset = self._create_dataset(
            params['parquet_path'], 
            params.get('train_ratio', 1.0),
            normalization_stats=None,  # Will be computed later
            use_state_cache=params.get('use_state_cache', True)
        )
        
        # Compute feature statistics for normalization
        self.means, self.variances = compute_features_statistics(
            self.train_dataset, 
            params, 
            len(params['feature_columns'])
        )
        
        # Reinitialize dataset with normalization stats
        self.train_dataset.normalization_stats = {'means': self.means, 'variances': self.variances}
        
        # Create validation dataset
        if 'validation_parquet_path' in params:
            self.val_dataset = self._create_dataset(
                params['validation_parquet_path'],
                params.get('val_ratio', 1.0),
                normalization_stats={'means': self.means, 'variances': self.variances},
                use_state_cache=params.get('use_state_cache', True)
            )
        else:
            self.val_dataset = None
            
        # Set up samplers based on params
        self._setup_samplers()
        
        # Set up dataloaders
        self._setup_dataloaders()
        
        # Training metrics
        self.best_val_metrics = {
            'balanced_acc': 0.0,
            'f1_score': 0.0,
            'anomaly_recall': 0.0
        }
        
        # Loss weights
        self.vae_loss_weight = params.get('vae_loss_weight', 0.2)
        self.supervised_loss_weight = params.get('supervised_loss_weight', 1.0)
        
        # Dynamic weighting parameters
        self.use_dynamic_weights = params.get('use_dynamic_weights', True)
        
    def _create_dataset(self, parquet_path, ratio, normalization_stats=None, use_state_cache=True):
        """
        Create dataset from parquet file
        """
        return SequenceStateCacheDataset(
            parquet_path=parquet_path,
            feature_columns=self.params['feature_columns'],
            seq_len=self.params['seq_len'],
            stride=self.params.get('stride', None),
            ratio=ratio,
            seed=self.params['seed'],
            skip_anomalies=False,
            normalization_stats=normalization_stats,
            use_state_cache=use_state_cache,
            overlap_ratio=self.params.get('overlap_ratio', 0.5)
        )
    
    def _setup_samplers(self):
        """
        Create train and validation samplers based on parameters
        """
        # Determine if we should use balanced sampling
        use_balanced_sampling = self.params.get('use_balanced_sampling', True)
        
        if use_balanced_sampling:
            self.train_sampler = BalancedSequenceSampler(
                self.train_dataset,
                oversample_ratio=self.params.get('oversample_ratio', 0.8),
                undersample_ratio=self.params.get('undersample_ratio', 0.3),
                seed=self.params['seed']
            )
        else:
            # Use neighbor-preserving sampler if balanced sampling is disabled
            self.train_sampler = NeighborPreservingSampler(
                self.train_dataset,
                window_size=self.params.get('neighbor_window_size', 3),
                seed=self.params['seed']
            )
            
        # For validation, we use sequential sampling
        self.val_sampler = None
        
    def _setup_dataloaders(self):
        """
        Create train and validation dataloaders
        """
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.params['batch_size'],
            sampler=self.train_sampler,
            collate_fn=custom_collate_fn,
            num_workers=self.params.get('num_workers', 4),
            pin_memory=True
        )
        
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.params['batch_size'],
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=self.params.get('num_workers', 4),
                pin_memory=True
            )
        else:
            self.val_loader = None
    
    def train(self):
        """
        Train the model
        """
        num_epochs = self.params.get('num_epochs', 10)
        output_dir = self.params['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Log model size
        log_model_size(self.model, self.device)
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            # Train one epoch
            train_metrics = self._train_epoch(epoch, num_epochs)
            
            # Validate
            if self.val_loader:
                val_metrics = self._validate()
                
                # Update learning rate scheduler
                self.scheduler.step(val_metrics['balanced_acc'])
                
                # Save best model
                if val_metrics['balanced_acc'] > self.best_val_metrics['balanced_acc']:
                    self.best_val_metrics = val_metrics
                    checkpoint_state = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_val_metrics': self.best_val_metrics,
                        'params': self.params,
                        'features_stats': {
                            'means': self.means.tolist(),
                            'variances': self.variances.tolist()
                        }
                    }
                    filename = os.path.join(output_dir, 'best_model.pt')
                    save_checkpoint(checkpoint_state, filename)
                    logging.info(f"Saved best model at epoch {epoch} with balanced accuracy: {val_metrics['balanced_acc']:.4f}")
            
            # Save checkpoint every few epochs
            if epoch % self.params.get('checkpoint_interval', 5) == 0:
                checkpoint_state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_metrics': self.best_val_metrics,
                    'params': self.params,
                    'features_stats': {
                        'means': self.means.tolist(),
                        'variances': self.variances.tolist()
                    }
                }
                filename = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
                save_checkpoint(checkpoint_state, filename)
                
        # Return best validation metrics
        return self.best_val_metrics
        
    def _train_epoch(self, epoch, num_epochs):
        """
        Train one epoch
        """
        self.model.train()
        
        total_loss = 0.0
        total_vae_loss = 0.0
        total_clf_loss = 0.0
        total_batches = 0
        
        all_true_labels = []
        all_pred_labels = []
        
        # Use class counts from the dataset for loss weighting
        if self.use_dynamic_weights:
            class_counts = self.train_dataset.counts
            class_weights = self.model.get_loss_weights(class_counts).to(self.device)
        else:
            class_weights = None
            
        # Create progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")
        
        # Train on batches
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            # Move batch to device
            features = batch['features'].to(self.device)    # shape: [B, T, num_features]
            true_labels = batch['labels'].to(self.device)   # shape: [B, T]
            
            # Forward pass
            embedded = self.model.feature_embedding(features)
            embedded = self.model.positional_encoding(embedded)
            
            # Phase 1: VAE
            reconstruction, mu, logvar, z = self.model.vae(embedded)
            
            # Phase 2: Classification
            class_logits = self.model.classifier(embedded, z)
            
            # Compute losses
            vae_loss = self.model.compute_vae_loss(features, reconstruction, mu, logvar)
            
            # Use weighted focal loss for classification
            if class_weights is not None:
                focal_loss = WeightedFocalLoss(
                    weights=class_weights,
                    alpha=self.params.get('focal_alpha', 0.5),
                    gamma=self.params.get('focal_gamma', 2.0)
                )
                clf_loss = focal_loss(class_logits, true_labels)
            else:
                clf_loss = self.model.compute_classification_loss(class_logits, true_labels)
            
            # Combined loss
            loss = self.supervised_loss_weight * clf_loss + self.vae_loss_weight * vae_loss
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.get('grad_clip', 1.0))
            
            # Update weights
            self.optimizer.step()
            
            # Collect metrics
            total_loss += loss.item()
            total_vae_loss += vae_loss.item()
            total_clf_loss += clf_loss.item()
            total_batches += 1
            
            # Compute classification metrics
            pred_labels = torch.argmax(class_logits, dim=-1)
            all_true_labels.append(true_labels.cpu())
            all_pred_labels.append(pred_labels.cpu())
            
            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "VAE": f"{vae_loss.item():.4f}",
                "CLF": f"{clf_loss.item():.4f}"
            })
            
        # Compute aggregated metrics
        all_true_labels = torch.cat(all_true_labels, dim=0).view(-1).numpy()
        all_pred_labels = torch.cat(all_pred_labels, dim=0).view(-1).numpy()
        
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_labels, all_pred_labels, average='weighted')
        
        # Calculate normal vs anomaly metrics
        normal_mask = (all_true_labels == 0)
        anomaly_mask = (all_true_labels != 0)
        
        normal_acc = (all_pred_labels[normal_mask] == all_true_labels[normal_mask]).mean() \
            if normal_mask.sum() > 0 else 0
        anomaly_acc = (all_pred_labels[anomaly_mask] == all_true_labels[anomaly_mask]).mean() \
            if anomaly_mask.sum() > 0 else 0
        
        balanced_acc = (normal_acc + anomaly_acc) / 2
        
        # Log metrics
        avg_loss = total_loss / total_batches
        avg_vae_loss = total_vae_loss / total_batches
        avg_clf_loss = total_clf_loss / total_batches
        
        logging.info(f"Epoch {epoch} Train - "
                     f"Loss: {avg_loss:.4f}, VAE: {avg_vae_loss:.4f}, CLF: {avg_clf_loss:.4f}, "
                     f"Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, "
                     f"Normal Acc: {normal_acc:.4f}, Anomaly Acc: {anomaly_acc:.4f}, "
                     f"Balanced Acc: {balanced_acc:.4f}")
        
        # Return metrics
        return {
            'loss': avg_loss,
            'vae_loss': avg_vae_loss,
            'clf_loss': avg_clf_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'normal_acc': normal_acc,
            'anomaly_acc': anomaly_acc,
            'balanced_acc': balanced_acc
        }
    
    def _validate(self):
        """
        Validate the model with robust error handling
        """
        self.model.eval()
        
        all_true_labels = []
        all_pred_labels = []
        all_class_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", unit="batch"):
                try:
                    # Move batch to device
                    features = batch['features'].to(self.device)
                    true_labels = batch['labels'].to(self.device)
                    
                    # Ensure true_labels are within valid range
                    true_labels = torch.clamp(true_labels, 0, 4)  # Assuming 5 classes (0-4)
                    
                    # Forward pass with try-except
                    try:
                        class_logits = self.model(features)
                        
                        # Store predictions and true labels
                        pred_labels = torch.argmax(class_logits, dim=-1)
                        class_probs = F.softmax(class_logits, dim=-1)
                        
                        # Safely move to CPU
                        all_true_labels.append(true_labels.cpu())
                        all_pred_labels.append(pred_labels.cpu())
                        all_class_probs.append(class_probs.cpu())
                    except Exception as e:
                        logging.error(f"Error in model forward pass: {e}")
                        continue
                        
                except Exception as e:
                    logging.error(f"Error processing batch: {e}")
                    continue
        
        if not all_true_labels:
            logging.error("No valid batches found during validation")
            return {
                'balanced_acc': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
        
        # Concatenate results
        try:
            all_true_labels = torch.cat(all_true_labels, dim=0).view(-1).numpy()
            all_pred_labels = torch.cat(all_pred_labels, dim=0).view(-1).numpy()
            all_class_probs = torch.cat(all_class_probs, dim=0).numpy()
        except Exception as e:
            logging.error(f"Error concatenating results: {e}")
            return {
                'balanced_acc': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
        
        # Calculate confusion matrix
        try:
            cm = confusion_matrix(all_true_labels, all_pred_labels)
        except Exception as e:
            logging.error(f"Error calculating confusion matrix: {e}")
            cm = np.zeros((5, 5))  # Default empty matrix
        
        # Calculate per-class metrics with error handling
        try:
            precision, recall, f1, support = precision_recall_fscore_support(
                all_true_labels, all_pred_labels, average=None, zero_division=0)
        except Exception as e:
            logging.error(f"Error calculating per-class metrics: {e}")
            precision = recall = f1 = support = np.zeros(5)
        
        # Calculate weighted metrics
        try:
            weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
                all_true_labels, all_pred_labels, average='weighted', zero_division=0)
        except Exception as e:
            logging.error(f"Error calculating weighted metrics: {e}")
            weighted_precision = weighted_recall = weighted_f1 = 0.0
        
        # Calculate normal vs anomaly metrics
        normal_mask = (all_true_labels == 0)
        anomaly_mask = (all_true_labels != 0)
        
        normal_total = normal_mask.sum()
        anomaly_total = anomaly_mask.sum()
        
        if normal_total > 0:
            normal_acc = (all_pred_labels[normal_mask] == all_true_labels[normal_mask]).mean()
        else:
            normal_acc = 0.0
            
        if anomaly_total > 0:
            anomaly_acc = (all_pred_labels[anomaly_mask] == all_true_labels[anomaly_mask]).mean()
        else:
            anomaly_acc = 0.0
        
        # Balanced accuracy
        balanced_acc = (normal_acc + anomaly_acc) / 2
        
        # Log metrics
        logging.info(f"Validation - "
                    f"Weighted Precision: {weighted_precision:.4f}, "
                    f"Weighted Recall: {weighted_recall:.4f}, "
                    f"Weighted F1: {weighted_f1:.4f}")
        
        logging.info(f"Validation - "
                    f"Normal Acc: {normal_acc:.4f}, "
                    f"Anomaly Acc: {anomaly_acc:.4f}, "
                    f"Balanced Acc: {balanced_acc:.4f}")
        
        # Log per-class metrics
        class_names = ['normal'] + list(self.train_dataset.anomaly_mapping.keys())
        for i in range(len(precision)):
            if i < len(class_names):
                class_name = class_names[i]
            else:
                class_name = f"Class {i}"
            logging.info(f"Class {i} ({class_name}): "
                        f"Precision: {precision[i]:.4f}, "
                        f"Recall: {recall[i]:.4f}, "
                        f"F1: {f1[i]:.4f}, "
                        f"Support: {support[i]}")
        
        # Plot and save confusion matrix
        try:
            self._plot_confusion_matrix(cm, class_names)
        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {e}")
        
        # Return metrics
        return {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1_score': weighted_f1,
            'normal_acc': normal_acc,
            'anomaly_acc': anomaly_acc,
            'balanced_acc': balanced_acc,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1
        }
    
    def _plot_confusion_matrix(self, cm, class_names):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save figure
        output_dir = self.params['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

def train_model(model, params, device):
    """Train model wrapper function"""
    trainer = TwoPhaseTrainer(model, params, device)
    best_metrics = trainer.train()
    return model, best_metrics

def main_training_pipeline(params, model=None, train=True):
    """Main training pipeline"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if train:
        # Initialize model if not provided
        if model is None:
            feature_ranges = params.get('feature_ranges', [1023, 30, 15, 32, 1, 8, 1])
            model = TwoPhaseModel(
                feature_ranges=feature_ranges,
                embedding_dim=params.get('embedding_dim', 8),
                hidden_dim=params.get('hidden_dim', 64),
                latent_dim=params.get('latent_dim', 32),
                num_classes=params.get('num_classes', 5),
                num_layers=params.get('num_layers', 2),
                dropout=params.get('dropout', 0.2)
            )
        
        model.to(device)
        model, best_metrics = train_model(model, params, device)
        logging.info(f"Training complete. Best validation metrics: {best_metrics}")
    else:
        # Load the model from the checkpoint
        if model is None:
            feature_ranges = params.get('feature_ranges', [1023, 30, 15, 32, 1, 8, 1])
            model = TwoPhaseModel(
                feature_ranges=feature_ranges,
                embedding_dim=params.get('embedding_dim', 8),
                hidden_dim=params.get('hidden_dim', 64),
                latent_dim=params.get('latent_dim', 32),
                num_classes=params.get('num_classes', 5),
                num_layers=params.get('num_layers', 2),
                dropout=params.get('dropout', 0.2)
            )
        
        checkpoint_path = os.path.join(params['output_dir'], 'best_model.pt')
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 1e-3))
        _, best_metrics, loaded_params = load_checkpoint(checkpoint_path, model, optimizer)
        
        # Update params with loaded params
        for key, value in loaded_params.items():
            if key not in params:
                params[key] = value
                
        logging.info(f"Loaded model for evaluation. Best metrics: {best_metrics}")
        
        # Load normalization statistics
        if 'features_stats' in loaded_params:
            stats = loaded_params['features_stats']
        else:
            stats_file = params['features_stats_json']
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                
        means = torch.tensor(stats['means'], dtype=torch.float32)
        variances = torch.tensor(stats['variances'], dtype=torch.float32)
        
        # Validate the model
        val_metrics = validate_csv(model, params, means, variances, device)
        logging.info(f"Validation metrics: {val_metrics}")

    return model, params

if __name__ == '__main__':
    # Default hyperparameters and settings
    params = {
        'parquet_path': 'unscaled_pdsch_val.parquet',
        'validation_parquet_path': 'unscaled_pdsch_val_min.parquet', 
        'output_dir': './output',
        'feature_columns': ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"],
        'feature_ranges': [1023, 30, 15, 32, 1, 8, 1],
        'seq_len': 100,
        'stride': 50,
        'overlap_ratio': 0.5,
        'train_ratio': 1.0,
        'val_ratio': 1.0,
        'seed': 42,
        'batch_size': 64,
        'num_epochs': 20,
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
        'vae_loss_weight': 0.2,
        'supervised_loss_weight': 1.0,
        'use_balanced_sampling': True,
        'oversample_ratio': 0.8,
        'undersample_ratio': 0.3,
        'use_state_cache': True,
        'use_dynamic_weights': True,
        'num_workers': 4,
        'checkpoint_interval': 5,
        'features_stats_json': 'features_stats.json',
        'num_classes': 5,
        'train': True,
    }

    start_logging(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = TwoPhaseModel(
        feature_ranges=params['feature_ranges'],
        embedding_dim=params['embedding_dim'],
        hidden_dim=params['hidden_dim'],
        latent_dim=params['latent_dim'],
        num_classes=params['num_classes'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    )
    model.to(device)

    # Train or evaluate
    model, params = main_training_pipeline(params, model, params['train'])
    logging.info("Process complete.")