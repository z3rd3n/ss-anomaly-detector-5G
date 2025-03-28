import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from collections import Counter, defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx
from dataset import create_data_loaders
from trainer import BiGRUAnomalyDetector, setup_logger

# Set up paths - updated for your specific file
MODEL_PATH = "best_model.pth"
CONFIG_PATH = MODEL_PATH  # Use the same file for both
TEST_DATA_PATH = "unscaled_pdsch_val_min.parquet"
RESULTS_DIR = "analysis_results"

# Create results directory and subdirectories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/figures", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/tables", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/statistics", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/false_positives", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/visualizations", exist_ok=True)

# Configure logger
logger = setup_logger(log_file=f"{RESULTS_DIR}/inference.log")

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class AnomalyAnalyzer:
    def __init__(self, model_path, config_path, test_data_path, results_dir):
        self.model_path = model_path
        self.config_path = config_path
        self.test_data_path = test_data_path
        self.results_dir = results_dir
        self.device = device
        
        # Load configuration
        self.load_config()
        
        # Load model
        self.load_model()
        
        # Setup class mappings
        self.setup_mappings()
        
        # Load test data
        self.load_test_data()
        
    def setup_mappings(self):
        """Setup class mappings for easier interpretation"""
        self.class_names = ['Normal', 'Type 1', 'Type 2', 'Type 3', 'Type 4']
        self.class_colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        self.feature_names = ['SFN', 'Slot', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI']
        
        # Anomaly type descriptions for reporting
        self.anomaly_descriptions = {
            1: "Unnecessary Retransmission: A retransmission was sent despite the previous transmission being successful",
            2: "Missing Retransmission: A retransmission should have been triggered but wasn't",
            3: "New Data No Retransmission: New data was sent while a retransmission was pending",
            4: "Maximum Retransmissions Achieved: The maximum number of retransmissions was reached"
        }
        
    def load_config(self):
        """Load model configuration from saved file"""
        logger.info(f"Loading configuration from: {self.config_path}")
        
        # Load the checkpoint (contains both model state and config)
        checkpoint = torch.load(self.config_path, map_location=self.device)
        
        # Extract config from the checkpoint
        if 'config' in checkpoint:
            self.config = checkpoint['config']
            logger.info(f"Configuration loaded successfully")
        else:
            raise ValueError(f"Config not found in checkpoint file: {self.config_path}")
        
        # Set threshold for binary classification
        self.threshold = self.config.get('threshold', 0.5)
        logger.info(f"Using classification threshold: {self.threshold}")
    
    def load_model(self):
        """Load the trained model"""
        logger.info(f"Loading model from: {self.model_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Initialize model with config
        self.model = BiGRUAnomalyDetector(
            feature_dims=self.config['all_feature_dims'],
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            gru_dropout=self.config['gru_dropout'],
            position_encoding=self.config['position_encoding'],
            bidirectional=self.config['bidirectional'],
            attention_heads=self.config['attention_heads'],
            attention_dropout=self.config['attention_dropout'],
            classifier_hidden_dim=self.config['classifier_hidden_dim'],
            classifier_dropout=self.config['classifier_dropout'],
            device=self.device
        ).to(self.device)
        
        # Load weights from the model_state_dict key
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model weights loaded successfully")
        else:
            raise ValueError(f"model_state_dict not found in checkpoint file: {self.model_path}")
        
        self.model.eval()
        
        # Report model size
        total_params = self.model.count_parameters()
        logger.info(f"Model loaded successfully with {total_params:,} parameters")
        
    def load_test_data(self):
        """Load and prepare test data"""
        logger.info(f"Loading test data from: {self.test_data_path}")
        
        # Create data loaders using the function from dataset.py
        self.train_loader, self.test_loader, _ = create_data_loaders(
            train_parquet_path=self.config['train_parquet_path'],
            test_parquet_path=self.test_data_path,
            all_features=self.config['all_features'],
            all_feature_dims=self.config['all_feature_dims'],
            seq_len=self.config['seq_len'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            sample_fraction=1.0  # Use all test data
        )
        
        logger.info(f"Test data loaded successfully with {len(self.test_loader)} batches")

    def analyze_dataset_statistics(self):
        """Generate and save detailed dataset statistics"""
        logger.info("Analyzing dataset statistics...")
        
        # Get dataset from test loader
        dataset = self.test_loader.dataset
        dataset_train = self.train_loader.dataset
        
        # Basic statistics
        total_sequences = len(dataset)
        total_timesteps = sum(len(seq) for seq in dataset.labels)
        
        # Count by class for test dataset
        class_counts = Counter()
        for seq_labels in dataset.labels:
            for label in seq_labels:
                class_counts[label.item()] += 1
                
        # Calculate percentages for test dataset
        class_percentages = {cls: count/total_timesteps*100 for cls, count in class_counts.items()}
        
        # Create a DataFrame for easier reporting for test dataset
        stats_df = pd.DataFrame({
            'Class': [self.class_names[i] for i in range(5)],
            'Count': [class_counts[i] for i in range(5)],
            'Percentage': [class_percentages.get(i, 0) for i in range(5)]
        })
        
        # Save statistics to CSV for test dataset
        stats_file = f"{self.results_dir}/statistics/dataset_statistics.csv"
        stats_df.to_csv(stats_file, index=False)
        logger.info(f"Dataset statistics saved to {stats_file}")
        
        # Create visualization of class distribution for test dataset with logarithmic scale
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Class', y='Count', hue='Class', data=stats_df, palette=self.class_colors, legend=False)
        plt.yscale('log')  # Set log scale for y-axis to handle large differences
        
        # Add count and percentage labels for test dataset
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2.,
                    height * 1.1,  # Adjust vertical position of text in log scale
                    f'{height:,}\n({stats_df.iloc[i]["Percentage"]:.2f}%)',
                    ha="center", fontsize=10)
        
        plt.title('Class Distribution in Test Dataset (Log Scale)', fontsize=14)
        plt.ylabel('Count (log scale)', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        plt.tight_layout()
        
        # Save figure for test dataset
        plt.savefig(f"{self.results_dir}/figures/class_distribution_log.png", dpi=300, bbox_inches='tight')
        
        # Also create a percentage-based visualization
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Class', y='Percentage', hue='Class', palette=self.class_colors, data=stats_df, legend=False)
        
        # Add percentage labels
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2.,
                    height + 0.5,
                    f'{height:.2f}%',
                    ha="center", fontsize=10)
        
        plt.title('Class Distribution in Test Dataset (Percentage)', fontsize=14)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        plt.tight_layout()
        
        # Save percentage-based figure
        plt.savefig(f"{self.results_dir}/figures/class_distribution_percentage.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Repeat the same process for the training dataset
        total_sequences_train = len(dataset_train)
        total_timesteps_train = sum(len(seq) for seq in dataset_train.labels)
        
        # Count by class for training dataset
        class_counts_train = Counter()
        for seq_labels in dataset_train.labels:
            for label in seq_labels:
                class_counts_train[label.item()] += 1
                
        # Calculate percentages for training dataset
        class_percentages_train = {cls: count/total_timesteps_train*100 for cls, count in class_counts_train.items()}
        
        # Create a DataFrame for easier reporting for training dataset
        stats_df_train = pd.DataFrame({
            'Class': [self.class_names[i] for i in range(5)],
            'Count': [class_counts_train[i] for i in range(5)],
            'Percentage': [class_percentages_train.get(i, 0) for i in range(5)]
        })
        
        # Save statistics to CSV for training dataset
        stats_file_train = f"{self.results_dir}/statistics/dataset_train_statistics.csv"
        stats_df_train.to_csv(stats_file_train, index=False)
        logger.info(f"Training dataset statistics saved to {stats_file_train}")
        
        # Create visualization of class distribution for training dataset with logarithmic scale
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Class', y='Count', hue='Class', data=stats_df_train, palette=self.class_colors, legend=False)
        plt.yscale('log')  # Set log scale for y-axis
        
        # Add count and percentage labels for training dataset
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2.,
                    height * 1.1,  # Adjust for log scale
                    f'{height:,}\n({stats_df_train.iloc[i]["Percentage"]:.2f}%)',
                    ha="center", fontsize=10)
        
        plt.title('Class Distribution in Training Dataset (Log Scale)', fontsize=14)
        plt.ylabel('Count (log scale)', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        plt.tight_layout()
        
        # Save figure for training dataset
        plt.savefig(f"{self.results_dir}/figures/class_distribution_train_log.png", dpi=300, bbox_inches='tight')
        
        # Also create a percentage-based visualization for training set
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Class', y='Percentage', hue='Class',data=stats_df_train, palette=self.class_colors, legend=False)
        
        # Add percentage labels
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2.,
                    height + 0.5,
                    f'{height:.2f}%',
                    ha="center", fontsize=10)
        
        plt.title('Class Distribution in Training Dataset (Percentage)', fontsize=14)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        plt.tight_layout()
        
        # Save percentage-based figure for training set
        plt.savefig(f"{self.results_dir}/figures/class_distribution_train_percentage.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add Class Weight Visualization (for training dataset only)
        self.visualize_class_weights(dataset_train, stats_df_train)
        
        return stats_df
    
    def visualize_class_weights(self, dataset_train, stats_df_train):
        """Create visualizations of class weights"""
        logger.info("Visualizing class weights...")
        
        # Get class weights from training dataset if available
        if hasattr(dataset_train, 'class_weights') and hasattr(dataset_train, 'binary_weights'):
            class_weights = dataset_train.class_weights.cpu().numpy()
            binary_weights = dataset_train.binary_weights.cpu().numpy()
            
            # Create a dataframe for multiclass weights
            weight_df = pd.DataFrame({
                'Class': [self.class_names[i] for i in range(5)],
                'Weight': class_weights,
                'Count': [stats_df_train.iloc[i]['Count'] for i in range(5)]
            })
            
            # Plot multiclass weights
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Class', y='Weight', hue='Class',data=weight_df, palette=self.class_colors, legend=False)
            
            # Add weight and count labels
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x() + p.get_width()/2.,
                        height + 0.1,
                        f'{height:.2f}\n({weight_df.iloc[i]["Count"]:,})',
                        ha="center", fontsize=10)
            
            plt.title('Class Weights for Training Dataset', fontsize=14)
            plt.ylabel('Weight', fontsize=12)
            plt.xlabel('Class', fontsize=12)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{self.results_dir}/figures/class_weights.png", dpi=300, bbox_inches='tight')
            
            # Create a dataframe for binary weights
            binary_df = pd.DataFrame({
                'Class': ['Normal', 'Anomaly'],
                'Weight': binary_weights
            })
            
            # Plot binary weights
            plt.figure(figsize=(8, 6))
            ax = sns.barplot(x='Class', y='Weight', data=binary_df, palette=['#2ca02c', '#d62728'])
            
            # Add weight labels
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x() + p.get_width()/2.,
                        height + 0.1,
                        f'{height:.2f}',
                        ha="center", fontsize=10)
            
            plt.title('Binary Class Weights (Normal vs Anomaly)', fontsize=14)
            plt.ylabel('Weight', fontsize=12)
            plt.xlabel('Class', fontsize=12)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{self.results_dir}/figures/binary_weights.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Class weight visualizations saved")
        else:
            logger.info("Class weights not available in dataset, skipping visualization")
        
    def run_inference(self):
        """Run inference on test data and collect predictions and metrics"""
        logger.info("Running inference on test data...")
        
        self.model.eval()
        all_binary_preds = []
        all_binary_probs = []
        all_labels = []
        all_timesteps = []
        all_feature_data = []
        all_attn_weights = []
        all_errors = []
        all_seq_indices = []  # Track sequence indices for length analysis
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Inference")):
                # Move data to device
                feature_data = batch['feature_data'].to(self.device)
                labels = batch['label'].to(self.device)
                timestamps = batch['timestamp']
                
                # Forward pass
                outputs = self.model(feature_data)
                
                # Get predictions
                binary_probs = outputs['binary_probs'].squeeze(-1)  # [batch_size, seq_len]
                binary_preds = (binary_probs >= self.threshold).float()
                
                # Store results
                all_binary_preds.append(binary_preds.cpu())
                all_binary_probs.append(binary_probs.cpu())
                all_labels.append(labels.cpu())
                all_timesteps.append(timestamps)
                all_feature_data.append(feature_data.cpu())
                all_attn_weights.append(outputs['instance_attn_weights'].cpu())
                all_errors.append(outputs['error_per_timestep'].cpu())
                
                # Store sequence indices
                batch_size = feature_data.size(0)
                seq_indices = []
                for i in range(batch_size):
                    seq_idx = batch_idx * self.config['batch_size'] + i
                    seq_len = sum([1 for t in timestamps[i] if t])  # Count valid timestamps
                    seq_indices.append((seq_idx, seq_len))
                all_seq_indices.extend(seq_indices)
        
        # Concatenate results
        self.binary_preds = torch.cat(all_binary_preds, dim=0).numpy()
        self.binary_probs = torch.cat(all_binary_probs, dim=0).numpy()
        self.labels = torch.cat(all_labels, dim=0).numpy()
        self.feature_data = torch.cat(all_feature_data, dim=0).numpy()
        self.attn_weights = torch.cat(all_attn_weights, dim=0).numpy()
        self.errors = torch.cat(all_errors, dim=0).numpy()
        self.seq_indices = all_seq_indices
        
        # Flatten for metrics
        self.binary_preds_flat = self.binary_preds.reshape(-1)
        self.binary_probs_flat = self.binary_probs.reshape(-1)
        self.labels_flat = self.labels.reshape(-1)
        
        # Create binary labels (normal vs anomaly)
        self.binary_labels_flat = (self.labels_flat > 0).astype(int)
        
        # Store timesteps
        self.timesteps = []
        for batch_timestamps in all_timesteps:
            for seq_timestamps in batch_timestamps:
                self.timesteps.extend(seq_timestamps)
        
        logger.info(f"Inference completed on {len(self.binary_preds_flat)} timesteps")
        
        return self.binary_preds, self.binary_probs, self.labels
    
    def calculate_metrics(self):
        """Calculate and store performance metrics"""
        logger.info("Calculating performance metrics...")
        
        # Calculate binary classification metrics
        self.metrics = {}
        
        # Basic metrics
        self.metrics['accuracy'] = accuracy_score(self.binary_labels_flat, self.binary_preds_flat)
        self.metrics['precision'] = precision_score(self.binary_labels_flat, self.binary_preds_flat)
        self.metrics['recall'] = recall_score(self.binary_labels_flat, self.binary_preds_flat)
        self.metrics['f1'] = f1_score(self.binary_labels_flat, self.binary_preds_flat)
        
        # ROC AUC
        try:
            self.metrics['auc'] = roc_auc_score(self.binary_labels_flat, self.binary_probs_flat)
        except:
            self.metrics['auc'] = 0.0
            logger.warning("Could not calculate ROC AUC score")
        
        # Confusion matrix
        self.metrics['confusion_matrix'] = confusion_matrix(self.binary_labels_flat, self.binary_preds_flat)
        
        # Class-specific metrics
        self.class_metrics = {}
        for i in range(5):
            # For class i, calculate detection rate
            class_indices = (self.labels_flat == i)
            
            if np.sum(class_indices) > 0:
                if i == 0:  # Normal class
                    class_correct = np.sum((self.binary_preds_flat == 0) & class_indices)
                else:  # Anomaly classes
                    class_correct = np.sum((self.binary_preds_flat == 1) & class_indices)
                
                detection_rate = class_correct / np.sum(class_indices)
                self.class_metrics[i] = {
                    'total': np.sum(class_indices),
                    'correct': class_correct,
                    'detection_rate': detection_rate
                }
            else:
                self.class_metrics[i] = {
                    'total': 0,
                    'correct': 0,
                    'detection_rate': 0.0
                }
        
        # Log metrics
        logger.info("Binary Classification Metrics:")
        logger.info(f"  Accuracy: {self.metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {self.metrics['precision']:.4f}")
        logger.info(f"  Recall: {self.metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {self.metrics['f1']:.4f}")
        logger.info(f"  ROC AUC: {self.metrics['auc']:.4f}")
        
        # Log confusion matrix
        cm = self.metrics['confusion_matrix']
        logger.info("\nConfusion Matrix (Normal vs Anomaly):")
        logger.info("                  Predicted")
        logger.info("                Normal  Anomaly")
        logger.info(f"Actual Normal   {cm[0][0]:<8} {cm[0][1]:<8}")
        logger.info(f"Actual Anomaly  {cm[1][0]:<8} {cm[1][1]:<8}")
        
        # Log class detection rates
        logger.info("\nClass Detection Rates:")
        for i in range(5):
            metrics = self.class_metrics[i]
            logger.info(f"  Class {i} ({self.class_names[i]}): "
                       f"{metrics['correct']}/{metrics['total']} = "
                       f"{metrics['detection_rate']:.4f}")
        
        return self.metrics
    
    def plot_roc_curve(self):
        """Plot ROC curve for binary classification"""
        logger.info("Plotting ROC curve...")
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(self.binary_labels_flat, self.binary_probs_flat)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (area = {self.metrics["auc"]:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Add threshold marker
        # Find the closest threshold to our selected one
        bin_preds = (self.binary_probs_flat >= self.threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.binary_labels_flat, bin_preds).ravel()
        current_fpr = fp / (fp + tn)
        current_tpr = tp / (tp + fn)
        plt.plot(current_fpr, current_tpr, 'ro', markersize=10, 
                 label=f'Threshold = {self.threshold}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(self):
        """Plot precision-recall curve for binary classification"""
        logger.info("Plotting precision-recall curve...")
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(
            self.binary_labels_flat, self.binary_probs_flat)
        
        # Calculate average precision
        avg_precision = average_precision_score(self.binary_labels_flat, self.binary_probs_flat)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkorange', lw=2,
                 label=f'PR curve (AP = {avg_precision:.4f})')
        
        # Add threshold marker
        # Find the closest threshold index to our selected one
        threshold_idx = np.argmin(np.abs(thresholds - self.threshold)) if len(thresholds) > 0 else 0
        if threshold_idx < len(precision) - 1:  # Ensure we're within bounds
            plt.plot(recall[threshold_idx], precision[threshold_idx], 'ro', markersize=10,
                    label=f'Threshold = {self.threshold}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/precision_recall_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_threshold_analysis(self):
        """Plot metrics vs threshold to help select optimal threshold"""
        logger.info("Plotting threshold analysis...")
        
        # Define thresholds to check
        thresholds = np.linspace(0.05, 0.95, 19)  # 0.05 to 0.95 in steps of 0.05
        
        # Metrics to track
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        # Calculate metrics for each threshold
        for threshold in thresholds:
            # Apply threshold to get binary predictions
            preds = (self.binary_probs_flat >= threshold).astype(int)
            
            # Calculate metrics
            acc = accuracy_score(self.binary_labels_flat, preds)
            prec = precision_score(self.binary_labels_flat, preds)
            rec = recall_score(self.binary_labels_flat, preds)
            f1 = f1_score(self.binary_labels_flat, preds)
            
            # Store metrics
            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1)
        
        # Create a figure
        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, accuracies, 'o-', label='Accuracy', lw=2)
        plt.plot(thresholds, precisions, 's-', label='Precision', lw=2)
        plt.plot(thresholds, recalls, '^-', label='Recall', lw=2)
        plt.plot(thresholds, f1_scores, 'D-', label='F1 Score', lw=2)
        
        # Add vertical line for current threshold
        plt.axvline(x=self.threshold, color='r', linestyle='--', label=f'Current Threshold ({self.threshold})')
        
        # Add grid, labels, and legend
        plt.grid(True, alpha=0.3)
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Metric Value', fontsize=12)
        plt.title('Classification Metrics vs. Threshold', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/threshold_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save the data as CSV
        threshold_df = pd.DataFrame({
            'Threshold': thresholds,
            'Accuracy': accuracies,
            'Precision': precisions,
            'Recall': recalls,
            'F1_Score': f1_scores
        })
        threshold_df.to_csv(f"{self.results_dir}/statistics/threshold_analysis.csv", index=False)
        
        # Find optimal thresholds for each metric
        best_acc_idx = np.argmax(accuracies)
        best_f1_idx = np.argmax(f1_scores)
        
        logger.info(f"Optimal threshold for accuracy: {thresholds[best_acc_idx]:.2f} (Accuracy = {accuracies[best_acc_idx]:.4f})")
        logger.info(f"Optimal threshold for F1 score: {thresholds[best_f1_idx]:.2f} (F1 = {f1_scores[best_f1_idx]:.4f})")
    
    def plot_class_detection_rates(self):
        """Plot detection rates by class"""
        logger.info("Plotting class detection rates...")
        
        # Create DataFrame for easier plotting
        class_df = pd.DataFrame({
            'Class': [self.class_names[i] for i in range(5)],
            'Detection Rate': [self.class_metrics[i]['detection_rate'] for i in range(5)],
            'Total': [self.class_metrics[i]['total'] for i in range(5)]
        })
        
        # Create figure
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Class', y='Detection Rate', hue='Class', palette=self.class_colors, data=class_df, legend=False)
        
        # Add rate and count labels
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2.,
                    height + 0.02,
                    f'{height:.4f}\n({class_df.iloc[i]["Total"]:,})',
                    ha="center", fontsize=10)
        
        plt.title('Detection Rate by Class', fontsize=14)
        plt.ylabel('Detection Rate', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        plt.ylim(0, 1.1)  # Set y-axis to max 1.1 for better visualization
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/class_detection_rates.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_normalized_confusion_matrix(self, cm, class_names, output_dir=None, filename="normalized_confusion_matrix.png", 
                                    figsize=(10, 8), exclude_normal=True):
        """
        Plot a professional normalized confusion matrix for graduate thesis.
        
        Args:
            cm: Confusion matrix array (raw counts)
            class_names: List of class names
            output_dir: Directory to save the plot (uses self.results_dir if None)
            filename: Filename for the saved plot
            figsize: Figure size as (width, height) tuple
            exclude_normal: If True, excludes the Normal class (class 0) from the matrix
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        
        if output_dir is None:
            output_dir = f"{self.results_dir}/figures"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter out normal class if specified
        if exclude_normal and len(class_names) > 1:
            anomaly_indices = list(range(1, len(class_names)))
            cm_filtered = cm[anomaly_indices, :][:, anomaly_indices]
            class_names_filtered = [class_names[i] for i in anomaly_indices]
        else:
            cm_filtered = cm
            class_names_filtered = class_names
        
        # Create normalized confusion matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = np.divide(cm_filtered.astype('float'), 
                            np.maximum(cm_filtered.sum(axis=1)[:, np.newaxis], 1))
            cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
        
        # Set publication-quality styling
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.edgecolor': '#333333',
            'axes.linewidth': 1.0
        })
        
        # Create the main visualization with percentages only (publication style)
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        
        # Use a professional colormap - Blues is common in academic publications
        cmap = plt.cm.Blues
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        
        # Create the heatmap
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, norm=norm)
        
        # Add colorbar with refined styling
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Frequency', rotation=270, labelpad=20, fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        # Configure axes and ticks
        tick_marks = np.arange(len(class_names_filtered))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names_filtered, fontsize=11, rotation=45, ha='right')
        ax.set_yticklabels(class_names_filtered, fontsize=11)
        
        # Create text annotations with percentages
        thresh = cm_norm.max() / 2
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                # Only display cell value - clean academic style
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", 
                        ha="center", va="center", 
                        color="white" if cm_norm[i, j] > thresh else "black",
                        fontsize=10)
        
        # Refine titles and labels
        title_suffix = " (Anomaly Classes)" if exclude_normal else ""
        ax.set_title(f"Confusion Matrix{title_suffix}", fontsize=14, pad=10)
        ax.set_xlabel("Predicted Class", fontsize=12, labelpad=10)
        ax.set_ylabel("True Class", fontsize=12, labelpad=10)
        
        # Ensure grid lines appear between cells
        ax.set_xticks(np.arange(-.5, len(class_names_filtered), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(class_names_filtered), 1), minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        
        # Adjust layout and save with high resolution
        fig.tight_layout()
        
        # Modify filename if needed
        if exclude_normal:
            filename = filename.replace(".png", "_anomaly_only.png")
        
        # Save high-resolution image suitable for publication
        plt.savefig(os.path.join(output_dir, "aaa"), dpi=600, bbox_inches="tight", 
                    format="png", facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Professional confusion matrix plot saved to {output_dir}")
        
        return cm_norm
    
    def plot_confusion_matrix(self):
        """Plot detailed confusion matrix with both raw counts and percentages"""
        logger.info("Plotting confusion matrices...")
        
        # Get the binary confusion matrix
        binary_cm = self.metrics['confusion_matrix']
        
        # Plot the binary confusion matrix with normalization
        self.plot_normalized_confusion_matrix(
            binary_cm, 
            ['Normal', 'Anomaly'], 
            filename="binary_confusion_matrix_normalized.png"
        )
        
        # Create a multi-class confusion matrix (0-4)
        cm_multiclass = confusion_matrix(self.labels_flat, 
                                        (self.binary_preds_flat * self.labels_flat))
        
        # Binary labels for all anomaly classes (1-4) are 1
        # But predictions may not match the specific anomaly type
        # We need to set predictions for normal class correctly
        for i in range(1, 5):
            # Where true label is i (anomaly) but prediction is 0 (normal)
            cm_multiclass[i, 0] = np.sum((self.labels_flat == i) & (self.binary_preds_flat == 0))
        
        # Plot the multi-class normalized confusion matrix
        self.plot_normalized_confusion_matrix(
            cm_multiclass,
            self.class_names,
            filename="multiclass_confusion_matrix_normalized.png"
        )
        
        # Create a DataFrame for the confusion matrix (we keep this for compatibility)
        cm_df = pd.DataFrame(
            cm_multiclass,
            index=[f'True {name}' for name in self.class_names],
            columns=[f'Pred {name}' if i == 0 else f'Pred Anomaly{i}' if i > 0 else 'Pred Normal' 
                    for i, name in enumerate(self.class_names)]
        )
        
        # Create figure for the traditional confusion matrix visualization
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True)
        plt.title('Confusion Matrix (Detailed)', fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Multiple confusion matrix visualizations saved")

    def plot_binary_confusion_matrix(self):
        """Plot professional binary confusion matrix for graduate thesis"""
        logger.info("Plotting binary confusion matrix...")
        
        # Get the confusion matrix for binary classification
        cm = self.metrics['confusion_matrix']
        
        # Extract values
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics (stored but not displayed on plot)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Create figure with professional proportions
        plt.figure(figsize=(7, 6), facecolor='white')
        
        # Prepare cell text
        cell_text = [
            ['{:,}'.format(tn), '{:,}'.format(fp)],
            ['{:,}'.format(fn), '{:,}'.format(tp)]
        ]
        
        # Calculate professional color scheme
        total = np.sum(cm)
        norm_values = cm / total
        
        # Blues for true predictions, lighter colors for false predictions
        cell_colors = [
            [plt.cm.Blues(0.4 + 0.6 * norm_values[0, 0]), plt.cm.Greys(0.3 + 0.5 * norm_values[0, 1])],
            [plt.cm.Greys(0.3 + 0.5 * norm_values[1, 0]), plt.cm.Blues(0.4 + 0.6 * norm_values[1, 1])]
        ]
        
        # Create the table with a professional appearance
        table = plt.table(
            cellText=cell_text,
            cellColours=cell_colors,
            colLabels=['Predicted\nNormal', 'Predicted\nAnomaly'],
            rowLabels=['True\nNormal', 'True\nAnomaly'],
            loc='center',
            cellLoc='center'
        )
        
        # Enhance table styling
        table.scale(1, 1.8)
        table.set_fontsize(12)
        
        # Style the cells
        for key, cell in table.get_celld().items():
            cell.set_edgecolor('white')
            cell.set_linewidth(1.5)
            
            # Make headers bold
            if key[0] == 0 or key[1] == -1:
                cell.set_text_props(weight='bold', fontsize=13)
        
        # Hide axes
        plt.axis('off')
        
        # Add title
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', y=0.9)
        
        # Save figure with high resolution
        plt.savefig(f"{self.results_dir}/figures/binary_confusion_matrix.png", dpi=600, bbox_inches='tight')
        plt.close()
        
        logger.info("Binary confusion matrix plot saved")
    
    def analyze_false_positives(self):
        """Analyze false positives to identify potential unknown anomalies"""
        logger.info("Analyzing false positives...")
        
        # Find false positives
        false_pos_mask = (self.binary_preds_flat == 1) & (self.binary_labels_flat == 0)
        false_pos_indices = np.where(false_pos_mask)[0]
        
        if len(false_pos_indices) == 0:
            logger.info("No false positives found.")
            return
        
        logger.info(f"Found {len(false_pos_indices)} false positives")
        
        # Get data for false positives
        fp_probs = self.binary_probs_flat[false_pos_indices]
        fp_features = self.feature_data.reshape(-1, self.feature_data.shape[-1])[false_pos_indices]
        
        # Get timestamps for false positives
        fp_timestamps = [self.timesteps[i] for i in false_pos_indices if i < len(self.timesteps)]
        
        # Create a DataFrame to store false positive data
        fp_df = pd.DataFrame({
            'timestamp': fp_timestamps[:len(fp_probs)],
            'probability': fp_probs,
            **{f'{self.feature_names[i]}': fp_features[:, i] for i in range(len(self.feature_names))}
        })
        
        # Sort by probability (highest first)
        fp_df = fp_df.sort_values('probability', ascending=False)
        
        # Save to CSV
        fp_file = f"{self.results_dir}/false_positives/false_positives.csv"
        fp_df.to_csv(fp_file, index=False)
        logger.info(f"False positives data saved to {fp_file}")
        
        # Calculate statistics for normal data
        normal_mask = (self.binary_labels_flat == 0) & (self.binary_preds_flat == 0)
        normal_indices = np.where(normal_mask)[0]
        
        if len(normal_indices) > 0:
            normal_features = self.feature_data.reshape(-1, self.feature_data.shape[-1])[normal_indices]
            normal_means = np.mean(normal_features, axis=0)
            normal_stds = np.std(normal_features, axis=0)
            
            # Calculate z-scores for false positives
            z_scores = np.abs((fp_features - normal_means) / (normal_stds + 1e-10))
            
            # Find the most suspicious features for each false positive
            suspicious_features = []
            for i, row_z_scores in enumerate(z_scores):
                # Find features with z-score > 1.5
                suspicious = [(self.feature_names[j], row_z_scores[j]) 
                              for j in range(len(self.feature_names)) 
                              if row_z_scores[j] > 1.5]
                
                # Sort by z-score (descending)
                suspicious.sort(key=lambda x: x[1], reverse=True)
                
                if suspicious:
                    features_str = "--".join([f"{f} ({z:.2f})" for f, z in suspicious[:3]])
                else:
                    features_str = "None"
                    
                suspicious_features.append(features_str)
            
            # Add to DataFrame
            fp_df['suspicious_features'] = suspicious_features[:len(fp_df)]
            
            # Save updated DataFrame
            fp_df.to_csv(fp_file, index=False)
            
            # Perform clustering analysis on false positives
            self.cluster_false_positives(fp_features, fp_df)
    
    def cluster_false_positives(self, fp_features, fp_df):
        """Cluster false positives to identify potential patterns"""
        logger.info("Clustering false positives...")
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(fp_features)
        
        # Apply PCA for visualization with 3 components instead of 2
        pca = PCA(n_components=3)
        fp_pca = pca.fit_transform(scaled_features)
        
        # Get the explained variance for each component
        explained_variance = pca.explained_variance_ratio_ * 100
        
        # Determine optimal number of clusters using silhouette method
        from sklearn.metrics import silhouette_score
        
        max_clusters = min(8, len(fp_features) // 5)  # Limit based on sample size
        if max_clusters < 2:
            logger.info("Not enough false positives for clustering")
            return
            
        scores = []
        for n in range(2, max_clusters + 1):
            # Skip if n exceeds sample size
            if n >= len(fp_features):
                break
                
            kmeans = KMeans(n_clusters=n, random_state=SEED, n_init=10)
            labels = kmeans.fit_predict(scaled_features)
            
            # Skip if only one cluster is found
            if len(np.unique(labels)) < 2:
                continue
                
            score = silhouette_score(scaled_features, labels)
            scores.append((n, score))
        
        if not scores:
            logger.info("Could not determine optimal clusters")
            return
            
        # Get optimal number of clusters
        optimal_k = max(scores, key=lambda x: x[1])[0]
        logger.info(f"Optimal number of clusters: {optimal_k}")
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=SEED, n_init=10)
        fp_clusters = kmeans.fit_predict(scaled_features)
        
        # Add cluster information to DataFrame
        fp_df['cluster'] = fp_clusters[:len(fp_df)]
        fp_df.to_csv(f"{self.results_dir}/false_positives/false_positives_clustered.csv", index=False)
        
        # Save PCA components and their explained variance
        pca_df = pd.DataFrame(fp_pca, columns=[f'PC{i+1}' for i in range(3)])
        pca_df['cluster'] = fp_clusters
        pca_df.to_csv(f"{self.results_dir}/false_positives/pca_components.csv", index=False)
        
        # Calculate cluster statistics
        cluster_stats = []
        for i in range(optimal_k):
            cluster_mask = fp_clusters == i
            cluster_count = np.sum(cluster_mask)
            
            # Get cluster centroid
            cluster_centroid = kmeans.cluster_centers_[i]
            
            # Calculate feature importance for this cluster
            feature_importance = np.abs(cluster_centroid - np.mean(scaled_features, axis=0))
            top_features = np.argsort(feature_importance)[::-1][:3]
            
            # Format top features
            top_features_str = ", ".join([f"{self.feature_names[j]} ({feature_importance[j]:.2f})" 
                                        for j in top_features])
            
            # Anomaly type prediction (speculative)
            prediction = self.predict_potential_anomaly_type(kmeans.cluster_centers_[i], scaler)
            
            cluster_stats.append({
                'Cluster': i,
                'Count': cluster_count,
                'Percentage': cluster_count / len(fp_clusters) * 100,
                'Top Features': top_features_str,
                'Possible Anomaly Type': prediction
            })
        
        # Save cluster statistics
        cluster_stats_df = pd.DataFrame(cluster_stats)
        cluster_stats_df.to_csv(f"{self.results_dir}/false_positives/cluster_statistics.csv", index=False)
        
        # Create 3D PCA plot
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each cluster with different color
        for i in range(optimal_k):
            cluster_points = fp_pca[fp_clusters == i]
            ax.scatter(
                cluster_points[:, 0], 
                cluster_points[:, 1], 
                cluster_points[:, 2],
                label=f'Cluster {i}',
                alpha=0.7
            )
        
        # Add cluster centers
        centers_pca = pca.transform(kmeans.cluster_centers_)
        ax.scatter(
            centers_pca[:, 0],
            centers_pca[:, 1],
            centers_pca[:, 2],
            marker='X',
            s=200,
            c='red',
            edgecolors='black',
            label='Cluster Centers'
        )
        
        # Add axis labels with explained variance
        ax.set_xlabel(f'PC1 ({explained_variance[0]:.2f}% variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_variance[1]:.2f}% variance)', fontsize=12)
        ax.set_zlabel(f'PC3 ({explained_variance[2]:.2f}% variance)', fontsize=12)
        
        ax.set_title('3D PCA Visualization of False Positive Clusters', fontsize=14)
        ax.legend()
        
        # Save the 3D plot
        plt.savefig(f"{self.results_dir}/figures/false_positive_clusters_3d.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create 2D plots for each pair of principal components for better readability
        component_pairs = [(0, 1), (0, 2), (1, 2)]
        for i, j in component_pairs:
            plt.figure(figsize=(12, 10))
            
            # Plot each cluster
            for cluster_id in range(optimal_k):
                mask = fp_clusters == cluster_id
                plt.scatter(
                    fp_pca[mask, i],
                    fp_pca[mask, j],
                    label=f'Cluster {cluster_id}',
                    alpha=0.7
                )
            
            # Plot cluster centers
            plt.scatter(
                centers_pca[:, i],
                centers_pca[:, j],
                marker='X',
                s=200,
                c='red',
                edgecolors='black',
                label='Cluster Centers'
            )
            
            # Add annotations for clusters
            for k, (x, y) in enumerate(centers_pca[:, [i, j]]):
                prediction = cluster_stats[k]['Possible Anomaly Type']
                plt.annotate(
                    f"Cluster {k}\n{prediction}",
                    (x, y),
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
            
            plt.xlabel(f'PC{i+1} ({explained_variance[i]:.2f}% variance)', fontsize=12)
            plt.ylabel(f'PC{j+1} ({explained_variance[j]:.2f}% variance)', fontsize=12)
            plt.title(f'PCA Visualization (PC{i+1} vs PC{j+1})', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{self.results_dir}/figures/false_positive_clusters_pc{i+1}_pc{j+1}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"False positive clustering analysis saved")
    
    def predict_potential_anomaly_type(self, cluster_center, scaler):
        """Make an educated guess about what anomaly type a cluster might represent"""
        # This is a simplified heuristic based on domain knowledge
        # Scale back to original feature space
        center_orig = scaler.inverse_transform(cluster_center.reshape(1, -1)).flatten()
        
        # Extract key feature values
        retx = center_orig[self.feature_names.index('ReTx')]
        crc = center_orig[self.feature_names.index('CRC')]
        ndi = center_orig[self.feature_names.index('NDI')]
        harq = center_orig[self.feature_names.index('HARQ')]
        
        # Apply domain-specific heuristics
        if retx > 4 and crc < 0.5:  # High retransmission count but CRC success
            return "Possible Unnecessary ReTx"
        elif crc > 0.5 and retx < 1:  # CRC failure but no retransmission
            return "Possible Missing ReTx"
        elif ndi > 0.5 and crc > 0.5 and retx < 1:  # New data with failed CRC but no retx
            return "Possible New Data No ReTx"
        elif retx > 7:  # Very high retransmission count
            return "Possible Max ReTx Achieved"
        else:
            return "Unknown Anomaly Pattern"
    
    def analyze_time_series(self):
        """Analyze and visualize time-series predictions on sample sequences"""
        logger.info("Analyzing time-series predictions...")
        
        # Select a few interesting sequences that contain anomalies 
        # and false positives for visualization
        
        # Find sequences with anomalies for each type
        interesting_seqs = {}
        for anomaly_type in range(1, 5):
            # Find sequence indices where this anomaly type appears
            seq_with_anomaly = [i for i in range(len(self.test_loader.dataset)) 
                               if anomaly_type in self.test_loader.dataset.labels[i]]
            
            if seq_with_anomaly:
                # Select a random sequence with this anomaly
                np.random.seed(SEED)
                interesting_seqs[f"anomaly_type_{anomaly_type}"] = np.random.choice(seq_with_anomaly)
        
        # Find a sequence with a false positive
        fp_indices = np.where((self.binary_preds_flat == 1) & (self.binary_labels_flat == 0))[0]
        if len(fp_indices) > 0:
            # Map from flat index to sequence and position
            seq_len = self.config['seq_len']
            seq_idx = fp_indices[0] // seq_len
            interesting_seqs["false_positive"] = seq_idx
        
        # Plot each interesting sequence
        for name, seq_idx in interesting_seqs.items():
            # Get sequence data
            seq_data = self.test_loader.dataset[seq_idx]
            
            # Extract data
            feature_data = seq_data['feature_data'].numpy()
            labels = seq_data['label'].numpy()
            timestamps = seq_data['timestamp']
            
            # Get model predictions
            with torch.no_grad():
                feature_tensor = torch.tensor(feature_data).unsqueeze(0).to(self.device)
                outputs = self.model(feature_tensor)
                
                binary_probs = outputs['binary_probs'].squeeze().cpu().numpy()
                binary_preds = (binary_probs >= self.threshold).astype(float)
                attention = outputs['instance_attn_weights'].squeeze().cpu().numpy()
                errors = outputs['error_per_timestep'].squeeze().cpu().numpy()
            
            # Create binary labels
            binary_labels = (labels > 0).astype(float)
            
            # Create sequence position indices for x-axis (instead of timestamps)
            positions = np.arange(len(labels))
            
            # Filter out positions with empty timestamps if needed
            valid_positions = []
            valid_labels = []
            valid_binary_labels = []
            valid_binary_preds = []
            valid_binary_probs = []
            valid_attention = []
            valid_errors = []
            
            for i in range(len(positions)):
                if i < len(timestamps) and timestamps[i]:  # Use valid timesteps
                    valid_positions.append(positions[i])
                    valid_labels.append(labels[i])
                    valid_binary_labels.append(binary_labels[i])
                    valid_binary_preds.append(binary_preds[i])
                    valid_binary_probs.append(binary_probs[i])
                    valid_attention.append(attention[i])
                    valid_errors.append(errors[i])
            
            # If no valid positions (unlikely), just use all positions
            if not valid_positions:
                valid_positions = positions
                valid_labels = labels
                valid_binary_labels = binary_labels
                valid_binary_preds = binary_preds
                valid_binary_probs = binary_probs
                valid_attention = attention
                valid_errors = errors
            
            # Create multi-panel figure
            fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
            
            # Plot 1: True labels
            ax = axes[0]
            ax.plot(valid_positions, valid_labels, 'o-', label='True Labels', color='black', markersize=6)
            ax.set_ylabel('Anomaly Class', fontsize=12)
            ax.set_title(f'Time-Series Analysis: {name.replace("_", " ").title()}', fontsize=14)
            ax.set_ylim(-0.5, 4.5)
            ax.set_yticks(range(5))
            ax.set_yticklabels(self.class_names)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Plot 2: Binary predictions
            ax = axes[1]
            ax.plot(valid_positions, valid_binary_labels, 'o-', label='True Binary', color='black', markersize=6, alpha=0.7)
            ax.plot(valid_positions, valid_binary_preds, 's-', label='Predicted Binary', color='red', markersize=6)
            ax.set_ylabel('Binary Label', fontsize=12)
            ax.set_ylim(-0.2, 1.2)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Normal', 'Anomaly'])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Plot 3: Probability scores
            ax = axes[2]
            ax.plot(valid_positions, valid_binary_probs, 'o-', label='Anomaly Probability', color='blue', markersize=6)
            ax.axhline(y=self.threshold, color='r', linestyle='--', label=f'Threshold ({self.threshold})')
            ax.set_ylabel('Probability', fontsize=12)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Plot 4: Attention weights and reconstruction error
            ax = axes[3]
            ax.plot(valid_positions, valid_attention, 'o-', label='Attention Weights', color='purple', markersize=6)
            max_error = max(valid_errors) if valid_errors else 1
            ax.plot(valid_positions, [e/max_error for e in valid_errors], 
                   's-', label='Norm. Recon. Error', color='green', markersize=6, alpha=0.7)
            ax.set_ylabel('Weight/Error', fontsize=12)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # X-axis label
            ax.set_xlabel('Sequence Position', fontsize=12)
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{self.results_dir}/figures/time_series_{name}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Also create an attention heatmap for this sequence
            self.plot_attention_heatmap(name, feature_data, labels, attention)
            
            logger.info(f"Saved time-series analysis for {name}")
    
    def plot_attention_heatmap(self, name, feature_data, labels, attention_weights):
        """Create attention heatmap visualization for a sequence"""
        # Create a figure
        plt.figure(figsize=(12, 8))
        
        # Create attention heatmap 
        # For each position in the sequence, show attention weights
        att_data = attention_weights.reshape(-1, 1)
        
        # Create a DataFrame for the heatmap with positions and feature values
        heatmap_data = []
        for i, (attention, features) in enumerate(zip(attention_weights, feature_data)):
            # Create a row for this position
            row = {
                'Position': i,
                'Attention': attention,
                'Label': int(labels[i]),
                'Class': self.class_names[int(labels[i])]
            }
            # Add feature values
            for j, feature_name in enumerate(self.feature_names):
                row[feature_name] = features[j]
            
            heatmap_data.append(row)
        
        heatmap_df = pd.DataFrame(heatmap_data)
        
        # Create a pivot table for the heatmap - attention weights across sequence
        pivot_data = np.zeros((1, len(heatmap_df)))
        pivot_data[0, :] = heatmap_df['Attention'].values
        
        # Create custom colormap where high attention is darker
        cmap = plt.cm.get_cmap('YlOrRd')
        
        # Plot heatmap
        ax = plt.gca()
        im = ax.imshow(pivot_data, cmap=cmap, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Attention Weight', fontsize=12)
        
        # Set up x-axis (positions)
        ax.set_xticks(range(len(heatmap_df)))
        
        # Label x ticks with position, class and key features
        x_labels = []
        for i, row in heatmap_df.iterrows():
            if row['Label'] > 0:
                # For anomalies, show class and key features
                label = f"{i}: {row['Class'][:6]}\nHARQ={int(row['HARQ'])}, CRC={int(row['CRC'])}\nReTx={int(row['ReTx'])}"
            else:
                # For normal, just show position
                label = f"{i}"
            x_labels.append(label)
        
        # Only show some labels to avoid overcrowding
        if len(x_labels) > 20:
            # Show labels for anomalies and a subset of normal points
            sparse_labels = []
            for i, label in enumerate(x_labels):
                if ":" in label and label.split(":")[1].strip() != "Normal" or i % 5 == 0:
                    sparse_labels.append(label)
                else:
                    sparse_labels.append("")
            x_labels = sparse_labels
        
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        
        # Remove y ticks (only 1 row)
        ax.set_yticks([])
        
        # Add title and labels
        plt.title(f'Attention Weights: {name.replace("_", " ").title()}', fontsize=14)
        plt.xlabel('Sequence Position', fontsize=12)
        
        # Highlight anomaly positions
        anomaly_positions = [i for i, label in enumerate(labels) if label > 0]
        for pos in anomaly_positions:
            # Draw box around anomaly positions
            rect = plt.Rectangle((pos-0.5, -0.5), 1, 1, fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(rect)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f"{self.results_dir}/figures/attention_heatmap_{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved attention heatmap for {name}")
    
    def visualize_feature_importance(self):
        """Visualize feature importance based on model outputs"""
        logger.info("Visualizing feature importance...")
        
        # For each anomaly type, find instances where it occurs
        anomaly_feature_importance = {}
        
        for anomaly_type in range(1, 5):
            # Find indices for this anomaly type
            indices = np.where(self.labels_flat == anomaly_type)[0]
            
            if len(indices) == 0:
                continue
                
            # Extract feature data for these indices
            type_features = self.feature_data.reshape(-1, self.feature_data.shape[-1])[indices]
            
            # Extract normal feature data for comparison
            normal_indices = np.where(self.labels_flat == 0)[0]
            normal_features = self.feature_data.reshape(-1, self.feature_data.shape[-1])[normal_indices]
            
            # Calculate mean and std for each feature
            type_means = np.mean(type_features, axis=0)
            type_stds = np.std(type_features, axis=0)
            
            normal_means = np.mean(normal_features, axis=0)
            normal_stds = np.std(normal_features, axis=0)
            
            # Calculate feature importance as (normalized) absolute difference in means
            feature_importance = np.abs(type_means - normal_means) / (normal_stds + 1e-6)
            
            # Store feature importance
            anomaly_feature_importance[anomaly_type] = {
                'importance': feature_importance,
                'count': len(indices)
            }
        
        # Create feature importance plot
        plt.figure(figsize=(14, 10))
        x = np.arange(len(self.feature_names))
        width = 0.2
        
        # Create a bar for each anomaly type
        for i, anomaly_type in enumerate(range(1, 5)):
            if anomaly_type not in anomaly_feature_importance:
                continue
                
            importance = anomaly_feature_importance[anomaly_type]['importance']
            count = anomaly_feature_importance[anomaly_type]['count']
            
            plt.bar(x + (i - 1.5) * width, importance, width, 
                   label=f'{self.class_names[anomaly_type]} (n={count})', 
                   color=self.class_colors[anomaly_type], alpha=0.8)
        
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Feature Importance', fontsize=12)
        plt.title('Feature Importance by Anomaly Type', fontsize=14)
        plt.xticks(x, self.feature_names)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a heatmap of feature values by class
        feature_values_by_class = np.zeros((5, len(self.feature_names)))
        
        for cls in range(5):
            indices = np.where(self.labels_flat == cls)[0]
            if len(indices) > 0:
                features = self.feature_data.reshape(-1, self.feature_data.shape[-1])[indices]
                feature_values_by_class[cls] = np.mean(features, axis=0)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(feature_values_by_class, annot=True, fmt=".2f", cmap="YlGnBu",
                   xticklabels=self.feature_names, yticklabels=self.class_names)
        plt.title('Average Feature Values by Class', fontsize=14)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/feature_values_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_class_distribution_timeline(self):
        """Plot distribution of anomaly types over sequence indices to show patterns"""
        logger.info("Plotting class distribution timeline...")
        
        # Create data for timeline
        sequence_positions = []
        anomaly_types = []
        class_labels = []
        
        # Collect anomaly occurrences
        for i, seq_labels in enumerate(self.test_loader.dataset.labels):
            for j, label in enumerate(seq_labels):
                label = label.item()
                if label > 0:  # Only anomalies
                    sequence_positions.append(i)
                    anomaly_types.append(label)
                    class_labels.append(self.class_names[label])
        
        # If no anomalies found, skip
        if not sequence_positions:
            logger.info("No anomalies found for timeline visualization")
            return
        
        # Create DataFrame for plotting
        timeline_df = pd.DataFrame({
            'Sequence Index': sequence_positions,
            'Anomaly Type': anomaly_types,
            'Class': class_labels
        })
        
        # Sort by sequence index
        timeline_df = timeline_df.sort_values('Sequence Index')
        
        # Create figure
        plt.figure(figsize=(14, 8))
        scatter = plt.scatter(
            timeline_df['Sequence Index'], 
            timeline_df['Anomaly Type'],
            c=[self.class_colors[t] for t in timeline_df['Anomaly Type']],
            s=50, alpha=0.7
        )
        
        # Add legend
        classes = []
        for i in range(1, 5):
            if i in timeline_df['Anomaly Type'].values:
                classes.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=self.class_colors[i], 
                                         markersize=10, label=self.class_names[i]))
                
        plt.legend(handles=classes, loc='upper right')
        
        # Add labels and title
        plt.xlabel('Sequence Index', fontsize=12)
        plt.ylabel('Anomaly Type', fontsize=12)
        plt.title('Anomaly Distribution Across Sequences', fontsize=14)
        
        # Set y-ticks to show class names
        plt.yticks([1, 2, 3, 4], [self.class_names[i] for i in range(1, 5)])
        
        # Add grid
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/class_distribution_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a stacked histogram to show density of anomalies
        plt.figure(figsize=(14, 8))
        
        # Count anomalies per sequence
        anomaly_counts = timeline_df.groupby(['Sequence Index', 'Anomaly Type']).size().unstack().fillna(0)
        
        # Sort by sequence index to ensure proper order
        if not anomaly_counts.empty:
            anomaly_counts = anomaly_counts.sort_index()
            
            # Get colors for each anomaly type
            colors = [self.class_colors[i] for i in anomaly_counts.columns]
            
            # Create histogram
            if 1 in anomaly_counts.columns:
                bottom = None
                for i, col in enumerate(anomaly_counts.columns):
                    if i == 0:
                        plt.bar(anomaly_counts.index, anomaly_counts[col], 
                              label=self.class_names[col], color=self.class_colors[col], alpha=0.8)
                        bottom = anomaly_counts[col].values
                    else:
                        plt.bar(anomaly_counts.index, anomaly_counts[col], 
                              bottom=bottom, label=self.class_names[col], 
                              color=self.class_colors[col], alpha=0.8)
                        bottom = bottom + anomaly_counts[col].values
                
                plt.xlabel('Sequence Index', fontsize=12)
                plt.ylabel('Anomaly Count', fontsize=12)
                plt.title('Anomaly Density Across Sequences', fontsize=14)
                plt.legend(loc='upper right')
                plt.grid(True, alpha=0.3)
                
                # Save figure
                plt.savefig(f"{self.results_dir}/figures/anomaly_density.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                logger.info("Not enough data for stacked histogram")
                plt.close()
        else:
            logger.info("No anomalies found for density visualization")
            plt.close()
        
        logger.info("Class distribution timeline visualizations created")
    
    def plot_feature_distributions(self):
        """Create feature distribution visualizations by class"""
        logger.info("Plotting feature distributions by class...")
        
        # For each feature, create a violin plot or boxplot showing distribution by class
        for i, feature_name in enumerate(self.feature_names):
            plt.figure(figsize=(12, 8))
            
            # Extract feature data for each class
            feature_data = []
            class_data = []
            
            # Flatten the data
            flat_features = self.feature_data.reshape(-1, self.feature_data.shape[-1])
            
            # For each class
            for cls in range(5):
                # Get indices for this class
                indices = np.where(self.labels_flat == cls)[0]
                
                if len(indices) > 0:
                    # Extract feature values
                    cls_feature_values = flat_features[indices, i]
                    
                    # Add to data lists
                    feature_data.extend(cls_feature_values)
                    class_data.extend([self.class_names[cls]] * len(cls_feature_values))
            
            # Create DataFrame
            df = pd.DataFrame({
                'Feature Value': feature_data,
                'Class': class_data
            })
            
            # Create violin plot
            ax = sns.violinplot(x='Class', y='Feature Value', data=df, palette=self.class_colors)
            
            # Add title and labels
            plt.title(f'Distribution of {feature_name} by Class', fontsize=14)
            plt.xlabel('Class', fontsize=12)
            plt.ylabel(f'{feature_name} Value', fontsize=12)
            
            # If feature has many discrete values, show them as boxplot instead
            if len(np.unique(feature_data)) <= 10:
                plt.figure(figsize=(12, 8))
                # Use count plot for categorical features
                ax = sns.countplot(x='Feature Value', hue='Class', data=df, palette=self.class_colors)
                plt.title(f'Distribution of {feature_name} by Class', fontsize=14)
                plt.xlabel(f'{feature_name} Value', fontsize=12)
                plt.ylabel('Count', fontsize=12)
                plt.legend(title='Class')
                plt.tight_layout()
                plt.savefig(f"{self.results_dir}/figures/feature_distribution_{feature_name}_categorical.png", dpi=300, bbox_inches='tight')
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{self.results_dir}/figures/feature_distribution_{feature_name}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Feature distribution visualizations created")
    
    def plot_correlation_heatmap(self):
        """Create correlation heatmap between features"""
        logger.info("Creating correlation heatmap...")
        
        # Calculate correlation for all features
        flat_features = self.feature_data.reshape(-1, self.feature_data.shape[-1])
        
        # Create DataFrame
        feature_df = pd.DataFrame(flat_features, columns=self.feature_names)
        
        # Calculate correlation
        corr_matrix = feature_df.corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                   mask=mask, cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Feature Correlation Heatmap', fontsize=14)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create correlation heatmap for each class separately
        for cls in range(5):
            # Get indices for this class
            indices = np.where(self.labels_flat == cls)[0]
            
            if len(indices) > 100:  # Only if enough samples
                # Extract feature values
                cls_features = flat_features[indices]
                
                # Create DataFrame
                cls_df = pd.DataFrame(cls_features, columns=self.feature_names)
                
                # Calculate correlation
                cls_corr = cls_df.corr()
                
                # Create heatmap
                plt.figure(figsize=(10, 8))
                mask = np.triu(np.ones_like(cls_corr, dtype=bool))  # Mask upper triangle
                sns.heatmap(cls_corr, annot=True, fmt=".2f", cmap="coolwarm",
                           mask=mask, cbar_kws={'label': 'Correlation Coefficient'})
                
                plt.title(f'Feature Correlation Heatmap - {self.class_names[cls]}', fontsize=14)
                plt.tight_layout()
                
                # Save figure
                plt.savefig(f"{self.results_dir}/figures/correlation_heatmap_class_{cls}.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info("Correlation heatmaps created")
    
    def analyze_sequence_length_performance(self):
        """Analyze how model performance varies with sequence length"""
        logger.info("Analyzing performance by sequence length...")
        
        # Group sequences by length
        seq_lengths = [seq_len for _, seq_len in self.seq_indices]
        
        # If sequence length data is not available, skip
        if not seq_lengths:
            logger.info("Sequence length data not available")
            return
        
        # Round to nearest group to have meaningful bins
        length_groups = {}
        for seq_idx, (_, seq_len) in enumerate(self.seq_indices):
            # Round to nearest 10
            group = (seq_len // 10) * 10
            if group not in length_groups:
                length_groups[group] = {
                    'binary_labels': [],
                    'binary_preds': [],
                    'binary_probs': []
                }
            
            # Get predictions for this sequence
            seq_binary_labels = self.binary_labels_flat[seq_idx * self.config['seq_len']:
                                                      (seq_idx+1) * self.config['seq_len']]
            seq_binary_preds = self.binary_preds_flat[seq_idx * self.config['seq_len']:
                                                    (seq_idx+1) * self.config['seq_len']]
            seq_binary_probs = self.binary_probs_flat[seq_idx * self.config['seq_len']:
                                                    (seq_idx+1) * self.config['seq_len']]
            
            # Add to group
            length_groups[group]['binary_labels'].extend(seq_binary_labels)
            length_groups[group]['binary_preds'].extend(seq_binary_preds)
            length_groups[group]['binary_probs'].extend(seq_binary_probs)
        
        # Calculate metrics for each length group
        length_metrics = []
        for group, data in sorted(length_groups.items()):
            if len(data['binary_labels']) > 10:  # Only consider groups with enough samples
                # Calculate metrics
                acc = accuracy_score(data['binary_labels'], data['binary_preds'])
                prec = precision_score(data['binary_labels'], data['binary_preds'], zero_division=0)
                rec = recall_score(data['binary_labels'], data['binary_preds'], zero_division=0)
                f1 = f1_score(data['binary_labels'], data['binary_preds'], zero_division=0)
                
                # Calculate AUC if possible
                try:
                    auc = roc_auc_score(data['binary_labels'], data['binary_probs'])
                except:
                    auc = 0.0
                
                # Calculate count
                count = len(data['binary_labels'])
                
                # Add to metrics list
                length_metrics.append({
                    'Length Group': group,
                    'Count': count,
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1 Score': f1,
                    'AUC': auc
                })
        
        # Create DataFrame
        length_df = pd.DataFrame(length_metrics)
        
        # Save to CSV
        length_df.to_csv(f"{self.results_dir}/statistics/performance_by_length.csv", index=False)
        
        # Create visualization
        plt.figure(figsize=(14, 8))
        
        # Plot metrics vs sequence length
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']:
            plt.plot(length_df['Length Group'], length_df[metric], 'o-', label=metric)
        
        plt.xlabel('Sequence Length', fontsize=12)
        plt.ylabel('Metric Value', fontsize=12)
        plt.title('Performance Metrics by Sequence Length', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/performance_by_length.png", dpi=300, bbox_inches='tight')
        
        # Also create a bar chart showing count by length group
        plt.figure(figsize=(12, 6))
        plt.bar(length_df['Length Group'], length_df['Count'])
        plt.xlabel('Sequence Length', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Sample Count by Sequence Length', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/count_by_length.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Sequence length performance analysis completed")
    
    def plot_probability_distribution(self):
        """Plot probability distribution for each class"""
        logger.info("Plotting probability distributions by class...")
        
        # Bin probabilities for histogram
        bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
        
        # Create figure with multiple histograms
        plt.figure(figsize=(14, 10))
        
        # For each class, plot probability distribution
        for cls in range(5):
            # Get indices for this class
            indices = np.where(self.labels_flat == cls)[0]
            
            if len(indices) > 0:
                # Get probabilities
                probs = self.binary_probs_flat[indices]
                
                # Plot histogram
                plt.hist(probs, bins=bins, alpha=0.6, label=f'{self.class_names[cls]} (n={len(probs)})',
                        color=self.class_colors[cls])
        
        # Add vertical line for threshold
        plt.axvline(x=self.threshold, color='red', linestyle='--', 
                   label=f'Threshold = {self.threshold}')
        
        # Add labels and legend
        plt.xlabel('Anomaly Probability', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Probability Distribution by Class', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/probability_distribution.png", dpi=300, bbox_inches='tight')
        
        # Also create a separate plot for each class
        for cls in range(5):
            # Get indices for this class
            indices = np.where(self.labels_flat == cls)[0]
            
            if len(indices) > 50:  # Only if enough samples
                # Get probabilities
                probs = self.binary_probs_flat[indices]
                
                # Create figure
                plt.figure(figsize=(10, 6))
                
                # Plot histogram
                plt.hist(probs, bins=bins, alpha=0.8, color=self.class_colors[cls])
                
                # Add vertical line for threshold
                plt.axvline(x=self.threshold, color='red', linestyle='--',
                          label=f'Threshold = {self.threshold}')
                
                # Add density curve
                if len(probs) > 100:
                    density = True
                    plt.hist(probs, bins=bins, alpha=0.4, density=density, 
                            color=self.class_colors[cls], histtype='step', 
                            linewidth=2, cumulative=False)
                
                # Calculate and show percentage of samples above threshold
                above_threshold = np.mean(probs >= self.threshold) * 100
                plt.title(f'Probability Distribution - {self.class_names[cls]}\n' +
                        f'{above_threshold:.1f}% samples above threshold', fontsize=14)
                
                plt.xlabel('Anomaly Probability', fontsize=12)
                plt.ylabel('Count', fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save figure
                plt.savefig(f"{self.results_dir}/figures/probability_distribution_class_{cls}.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info("Probability distribution visualizations created")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        logger.info("Generating summary report...")
        
        summary = []
        
        # Header
        summary.append("# Anomaly Detection Model Evaluation Summary")
        summary.append("\n## Model Information")
        summary.append(f"- Model: BiGRUAnomalyDetector")
        summary.append(f"- Parameters: {self.model.count_parameters():,}")
        
        # Dataset statistics
        summary.append("\n## Dataset Statistics")
        
        # Get dataset statistics from test loader
        dataset = self.test_loader.dataset
        total_sequences = len(dataset)
        total_timesteps = sum(len(seq) for seq in dataset.labels)
        
        summary.append(f"- Total sequences: {total_sequences}")
        summary.append(f"- Total timesteps: {total_timesteps}")
        summary.append(f"- Anomaly counts:")
        
        for i in range(5):
            class_count = np.sum(self.labels_flat == i)
            percentage = class_count / len(self.labels_flat) * 100
            summary.append(f"  - {self.class_names[i]}: {class_count} ({percentage:.2f}%)")
        
        # Performance metrics
        summary.append("\n## Performance Metrics")
        summary.append(f"- Accuracy: {self.metrics['accuracy']:.4f}")
        summary.append(f"- Precision: {self.metrics['precision']:.4f}")
        summary.append(f"- Recall: {self.metrics['recall']:.4f}")
        summary.append(f"- F1 Score: {self.metrics['f1']:.4f}")
        summary.append(f"- ROC AUC: {self.metrics['auc']:.4f}")
        
        # Class detection rates
        summary.append("\n## Class Detection Rates")
        for i in range(5):
            metrics = self.class_metrics[i]
            summary.append(f"- {self.class_names[i]}: {metrics['correct']}/{metrics['total']} = {metrics['detection_rate']:.4f}")
        
        # False positive analysis
        summary.append("\n## False Positive Analysis")
        
        # Count false positives
        false_pos_mask = (self.binary_preds_flat == 1) & (self.binary_labels_flat == 0)
        false_pos_count = np.sum(false_pos_mask)
        
        # Count true positives, true negatives, false negatives
        true_pos_mask = (self.binary_preds_flat == 1) & (self.binary_labels_flat == 1)
        true_neg_mask = (self.binary_preds_flat == 0) & (self.binary_labels_flat == 0)
        false_neg_mask = (self.binary_preds_flat == 0) & (self.binary_labels_flat == 1)
        
        true_pos_count = np.sum(true_pos_mask)
        true_neg_count = np.sum(true_neg_mask)
        false_neg_count = np.sum(false_neg_mask)
        
        total_count = len(self.binary_preds_flat)
        
        summary.append(f"- False positives: {false_pos_count} ({false_pos_count/total_count:.2%} of all samples)")
        summary.append(f"- False negatives: {false_neg_count} ({false_neg_count/total_count:.2%} of all samples)")
        summary.append(f"- True positives: {true_pos_count} ({true_pos_count/total_count:.2%} of all samples)")
        summary.append(f"- True negatives: {true_neg_count} ({true_neg_count/total_count:.2%} of all samples)")
        
        # Add clustering insights if available
        if os.path.exists(f"{self.results_dir}/false_positives/cluster_statistics.csv"):
            cluster_stats = pd.read_csv(f"{self.results_dir}/false_positives/cluster_statistics.csv")
            
            summary.append("\n### False Positive Clusters")
            summary.append(f"- Number of clusters: {len(cluster_stats)}")
            
            for _, row in cluster_stats.iterrows():
                summary.append(f"- Cluster {row['Cluster']}: {row['Count']} samples ({row['Percentage']:.2f}%)")
                summary.append(f"  - Top features: {row['Top Features']}")
                summary.append(f"  - Possible anomaly type: {row['Possible Anomaly Type']}")
        
        # Conclusion
        summary.append("\n## Conclusion")
        
        # Calculate total anomaly detection rate
        anomaly_samples = np.sum(self.binary_labels_flat == 1)
        detected_anomalies = np.sum((self.binary_preds_flat == 1) & (self.binary_labels_flat == 1))
        overall_detection_rate = detected_anomalies / anomaly_samples if anomaly_samples > 0 else 0
        
        summary.append(f"The model achieved an overall anomaly detection rate of {overall_detection_rate:.2%}.")
        
        # Best/worst performing anomaly types
        detection_rates = {i: self.class_metrics[i]['detection_rate'] for i in range(1, 5) 
                          if self.class_metrics[i]['total'] > 0}
        
        if detection_rates:
            best_class = max(detection_rates.items(), key=lambda x: x[1])
            worst_class = min(detection_rates.items(), key=lambda x: x[1])
            
            summary.append(f"\nThe model performed best on {self.class_names[best_class[0]]} anomalies with "
                          f"a detection rate of {best_class[1]:.2%}.")
            summary.append(f"The model struggled most with {self.class_names[worst_class[0]]} anomalies with "
                          f"a detection rate of {worst_class[1]:.2%}.")
        
        # Write summary to file
        with open(f"{self.results_dir}/summary_report.md", "w") as f:
            f.write("\n".join(summary))
            
        # Also create a text version
        with open(f"{self.results_dir}/summary_report.txt", "w") as f:
            f.write("\n".join(summary))
            
        logger.info(f"Summary report saved to {self.results_dir}/summary_report.md")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        # Step 1: Analyze dataset statistics
        self.analyze_dataset_statistics()
        
        # Step 2: Run inference
        self.run_inference()
        
        # Step 3: Calculate metrics
        self.calculate_metrics()
        
        # Step 4: Generate basic visualizations
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.plot_class_detection_rates()
        self.plot_confusion_matrix()
        self.plot_binary_confusion_matrix()
        
        # Step 5: Analyze threshold impact
        self.plot_threshold_analysis()
        
        # Step 6: Analyze error by sequence length
        self.analyze_sequence_length_performance()
        
        # Step 7: Analyze class distribution timeline
        self.plot_class_distribution_timeline()
        
        # Step 8: Visualize feature distributions by class
        self.plot_feature_distributions()
        
        # Step 9: Create correlation heatmap
        self.plot_correlation_heatmap()
        
        # Step 10: Plot probability distribution by class
        self.plot_probability_distribution()
        
        # Step 11: Analyze false positives
        self.analyze_false_positives()
        
        # Step 12: Analyze time-series data
        self.analyze_time_series()
        
        # Step 13: Visualize feature importance
        self.visualize_feature_importance()
        
        # Step 14: Generate summary report
        self.generate_summary_report()
        
        # Finish
        logger.info(f"Analysis complete. Results saved to {self.results_dir}")
        return self.metrics


def main():
    """Main entry point for the script"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run inference and analysis on anomaly detection model')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to model weights')
    parser.add_argument('--config', type=str, default=CONFIG_PATH, help='Path to model config')
    parser.add_argument('--data', type=str, default=TEST_DATA_PATH, help='Path to test data')
    parser.add_argument('--results', type=str, default=RESULTS_DIR, help='Path to save results')
    parser.add_argument('--threshold', type=float, default=None, help='Classification threshold (overrides config)')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = AnomalyAnalyzer(
        model_path=args.model,
        config_path=args.config,
        test_data_path=args.data,
        results_dir=args.results
    )
    
    # Override threshold if provided
    if args.threshold is not None:
        analyzer.threshold = args.threshold
        logger.info(f"Overriding threshold with {args.threshold}")
    
    # Run analysis
    metrics = analyzer.run_full_analysis()
    
    return metrics


if __name__ == "__main__":
    main()