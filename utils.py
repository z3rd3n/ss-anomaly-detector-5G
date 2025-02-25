import os
import logging
from datetime import datetime
import numpy as np
import torch
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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

def compute_numerical_statistics(dataset, numerical_features, output_file=None):
    """
    Compute statistics for numerical features.
    
    Args:
        dataset: Dataset containing numerical features
        numerical_features: List of numerical feature names
        output_file: Path to save statistics
        
    Returns:
        Dict with 'means' and 'stds' for normalization
    """
    logging.info("Computing statistics for numerical features...")
    
    if hasattr(dataset, 'numerical_features_tensor'):
        # If the dataset already has processed numerical features
        numerical_data = dataset.numerical_features_tensor.numpy()
    else:
        # Extract numerical features from the raw dataset
        df = pd.read_parquet(dataset.parquet_path)
        numerical_data = df[numerical_features].values
        
    # Compute mean and std along sample dimension
    means = np.nanmean(numerical_data, axis=0)
    stds = np.nanstd(numerical_data, axis=0)
    
    # Replace NaN and zero std with small value
    means = np.nan_to_num(means, nan=0.0)
    stds = np.nan_to_num(stds, nan=1.0)
    stds = np.maximum(stds, EPS)
    
    stats = {
        'means': means.tolist(),
        'stds': stds.tolist()
    }
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logging.info(f"Saved numerical statistics to {output_file}")
        
    return stats

def plot_confusion_matrix(cm, class_names, output_dir, filename):
    """
    Plot and save confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_binary_metrics(metrics_history, output_dir, prefix="binary"):
    """
    Plot binary classification metrics over time.
    
    Args:
        metrics_history: List of metric dictionaries
        output_dir: Directory to save plots
        prefix: Prefix for filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract epochs
    epochs = range(1, len(metrics_history) + 1)
    
    # Plot metrics
    metrics_to_plot = {
        'precision': 'Precision',
        'recall': 'Recall/Sensitivity',
        'f1': 'F1 Score',
        'accuracy': 'Accuracy',
        'specificity': 'Specificity',
        'auc': 'ROC AUC'
    }
    
    for metric, title in metrics_to_plot.items():
        if any(metric in m for m in metrics_history):
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, [m.get(metric, 0) for m in metrics_history], 'b-')
            plt.xlabel('Epochs')
            plt.ylabel(title)
            plt.title(f'{title} over Training')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'{prefix}_{metric}.png'))
            plt.close()

def plot_anomaly_type_detection(breakdown, output_dir, filename="anomaly_detection_by_type.png"):
    """
    Plot detection rate for each anomaly type.
    
    Args:
        breakdown: Dictionary mapping anomaly names to detection statistics
        output_dir: Directory to save the plot
        filename: Name of the output file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame from breakdown
    df = pd.DataFrame([
        {
            'anomaly_type': anomaly_type,
            'detection_rate': stats['detection_rate'],
            'count': stats['total_samples']
        }
        for anomaly_type, stats in breakdown.items()
    ])
    
    # Sort by anomaly type with "normal" first
    df['sort_order'] = df['anomaly_type'].apply(lambda x: 0 if x == 'normal' else 1)
    df = df.sort_values(['sort_order', 'anomaly_type']).reset_index(drop=True)
    
    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df['anomaly_type'], df['detection_rate'])
    
    # Add count labels on top of bars
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{df['count'].iloc[i]}",
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Anomaly Type')
    plt.ylabel('Detection Rate')
    plt.title('Detection Rate by Anomaly Type')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def analyze_sequence_patterns(dataset, predictions_df, output_dir, max_samples=10):
    """
    Analyze and visualize sequence patterns for correct and incorrect predictions.
    
    Args:
        dataset: The dataset with sequences
        predictions_df: DataFrame with predictions
        output_dir: Directory to save analysis
        max_samples: Maximum number of samples to analyze per category
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not hasattr(dataset, 'sequences') or 'original_label' not in predictions_df.columns:
        logging.warning("Cannot analyze sequence patterns: missing required data")
        return
    
    # Categories to analyze
    categories = [
        ('true_positives', (predictions_df['is_anomaly'] == 1) & (predictions_df['original_label'] != 0)),
        ('false_positives', (predictions_df['is_anomaly'] == 1) & (predictions_df['original_label'] == 0)),
        ('false_negatives', (predictions_df['is_anomaly'] == 0) & (predictions_df['original_label'] != 0)),
        ('true_negatives', (predictions_df['is_anomaly'] == 0) & (predictions_df['original_label'] == 0))
    ]
    
    results = {}
    
    for category_name, mask in categories:
        if not np.any(mask):
            continue
            
        # Sample sequences
        category_indices = np.where(mask)[0][:max_samples]
        
        # Extract sequence data
        category_data = []
        for idx in category_indices:
            if idx >= len(dataset):
                continue
                
            sample = dataset[idx]
            seq_data = {
                'harq_ids': sample['harq_ids'].numpy(),
                'original_labels': sample['original_labels'].numpy() if 'original_labels' in sample else None,
                'binary_labels': sample['binary_labels'].numpy(),
                'numerical_features': {
                    feature: sample['numerical_features'][:, i].numpy()
                    for i, feature in enumerate(dataset.numerical_features)
                },
                'categorical_features': {
                    feature: sample['categorical_features'][feature].numpy()
                    for feature in sample['categorical_features']
                }
            }
            category_data.append(seq_data)
            
        results[category_name] = category_data
    
    # Save as JSON (after converting numpy arrays to lists)
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    with open(os.path.join(output_dir, 'sequence_analysis.json'), 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    
    # Plot some patterns for visualization
    for category_name, data in results.items():
        if not data:
            continue
            
        # Plot HARQ patterns for a few sequences
        for i, seq in enumerate(data[:min(3, len(data))]):
            plt.figure(figsize=(12, 8))
            
            # Plot categorical features
            plt.subplot(3, 1, 1)
            for feature, values in seq['categorical_features'].items():
                plt.plot(values, label=feature)
            plt.xlabel('Sequence Step')
            plt.ylabel('Value')
            plt.title(f'{category_name} - Categorical Features (Sequence {i+1})')
            plt.legend()
            plt.grid(True)
            
            # Plot ReTx and CRC specifically
            plt.subplot(3, 1, 2)
            plt.plot(seq['categorical_features'].get('CRC', []), label='CRC')
            plt.plot(seq['numerical_features'].get('ReTx', []), label='ReTx')
            plt.plot(seq['categorical_features'].get('NDI', []), label='NDI')
            plt.xlabel('Sequence Step')
            plt.ylabel('Value')
            plt.title(f'{category_name} - Key Features')
            plt.legend()
            plt.grid(True)
            
            # Plot binary and original labels
            plt.subplot(3, 1, 3)
            plt.plot(seq['binary_labels'], label='Binary Labels')
            if seq['original_labels'] is not None:
                plt.plot(seq['original_labels'], label='Original Labels')
            plt.xlabel('Sequence Step')
            plt.ylabel('Label')
            plt.title(f'{category_name} - Labels')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{category_name}_sequence_{i+1}.png'))
            plt.close()

def extract_anomaly_patterns(dataset, predictions_df, output_dir):
    """
    Extract and analyze patterns for each anomaly type.
    
    Args:
        dataset: The dataset with sequences
        predictions_df: DataFrame with predictions
        output_dir: Directory to save analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if 'anomaly_type' not in predictions_df.columns:
        logging.warning("Cannot extract anomaly patterns: missing anomaly_type column")
        return
    
    # Group by anomaly type
    anomaly_types = predictions_df['anomaly_type'].unique()
    
    # Count correct/incorrect predictions by type
    type_stats = {}
    for anomaly_type in anomaly_types:
        mask = predictions_df['anomaly_type'] == anomaly_type
        if anomaly_type == 'normal':
            correct = ((predictions_df['is_anomaly'] == 0) & mask).sum()
        else:
            correct = ((predictions_df['is_anomaly'] == 1) & mask).sum()
            
        total = mask.sum()
        type_stats[anomaly_type] = {
            'total': total,
            'correct': correct,
            'accuracy': correct / total if total > 0 else 0
        }
    
    # Save statistics
    stats_df = pd.DataFrame([
        {
            'anomaly_type': anomaly_type,
            'total_samples': stats['total'],
            'correct_predictions': stats['correct'],
            'accuracy': stats['accuracy']
        }
        for anomaly_type, stats in type_stats.items()
    ])
    
    stats_df.to_csv(os.path.join(output_dir, 'anomaly_type_stats.csv'), index=False)
    
    # Plot accuracy by type
    plt.figure(figsize=(12, 6))
    bars = plt.bar(stats_df['anomaly_type'], stats_df['accuracy'])
    
    # Add count labels on top of bars
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{stats_df['total_samples'].iloc[i]}",
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Anomaly Type')
    plt.ylabel('Accuracy')
    plt.title('Prediction Accuracy by Anomaly Type')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'anomaly_type_accuracy.png'))
    plt.close()
    
    # For each anomaly type, create a file with detailed samples
    for anomaly_type in anomaly_types:
        if anomaly_type == 'normal':
            continue  # Skip normal samples for brevity
            
        # Extract samples of this type
        type_samples = predictions_df[predictions_df['anomaly_type'] == anomaly_type]
        
        # Save to CSV
        type_samples.to_csv(
            os.path.join(output_dir, f'samples_{anomaly_type}.csv'),
            index=False
        )
        
        # Add some analysis for common features of this anomaly type
        if hasattr(dataset, 'df'):
            # Find all samples of this type in the original dataset
            if anomaly_type == 'normal':
                df_samples = dataset.df[dataset.df['insight'].isna() | (dataset.df['insight'] == '')]
            else:
                df_samples = dataset.df[dataset.df['insight'].str.contains(anomaly_type, na=False)]
                
            if len(df_samples) > 0:
                # Calculate statistics
                stats = {}
                for feature in dataset.feature_columns:
                    if feature in df_samples.columns:
                        if df_samples[feature].dtype in [np.int64, np.float64]:
                            stats[feature] = {
                                'mean': df_samples[feature].mean(),
                                'std': df_samples[feature].std(),
                                'min': df_samples[feature].min(),
                                'max': df_samples[feature].max()
                            }
                
                # Save statistics
                with open(os.path.join(output_dir, f'stats_{anomaly_type}.json'), 'w') as f:
                    json.dump(stats, f, indent=2)

def find_undetected_anomalies(predictions_df, output_dir):
    """
    Analyze anomalies that the model failed to detect.
    
    Args:
        predictions_df: DataFrame with predictions
        output_dir: Directory to save analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if 'original_label' not in predictions_df.columns or 'anomaly_type' not in predictions_df.columns:
        logging.warning("Cannot analyze undetected anomalies: missing required columns")
        return
    
    # Find undetected anomalies (false negatives)
    undetected = predictions_df[
        (predictions_df['is_anomaly'] == 0) & 
        (predictions_df['original_label'] != 0)
    ]
    
    if len(undetected) == 0:
        logging.info("No undetected anomalies found")
        return
    
    logging.info(f"Found {len(undetected)} undetected anomalies")
    
    # Save to CSV
    undetected.to_csv(os.path.join(output_dir, 'undetected_anomalies.csv'), index=False)
    
    # Count by type
    type_counts = undetected['anomaly_type'].value_counts()
    
    # Plot
    plt.figure(figsize=(10, 6))
    type_counts.plot(kind='bar')
    plt.xlabel('Anomaly Type')
    plt.ylabel('Count')
    plt.title('Undetected Anomalies by Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'undetected_anomalies_by_type.png'))
    plt.close()
    
    # For each type, save some examples
    for anomaly_type, count in type_counts.items():
        type_samples = undetected[undetected['anomaly_type'] == anomaly_type]
        type_samples.to_csv(
            os.path.join(output_dir, f'undetected_{anomaly_type}.csv'),
            index=False
        )


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