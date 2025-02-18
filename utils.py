import os
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import ParquetSequenceDataset, custom_collate_fn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.decomposition import PCA
import torch.nn.functional as F

EPS = 1e-8

def start_logging(params=None):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = params['output_dir'] if params and 'output_dir' in params else 'output'
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'logTraining_{current_time}.log')
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
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    logging.info(f"Checkpoint saved to {filename}")

def load_checkpoint(filename: str, model: nn.Module, optimizer: torch.optim.Optimizer):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        best_f1 = checkpoint.get('best_f1', 0.0)
        params = checkpoint.get('params', {})
        logging.info(f"Loaded checkpoint '{filename}' (epoch {epoch}) with best F1: {best_f1:.4f}")
        return epoch, best_f1, params
    else:
        logging.warning(f"No checkpoint found at '{filename}'")
        return 0, 0.0, {}
    


def compute_features_statistics(train_loader, params, num_features):
    """
    Computes per-feature mean and variance from the training data.
    The computed statistics are saved to a JSON file specified in params['features_stats_json'].
    If the JSON file exists, load and return the statistics.
    
    Returns:
        means: torch.FloatTensor of shape [num_features]
        variances: torch.FloatTensor of shape [num_features]
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
    total_sum = torch.zeros(num_features, dtype=torch.float64)
    total_sum_sq = torch.zeros(num_features, dtype=torch.float64)
    count = 0
    for batch in tqdm(train_loader, desc="Computing features statistics", unit="batch"):
        # features: [B, T, num_features]
        features = batch['features'].to(torch.float64)
        B, T, F = features.shape
        count += B * T
        total_sum += features.sum(dim=(0,1))
        total_sum_sq += (features**2).sum(dim=(0,1))
    means = (total_sum / count).to(torch.float32)
    variances = ((total_sum_sq / count) - (means.double()**2)).to(torch.float32)
    stats = {'means': means.tolist(), 'variances': variances.tolist()}
    with open(stats_file, 'w') as f:
        json.dump(stats, f)
    logging.info(f"Features statistics saved to {stats_file}")
    return means, variances

def unnormalize_features(normalized_tensor, means, variances):
    """
    Unnormalizes a tensor that was normalized using (x - mean)/sqrt(var+EPS).
    Returns the unnormalized tensor cast to the nearest integer.
    """
    std = torch.sqrt(variances + EPS)
    unnorm = normalized_tensor.float() * std + means
    return torch.clamp(torch.round(unnorm), min=0).to(torch.int64)

def compute_anomaly_scores(model, batch, means, variances, ignore_indices={0, 1, 2}):
    """
    Given a batch (with key 'features') and the trained model,
    returns a tensor of anomaly scores of shape [B, T] and per-feature errors.
    """
    features = batch['features'].to(next(model.parameters()).device)  # [B, T, num_features]
    outputs = model(features)  # List of outputs; each: [B, T, 1]
    batch_size, seq_len, num_features = features.shape
    
    scores = torch.zeros(batch_size, seq_len, device=features.device)
    per_feature_errors = torch.zeros(batch_size, seq_len, num_features, device=features.device)
    
    # Compute variance for normalization (with EPS to avoid division by zero)
    var = variances.to(features.device) + EPS  # shape: [num_features]
    feature_weights = torch.tensor([1.0, 1.0, 1.0, 0.5, 1.5, 1.5, 1.0], device=features.device)
    
    for i in range(num_features):
        if i not in ignore_indices:
            target = features[:, :, i:i+1]
            pred = outputs[i]
            # Compute weighted normalized MSE error per feature
            error = feature_weights[i] * ((pred - target) ** 2) / var[i]
            scores += error.squeeze(-1)
            per_feature_errors[:, :, i] = error.squeeze(-1)
    
    return scores, per_feature_errors

def produce_anomalies_per_feature(model, val_loader, params, means, variances, ignore_indices={0, 1, 2}):
    """
    Optimized version that uses compute_anomaly_scores and calculates per-feature thresholds correctly.
    """
    model.eval()
    device = next(model.parameters()).device

    # Pre-allocate lists for batch processing
    all_timestamps = []
    all_true_features = []
    all_pred_features = []
    all_feature_errors = []
    all_per_feature_errors = []
    
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc="Computing anomalies", unit="batch") as val_bar:
            for batch in val_loader:
                features = batch['features'].to(device)
                
                # Compute anomaly scores and per-feature errors
                scores, per_feature_errors = compute_anomaly_scores(model, batch, means, variances, ignore_indices)
                
                # Get model predictions for all features at once
                outputs = model(features)
                pred_tensor = torch.cat(outputs, dim=-1)
                
                # Unnormalize features (do this on GPU)
                unnorm_pred = unnormalize_features(pred_tensor, means.to(device), variances.to(device))
                unnorm_true = unnormalize_features(features, means.to(device), variances.to(device))
                
                # Store results
                all_timestamps.extend(batch['timestamps'])
                all_true_features.append(unnorm_true.cpu())
                all_pred_features.append(unnorm_pred.cpu())
                all_feature_errors.append(scores.cpu())
                all_per_feature_errors.append(per_feature_errors.cpu())
                
                val_bar.update(1)
    
    # Concatenate all results
    all_true_features = torch.cat(all_true_features, dim=0)
    all_pred_features = torch.cat(all_pred_features, dim=0)
    all_feature_errors = torch.cat(all_feature_errors, dim=0)
    all_per_feature_errors = torch.cat(all_per_feature_errors, dim=0)
    
    # Calculate per-feature thresholds
    feature_names = [name for i, name in enumerate(params['feature_columns']) 
                    if i not in ignore_indices]
    thresholds = {}
    
    # Convert to numpy for percentile calculation
    per_feature_errors_np = all_per_feature_errors.numpy()
    
    # Calculate threshold for each feature separately
    for i, feature_name in enumerate(params['feature_columns']):
        if i not in ignore_indices:
            feature_errors = per_feature_errors_np[:, :, i].flatten()
            thresholds[feature_name] = np.percentile(feature_errors, params['percentile'])
    
    # Create anomalies DataFrame
    anomalies = []
    for idx in range(len(all_timestamps)):
        for t in range(all_true_features.shape[1]):
            feature_errors = {
                feature_name: all_per_feature_errors[idx, t, i].item()
                for i, feature_name in enumerate(params['feature_columns'])
                if i not in ignore_indices
            }
            
            # Check if any feature exceeds its threshold
            reasons = [name for name, error in feature_errors.items() 
                      if error > thresholds[name]]
            
            if reasons:
                anomaly = {
                    'timestamp': str(all_timestamps[idx][t]),
                    'true_features': all_true_features[idx, t].tolist(),
                    'predicted_features': all_pred_features[idx, t].tolist(),
                    'anomaly_score': all_feature_errors[idx, t].item(),
                    'feature_errors': feature_errors,
                    'anomaly_reason': ",".join(reasons)
                }
                anomalies.append(anomaly)
    
    # Create and sort DataFrame
    anomalies_df = pd.DataFrame(anomalies)
    if not anomalies_df.empty:
        anomalies_df = anomalies_df.sort_values(by='anomaly_score', ascending=False)
        anomalies_df = anomalies_df.drop_duplicates(subset='timestamp')
    
    return anomalies_df, thresholds



def validate_csv(model, params, means, variances):
    """
    Validates the model on validation data (now from a parquet file) by computing anomaly scores
    and comparing against ground truth extracted from the parquet.
    
    Ground truth is built by taking only those rows that have a valid 'insight' column,
    keeping only the 'timestamp_str' and 'insight' columns.
    """
    logging.info("Starting Parquet validation procedure...")

    # Create the validation dataset from the parquet file.
    val_dataset = ParquetSequenceDataset(
        parquet_path=params['validation_parquet_path'],  # New validation parquet file
        feature_columns=params['feature_columns'],
        seq_len=params['seq_len'],
        stride=params['stride'],
        split='train',
        validation_ratio=params['val_ratio'],
        seed=params['seed'],
        skip_anomalies=False,
        normalization_stats={'means': means, 'variances': variances}
    )
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=custom_collate_fn)

    # Compute anomalies using the existing procedure.
    anomalies_df, thresholds = produce_anomalies_per_feature(model, val_loader, params, means, variances)
    logging.info(f"Per-feature thresholds: {thresholds}")
    if anomalies_df.empty:
        logging.warning("No anomalies detected during validation!")
        return 0.0, 0.0, 0.0

    # Ensure anomaly timestamps are strings.
    anomalies_df["timestamp"] = anomalies_df["timestamp"].astype(str)

    # Load the ground truth parquet.
    df_gt = pd.read_parquet(params['validation_parquet_path'], columns=['timestamp_str', 'insight'])
    # Ensure both required columns exist.
    if "timestamp_str" not in df_gt.columns or "insight" not in df_gt.columns:
        raise ValueError("Expected both 'timestamp_str' and 'insight' columns in the validation parquet file!")
    
    # Filter to only take rows where 'insight' is not null, then keep only the two columns.
    df_gt_filtered = df_gt[df_gt['insight'].notna()][["timestamp_str", "insight"]]
    # Convert timestamps to strings.
    df_gt_filtered["timestamp_str"] = df_gt_filtered["timestamp_str"].astype(str)
    # Create a set of ground truth timestamps.
    gt_timestamps = set(df_gt_filtered["timestamp_str"].unique())

    # Mark anomalies as accurate if their timestamp is in the ground truth.
    anomalies_df["is_accurate"] = anomalies_df["timestamp"].apply(lambda ts: ts in gt_timestamps)

    # Compute precision, recall, and F1-score.
    num_accurate = anomalies_df["is_accurate"].sum()
    total_anoms = len(anomalies_df)
    total_gt = len(gt_timestamps)
    precision = (num_accurate / total_anoms) if total_anoms > 0 else 0.0
    recall = (num_accurate / total_gt) if total_gt > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    logging.info(f"Total anomalies reported by model: {total_anoms}")
    logging.info(f"Total ground truth anomalies (with insight): {total_gt}")
    logging.info(f"Total accurate detections: {num_accurate}")
    logging.info(f"[RESULT] Validation - Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1-Score: {f1_score*100:.2f}%")

    anomalies_df = anomalies_df.sort_values(by='anomaly_score', ascending=False)
    anomalies_csv_path = os.path.join(params['output_dir'], 'sorted_anomalies.csv')
    anomalies_df.to_csv(anomalies_csv_path, index=False)
    logging.info(f"Sorted anomalies saved to {anomalies_csv_path}")

    # Generate CSV of ground truth anomalies missed by the model.
    detected_gt_timestamps = set(anomalies_df[anomalies_df["is_accurate"]]["timestamp"])
    missed_gt_timestamps = gt_timestamps - detected_gt_timestamps
    missed_df = df_gt_filtered[df_gt_filtered["timestamp_str"].isin(missed_gt_timestamps)]
    missed_csv_path = os.path.join(params['output_dir'], 'missed_ground_truth_anomalies.csv')
    missed_df.to_csv(missed_csv_path, index=False)
    logging.info(f"Missed ground truth anomalies saved to {missed_csv_path}")

    if not df_gt_filtered.empty:
        detected_gt_df = anomalies_df[anomalies_df["is_accurate"]].merge(
            df_gt_filtered, left_on='timestamp', right_on='timestamp_str', how='left'
        )
        detection_counts = detected_gt_df['insight'].value_counts().sort_index()
        total_counts = df_gt_filtered['insight'].value_counts().sort_index()
        detection_rate = (detection_counts / total_counts * 100).fillna(0)
        fig, ax = plt.subplots(figsize=(10, 6))
        detection_rate.plot(kind='bar', ax=ax)
        ax.set_xlabel("Anomaly Type")
        ax.set_ylabel("Detection Rate (%)")
        ax.set_title("Detection Rate by Anomaly Type")
        plt.tight_layout()
        detection_rate_plot_path = os.path.join(params['output_dir'], 'anomaly_detection_rates.png')
        plt.savefig(detection_rate_plot_path)
        plt.close()
        logging.info(f"Anomaly detection rates plot saved to {detection_rate_plot_path}")
    else:
        logging.warning("No valid ground truth rows with 'insight' available; skipping detection rate plot.")

    # (b) Histogram of anomaly scores.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(anomalies_df['anomaly_score'], bins=50, color='skyblue', edgecolor='black')
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Anomaly Scores")
    histogram_plot_path = os.path.join(params['output_dir'], 'anomaly_score_histogram.png')
    plt.tight_layout()
    plt.savefig(histogram_plot_path)
    plt.close()
    logging.info(f"Anomaly score histogram saved to {histogram_plot_path}")

    # (c) Boxplot of anomaly scores by anomaly type (if available).
    if not df_gt_filtered.empty:
        detected_gt_df = anomalies_df[anomalies_df["is_accurate"]].merge(
            df_gt_filtered, left_on='timestamp', right_on='timestamp_str', how='left'
        )
        if not detected_gt_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            detected_gt_df.boxplot(column='anomaly_score', by='insight', ax=ax)
            ax.set_xlabel("Anomaly Type")
            ax.set_ylabel("Anomaly Score")
            ax.set_title("Anomaly Score Distribution by Anomaly Type")
            plt.suptitle("")
            boxplot_path = os.path.join(params['output_dir'], 'anomaly_score_boxplot_by_insight.png')
            plt.tight_layout()
            plt.savefig(boxplot_path)
            plt.close()
            logging.info(f"Anomaly score boxplot by insight saved to {boxplot_path}")
    # -------------------------------------------------------------------

    return precision, recall, f1_score

def plot_latent_space(model, val_loader, params):
    """
    Extracts latent representations from the validation loader and plots them
    using PCA. Supports 2D or 3D plotting depending on the number of components.
    """
    model.eval()
    all_latents = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            # If the model supports returning latents, e.g., via an optional flag
            outputs, latents = model(features, return_latents=True)
            all_latents.append(latents.cpu().numpy())
    if not all_latents:
        logging.warning("No latent representations found for plotting.")
        return
    all_latents = np.concatenate(all_latents, axis=0)
    n_components = params.get('pca_n_components', 3)
    pca = PCA(n_components=n_components)
    latent_pca = pca.fit_transform(all_latents)
    explained_variance = pca.explained_variance_ratio_
    
    if n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(latent_pca[:, 0], latent_pca[:, 1], latent_pca[:, 2],
                        c=latent_pca[:, 0], cmap='viridis', s=3)
        ax.set_title("Latent Space PCA (3 Components)")
        ax.set_xlabel(f"PC1 ({explained_variance[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({explained_variance[1]*100:.1f}% var)")
        ax.set_zlabel(f"PC3 ({explained_variance[2]*100:.1f}% var)")
    elif n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(latent_pca[:, 0], latent_pca[:, 1],
                        c=latent_pca[:, 0], cmap='viridis', s=3)
        ax.set_title("Latent Space PCA (2 Components)")
        ax.set_xlabel(f"PC1 ({explained_variance[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({explained_variance[1]*100:.1f}% var)")
        plt.colorbar(sc, ax=ax)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(np.arange(latent_pca.shape[0]), latent_pca[:, 0], c='blue', s=3)
        ax.set_xlabel("Sample index")
        ax.set_ylabel(f"PC1 ({explained_variance[0]*100:.1f}% var)")
        ax.set_title("Latent Space PCA (1 Component)")
    
    plt.savefig(os.path.join(params['output_dir'], 'latent_space_plots.png'))
    plt.close()
    logging.info("Saved latent space plots (PCA).")


def log_model_size(model: torch.nn.Module, device: torch.device = None) -> None:
    """
    Logs the model's parameter counts and estimated memory footprint.

    Args:
        model (torch.nn.Module): The model to analyze.
        device (torch.device, optional): If provided, moves the model to this device.
    """
    if device is not None:
        model.to(device)
    
    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Log parameter counts (using appropriate units)
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

    # Estimate memory footprint
    # Assuming each parameter is a 32-bit float (4 bytes)
    bytes_per_param = 4
    total_bytes = total_params * bytes_per_param
    size_mb = total_bytes / (1024**2)
    logging.info(f"Approximate model size: {size_mb:.2f} MB (assuming fp32)")
