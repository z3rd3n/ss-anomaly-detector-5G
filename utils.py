# utils.py
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
        best_f1 = checkpoint.get('best_val_class_acc', 0.0)
        params = checkpoint.get('params', {})
        logging.info(f"Loaded checkpoint '{filename}' (epoch {epoch}) with best validation accuracy: {best_f1:.4f}")
        return epoch, best_f1, params
    else:
        logging.warning(f"No checkpoint found at '{filename}'")
        return 0, 0.0, {}

def compute_features_statistics(data_loader, params, num_features):
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
    for batch in tqdm(data_loader, desc="Computing features statistics", unit="batch"):
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
    std = torch.sqrt(variances + EPS)
    unnorm = normalized_tensor.float() * std + means
    return torch.clamp(torch.round(unnorm), min=0).to(torch.int64)

def produce_anomalies_per_instance(model, val_loader, device):
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc="Computing anomalies", unit="batch") as val_bar:
            for batch in val_loader:
                true_labels = batch['labels'].to(device)
                features = batch['features'].to(device)
                class_logits = model(features)
                all_true_labels.append(true_labels.cpu())
                pred_labels = torch.argmax(class_logits, dim=-1)
                all_pred_labels.append(pred_labels.cpu())
                val_bar.update(1)
    
    all_true_labels = torch.cat(all_true_labels, dim=0)
    all_pred_labels = torch.cat(all_pred_labels, dim=0)
    true_labels_flat = all_true_labels.view(-1)
    pred_labels_flat = all_pred_labels.view(-1)
    return true_labels_flat, pred_labels_flat

def validate_csv(model, params, means, variances, device):
    logging.info("Starting Parquet validation procedure...")

    val_dataset = ParquetSequenceDataset(
        parquet_path=params['validation_parquet_path'],
        feature_columns=params['feature_columns'],
        seq_len=params['seq_len'],
        stride=params['stride'],
        ratio=params['val_ratio'],
        seed=params['seed'],
        skip_anomalies=False,
        normalization_stats={'means': means, 'variances': variances}
    )
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=custom_collate_fn)

    model.eval()
    pred_by_timestamp = {}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", unit="batch"):
            features = batch['features'].to(device)  # [B, T, num_features]
            true_labels = batch['labels']            # [B, T]
            class_logits = model(features)
            pred_labels = torch.argmax(class_logits, dim=-1).cpu()  # [B, T]
            for ts_list, true_seq, pred_seq in zip(batch['timestamps'], true_labels, pred_labels):
                for ts, t_label, p_label in zip(ts_list, true_seq.tolist(), pred_seq.tolist()):
                    pred_by_timestamp[ts] = (t_label, p_label)

    if not pred_by_timestamp:
        logging.warning("No predictions made during validation!")
        return 0.0

    all_true = []
    all_pred = []
    for t_label, p_label in pred_by_timestamp.values():
        all_true.append(t_label)
        all_pred.append(p_label)
    all_true = torch.tensor(all_true)
    all_pred = torch.tensor(all_pred)

    class_labels = torch.unique(all_true)
    class_accuracies = {}
    for label in class_labels:
        mask = (all_true == label)
        total = mask.sum().item()
        correct = (all_true[mask] == all_pred[mask]).sum().item()
        accuracy = correct / total if total > 0 else 0.0
        class_accuracies[label.item()] = (correct, total, accuracy)

    anomaly_mapping = {v: k for k, v in val_dataset.anomaly_mapping.items()}
    for label, (correct, total, accuracy) in class_accuracies.items():
        class_name = anomaly_mapping.get(label, f"Class {label}")
        logging.info(f"{class_name}: {correct}/{total} ({accuracy*100:.2f}%)")

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

    return balanced_acc

def plot_latent_space(model, val_loader, params):
    model.eval()
    all_latents = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            # Model returns (class_logits, latents) when return_latents=True
            _, latents = model(features, return_latents=True)
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
