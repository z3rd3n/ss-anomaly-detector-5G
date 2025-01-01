# utils.py
import logging
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from scipy import stats
import pandas as pd


def start_logging(params=None, approach=None):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if approach is not None:
        output_dir = os.path.join(approach, 'results')
    if params is None and approach is None:
        output_dir = 'results'
    else:
        experiment_name = f"T{1 if params.train else 0}_D{1 if params.detect else 0}_q{str(params.q)[-2:]}p{params.p}_s{params.seq_len}_h{params.n_heads}_e{params.e_layers}_d{params.model_dim}"
        output_dir = os.path.join(output_dir, experiment_name)
        params.output_dir = output_dir


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

    if params is not None:
        logging.info("Hyperparameters and settings:")
        for key, value in vars(params).items():
            logging.info(f"{key}: {value}")


def save_checkpoint(model, optimizer, epoch, loss, params):
    checkpoint_dir = os.path.join(params.output_dir, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_best.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    logging.info(f"Checkpoint saved: {checkpoint_path}")

def load_last_checkpoint(params, model, optimizer):
    checkpoint_dir = os.path.join(params.output_dir, 'checkpoints')
    result_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    if not result_files:
        logging.info(f"No checkpoints found in {checkpoint_dir}, aborting detection.")
        return

    def get_epoch(fname):
        return int(fname.split('_')[-1].replace('.pt',''))
    result_files_sorted = sorted(result_files, key=lambda x: get_epoch(x))
    last_ckpt = os.path.join(checkpoint_dir, result_files_sorted[-1])
    logging.info(f"Loading last checkpoint: {last_ckpt}")

    epoch, loss = load_checkpoint(model, optimizer, last_ckpt)
    logging.info(f"Loaded last checkpoint from epoch {epoch+1} with loss {loss:.4f}")


def load_checkpoint(model, optimizer, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    logging.info(f"Loaded checkpoint from epoch {epoch+1} with loss {loss:.4f}")
    return epoch, loss


def plot_training_curves(train_losses, val_losses, output_dir):
    """
    Saves a PNG plot of train vs validation loss over epochs.
    """
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Val Loss')
    ax.set_title('Training & Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.tight_layout()
    save_path = os.path.join(output_dir, 'train_val_loss_curve.png')
    plt.savefig(save_path)
    plt.close(fig)
    logging.info(f"Saved train/val loss curve to {save_path}")



def plot_attention_matrices(
    model, 
    dataloader, 
    device, 
    output_dir, 
    max_plots=5,   # how many batches you want to visualize
    max_heads=8,    # how many heads per batch you want to plot
    sample_idx=16    # which sample in the batch to visualize
):
    """
    Plots multi-head attention from the last encoder layer.
    
    Args:
        model: your model with a forward() returning queries_list, keys_list.
        dataloader: the DataLoader providing (features, etc).
        device: the device ('cpu' or 'cuda') to use.
        output_dir: where to save the generated plots.
        max_plots: number of batches to plot from the dataloader.
        max_heads: how many heads to visualize per batch.
        sample_idx: which sample in the batch we want to plot.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_plots:
            break  # stop if we already plotted enough

        features = batch['features'].to(device)  # shape [B, L, D]
        with torch.no_grad():
            # The model returns something like enc_out, [queries_per_layer], [keys_per_layer]
            _, queries_list, keys_list = model(features)

        queries = torch.stack(queries_list, dim=0).mean(dim=0) # shape [B, L, H, d_k]
        keys = torch.stack(keys_list, dim=0).mean(dim=0) # shape [B, L, H, d_k]
        
        # We'll visualize a single sample in the batch: sample_idx
        # queries_0 shape => [L, H, d_k]
        queries_0 = queries[sample_idx]  # shape [L, H, d_k]
        keys_0 = keys[sample_idx]        # shape [L, H, d_k]

        # Permute so each head is first: [H, L, d_k]
        queries_0 = queries_0.permute(1, 0, 2)  # => [H, L, d_k]
        keys_0 = keys_0.permute(1, 0, 2)        # => [H, L, d_k]

        # Now compute attention for each head: 
        # attention = Q * K^T => shape [H, L, L]
        # Usually we do scale = 1 / sqrt(d_k).
        d_k = queries_0.size(-1)
        scale = 1.0 / (d_k**0.5)
        attn_matrices = torch.bmm(queries_0, keys_0.transpose(1, 2))  # => [H, L, L]
        attn_matrices = attn_matrices * scale
        attn_matrices = torch.softmax(attn_matrices, dim=-1)          # => [H, L, L]

        # Plot up to max_heads heads
        num_heads = min(attn_matrices.size(0), max_heads)
        num_cols = 4
        num_rows = (num_heads + num_cols - 1) // num_cols  # ceiling division
        fig, axes = plt.subplots(
            nrows=num_rows,
            ncols=num_cols,
            figsize=(4*num_cols, 4*num_rows),  # wide enough for each head
            squeeze=False
        )

        for h in range(num_heads):
            attn_head_h = attn_matrices[h]  # shape [L, L]
            row = h // num_cols
            col = h % num_cols
            ax = axes[row, col]
            im = ax.imshow(
            attn_head_h.cpu().numpy(),
            cmap='hot',
            interpolation='nearest',
            aspect='auto'
            )
            ax.set_title(f"Batch {batch_idx}, Sample {sample_idx}, Head {h}")
            ax.set_xlabel("Key positions")
            ax.set_ylabel("Query positions")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide any unused subplots
        for h in range(num_heads, num_rows * num_cols):
            fig.delaxes(axes.flatten()[h])

        fig.tight_layout()
        save_path = os.path.join(output_dir, f"attention_batch{batch_idx}_sample{sample_idx}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logging.info(f"Saved attention matrix to {save_path}")


def plot_anomalies(all_scores, anomalies_mask, threshold, output_dir):
    """
    Plots the anomaly score over sample index, highlighting anomalies above threshold.
    """
    fig, ax = plt.subplots(figsize=(6,4))
    idx = np.arange(len(all_scores))
    ax.plot(idx, all_scores, label='Anomaly Score')
    ax.axhline(threshold, color='r', linestyle='--', label=f'Threshold={threshold:.2f}')
    # highlight anomalies
    ax.scatter(idx[anomalies_mask], all_scores[anomalies_mask], color='red', s=10, label='Detected Anomalies')
    ax.set_title('Anomaly Scores (Validation Set)')
    ax.set_xlabel('Index')
    ax.set_ylabel('Score')
    ax.legend()
    fig.tight_layout()
    save_path = os.path.join(output_dir, 'anomaly_scores.png')
    plt.savefig(save_path)
    plt.close(fig)
    logging.info(f"Saved anomaly score plot to {save_path}")

def unscale_features(features):
    scaler_json_path = "data/scaling_params.json"
    # 1) Load scaling parameters
    if not os.path.exists(scaler_json_path):
        logging.warning("Scaling files not found; will save scaled features as-is.")
        unscaled = features
    else:
        with open(scaler_json_path, 'r') as f:
            scaling_params = json.load(f)
            
        # Create numpy arrays from the parameters    
        mean = np.array(scaling_params['mean_'])
        scale = np.array(scaling_params['scale_'])
        
        # Perform inverse transform manually: X_orig = X_scaled * scale + mean
        unscaled = features * scale + mean

    # 2) Round to integers
    unscaled = np.rint(unscaled).astype(int)
    return unscaled

def unscale_and_save_anomalies(
    timestamps,
    features,
    enc_outputs,
    anomaly_scores,
    threshold,
    output_csv
):
  
    unscaled = unscale_features(features)
    predicted = unscale_features(enc_outputs)

    anomaly_indices = np.where(anomaly_scores >= threshold)[0]
    logging.info(f"Total anomaly count: {len(anomaly_indices)} from {len(anomaly_scores)} ratio {len(anomaly_indices)/len(anomaly_scores)} (threshold={threshold})")

    if len(anomaly_indices) == 0:
        logging.info(f"No anomalies found above threshold = {threshold}. No CSV created.")
        return
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    if len(anomaly_indices) > 0:
        anomalies = pd.DataFrame({
            'timestamp_str': [timestamps[i] for i in anomaly_indices],
            'SFN': unscaled[anomaly_indices, 0].astype(int),
            'SFN_p': predicted[anomaly_indices, 0].astype(int),
            'Slot': unscaled[anomaly_indices, 1].astype(int),
            'Slot_p': predicted[anomaly_indices, 1].astype(int),
            'CC': unscaled[anomaly_indices, 2].astype(int),
            'CC_p': predicted[anomaly_indices, 2].astype(int),
            'HARQ': unscaled[anomaly_indices, 3].astype(int),
            'HARQ_p': predicted[anomaly_indices, 3].astype(int),
            'MCS': unscaled[anomaly_indices, 4].astype(int),
            'MCS_p': predicted[anomaly_indices, 4].astype(int),
            'CRC': unscaled[anomaly_indices, 5].astype(int),
            'CRC_p': predicted[anomaly_indices, 5].astype(int),
            'ReTx': unscaled[anomaly_indices, 6].astype(int),
            'ReTx_p': predicted[anomaly_indices, 6].astype(int),
            'NDI': unscaled[anomaly_indices, 7].astype(int),
            'NDI_p': predicted[anomaly_indices, 7].astype(int),
            'threshold': threshold,
            'anomaly_score': anomaly_scores[anomaly_indices],  # Add anomaly scores,
            'distance_from_threshold': (anomaly_scores[anomaly_indices] - threshold)
        })
        
        # Sort by anomaly score in descending order
        anomalies = anomalies.sort_values('anomaly_score', ascending=False)
        anomalies.to_csv(output_csv, index=False)

    logging.info(f"Anomalies saved to CSV => {output_csv}")

    # Create histogram of anomaly scores
    plt.figure(figsize=(10, 6))
    plt.hist(anomaly_scores, bins=50, edgecolor='black')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})')
    plt.title('Distribution of Anomaly Scores')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('subAdjacent/results/anomaly_scores_histogram.png')
    plt.close()


def calculate_threshold_evt(scores, q=0.99, p=95):
    # Fit generalized Pareto distribution
    tail_scores = scores[scores > np.percentile(scores, p)]
    shape, loc, scale = stats.genpareto.fit(tail_scores)
    
    # Calculate threshold using inverse CDF
    threshold = stats.genpareto.ppf(q, shape, loc, scale)
    return threshold

def bring_approach(args):
    if args.approach == 'subAdjacent':
        from subAdjacent.trainer import train_model, detect_anomalies
        from subAdjacent.configClass import Config
        train_func = train_model
        detect_func = detect_anomalies
        config = Config()
    return config, train_func, detect_func