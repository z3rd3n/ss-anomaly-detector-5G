import logging
import os
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import csv


def start_logging(params):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(params.output_dir, 'logs')
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


def save_checkpoint(model, optimizer, epoch, loss, params):
    checkpoint_path = os.path.join(params.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    logging.info(f"Checkpoint saved: {checkpoint_path}")

def load_last_checkpoint(params, model, optimizer):
    result_files = [f for f in os.listdir(params.output_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    if not result_files:
        logging.info(f"No checkpoints found in {params.output_dir}, aborting detection.")
        return

    def get_epoch(fname):
        return int(fname.split('_')[-1].replace('.pt',''))
    result_files_sorted = sorted(result_files, key=lambda x: get_epoch(x))
    last_ckpt = os.path.join(params.output_dir, result_files_sorted[-1])
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


def plot_attention_matrices(model, dataloader, device, output_dir, max_plots=1):
    """
    Example of how to plot a single attention matrix from the last encoder layer.

    We only do `max_plots` heads (for demonstration).
    """
    model.eval()
    for i, batch in enumerate(dataloader):
        features = batch['features'].to(device)
        with torch.no_grad():
            enc_out, queries_list, keys_list = model(features)

        # queries_list[-1] & keys_list[-1] => last layer
        queries = queries_list[-1]  # shape [B, L, H, D]
        keys = keys_list[-1]       # shape [B, L, H, D]

        # We'll just pick the first sample & first head
        queries_0 = queries[0, :, 0, :]  # shape [L, D]
        keys_0 = keys[0, :, 0, :]        # shape [L, D]
        # Build attention matrix => (L,L)
        attn_matrix = queries_0 @ keys_0.T
        attn_matrix = attn_matrix / (1e-6 + attn_matrix.sum(dim=-1, keepdim=True))

        # Plot
        fig, ax = plt.subplots(figsize=(5,4))
        cax = ax.imshow(attn_matrix.cpu().numpy(), cmap='hot', interpolation='nearest')
        fig.colorbar(cax, ax=ax)
        ax.set_title("Attention Matrix (sample=0, head=0, last layer)")
        fig.tight_layout()

        # Save
        save_path = os.path.join(output_dir, f'attention_matrix_{i}.png')
        plt.savefig(save_path)
        plt.close(fig)
        logging.info(f"Saved attention matrix to {save_path}")

        # We only do max_plots
        if i+1 >= max_plots:
            break


def plot_loss_spans(rec_losses, sa_losses, output_dir):
    """
    Save a figure with reconstruction loss and SA loss vs sample index.
    rec_losses, sa_losses: arrays of shape [N], aligned with the same index.
    """
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(rec_losses, label='Reconstruction Loss')
    ax.plot(sa_losses, label='SA-con Loss')
    ax.set_title('Loss Curves Over Validation Samples')
    ax.set_xlabel('Index')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.tight_layout()
    save_path = os.path.join(output_dir, 'loss_spans.png')
    plt.savefig(save_path)
    plt.close(fig)
    logging.info(f"Saved rec/SA loss span plot to {save_path}")


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

def unscale_features(features, scaling_dir):
    # 1) Load scaling objects
    scaler_pkl_path = os.path.join(scaling_dir, "standard_scaler.pkl")
    if not os.path.exists(scaler_pkl_path):
        logging.warning("Scaling files not found; will save scaled features as-is.")
        unscaled = features
    else:
        with open(scaler_pkl_path, 'rb') as f:
            scaler = pickle.load(f)
            logging.info("Scaler mean_:", scaler.mean_)
            logging.info("Scaler scale_:", scaler.scale_)
        

        # If you used StandardScaler with shape [D], unscale
        # shape check
        unscaled = scaler.inverse_transform(features)

    # 2) Round to integers
    unscaled = np.rint(unscaled).astype(int)
    return unscaled

def unscale_and_save_anomalies(
    timestamps,
    features,
    anomalies_mask,
    threshold,
    feature_names,
    scaling_dir,
    output_csv
):
  
    unscaled = unscale_features(features, scaling_dir)

    anom_indices = np.where(anomalies_mask)[0]
    logging.info(f"Total anomaly count: {len(anom_indices)} (threshold={threshold:.4f})")

    if len(anom_indices) == 0:
        logging.info(f"No anomalies found above threshold = {threshold:.4f}. No CSV created.")
        return
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(["timestamp_str"] + feature_names)
        for idx in anom_indices:
            row_ts = timestamps[idx]
            row_feats = unscaled[idx].tolist()
            writer.writerow([row_ts] + row_feats)

    logging.info(f"Anomalies saved to CSV => {output_csv}")

def bring_approach(args):
    if args.approach == 'subAdjacent':
        from subAdjacent.trainer import train_model, detect_anomalies
        from subAdjacent.configClass import Config
        train_func = train_model
        detect_func = detect_anomalies
        config = Config()
    return {'params': config, 'train_func': train_func, 'detect_func': detect_func}