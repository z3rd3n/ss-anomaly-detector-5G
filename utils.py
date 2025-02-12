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
    # if approach is not None:
    #     output_dir = os.path.join(approach, 'results')
    # if params is None and approach is None:
    #     output_dir = 'results'
    # else:
    #     #experiment_name = f"q{str(params.q)[-2:]}p{params.p}_s{params.seq_len}_h{params.n_heads}_e{params.e_layers}_d{params.model_dim}"
    #     experiment_name = "surpriseTransformer"
    #     output_dir = os.path.join(output_dir, experiment_name)
    #     params.output_dir = output_dir

    # hard_coded
    output_dir = params.output_dir
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
    # Define the checkpoint directory and ensure it exists
    checkpoint_dir = os.path.join(params.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)  # Creates the directory if it doesn't exist

    # Define the checkpoint file path
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')

    # Prepare the checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'learning_rate': params.learning_rate if hasattr(params, 'learning_rate') else None,
    }

    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved: {checkpoint_path}")
    logging.info(f"Epoch: {epoch}, Loss: {loss:.4f}")

    if hasattr(params, 'learning_rate'):
        logging.info(f"Learning Rate: {params.learning_rate}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Helper to load a checkpoint from checkpoint_path into model/optimizer
    """
    logging.info(f"Loading checkpoint from {checkpoint_path} ...")
    device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logging.info("Checkpoint loaded.")


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
    device="cpu", 
    max_plots=8, 
    max_heads=4, 
    sample_idx=0
):
    """
    Returns a list of (title_string, figure) for each attention matrix plot.
    We won't save to disk. We'll just generate them so we can show them in Streamlit.
    """
    import math

    model.to(device)
    model.eval()
    plots = []
    batch_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_plots:
            break
        features = batch["features"].to(device)
        with torch.no_grad():
            enc_out, queries_list, keys_list = model(features)
        
        # For demonstration, let's say we just take the last layer's queries/keys:
        # queries shape [B, L, H, D], keys shape [B, L, H, D]
        if not queries_list:
            logging.warning("queries_list is empty, skipping attention plotting.")
            break
        queries = queries_list[-1]  # shape [B, L, H, D]
        keys = keys_list[-1]        # shape [B, L, H, D]
        if sample_idx >= queries.shape[0]:
            # skip if sample_idx not available
            logging.warning(f"sample_idx={sample_idx} out of range for this batch, skipping.")
            continue
        
        # Extract single sample
        q_0 = queries[sample_idx]  # [L, H, D]
        k_0 = keys[sample_idx]     # [L, H, D]
        # rearr => [H, L, D]
        q_0 = q_0.permute(1,0,2)
        k_0 = k_0.permute(1,0,2)
        # Compute attention per head => Q * K^T
        d_k = q_0.size(-1)
        scale = 1.0 / (d_k ** 0.5)
        attn_matrices = torch.bmm(q_0, k_0.transpose(1,2)) * scale
        attn_matrices = torch.softmax(attn_matrices, dim=-1)  # [H, L, L]

        # We'll plot up to max_heads
        num_heads = min(attn_matrices.size(0), max_heads)
        num_cols = 2
        num_rows = math.ceil(num_heads / num_cols)
        fig, axes = plt.subplots(
            nrows=num_rows,
            ncols=num_cols,
            figsize=(5*num_cols, 5*num_rows),
            squeeze=False
        )
        fig.suptitle(f"Batch {batch_idx}, Sample {sample_idx}, {num_heads} heads")

        for h in range(num_heads):
            attn_head_h = attn_matrices[h].cpu().numpy()
            r = h // num_cols
            c = h % num_cols
            ax = axes[r, c]
            im = ax.imshow(attn_head_h, cmap="hot", aspect="auto")
            ax.set_title(f"Head {h}")
            ax.set_xlabel("Key positions")
            ax.set_ylabel("Query positions")
            fig.colorbar(im, ax=ax)
        
        # Hide any extra subplots if heads < num_rows*num_cols
        for h in range(num_heads, num_rows*num_cols):
            r = h // num_cols
            c = h % num_cols
            fig.delaxes(axes[r, c])

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plots.append((f"Attention Batch{batch_idx}_Sample{sample_idx}", fig))
        batch_count += 1
    
    return plots


def plot_anomaly_scores(scores, threshold):
    """
    Returns (title_string, figure) for anomaly scores.
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    idx = np.arange(len(scores))
    ax.plot(idx, scores, label="Anomaly Score")
    ax.axhline(threshold, color='r', linestyle='--', label=f'Threshold={threshold:.2f}')
    
    # highlight anomalies
    anomalies_mask = scores > threshold
    ax.scatter(idx[anomalies_mask], scores[anomalies_mask], color='red', s=10, label='Detected Anomalies')
    ax.set_title("Anomaly Scores")
    ax.set_xlabel("Index")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    return ("Anomaly Scores", fig)

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

        mean = np.delete(mean, 2)
        scale = np.delete(scale, 2)
        
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

def count_trainable_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def calculate_threshold_evt(scores, q=0.99, p=95):
    # Fit generalized Pareto distribution
    tail_scores = scores[scores > np.percentile(scores, p)]
    shape, loc, scale = stats.genpareto.fit(tail_scores)
    
    # Calculate threshold using inverse CDF
    threshold = stats.genpareto.ppf(q, shape, loc, scale)
    return threshold

def bring_approach(args):
    if args.approach == 'subAdjacent':
        from approaches.subAdjacent.trainer import train_model, detect_anomalies, detect_anomalies_from_threshold
        from approaches.subAdjacent.configClass import Config
        train_func = train_model
        detect_func = detect_anomalies_from_threshold
        config = Config()

    elif args.approach == 'subAdjacentLSTM':
        from approaches.subAdjacentLSTM.trainer import train_model, detect_anomalies
        from approaches.subAdjacentLSTM.configClass import Config
        train_func = train_model
        detect_func = detect_anomalies
        config = Config()

    elif args.approach == 'subAdjacentEmbed':
        from approaches.subAdjacentEmbed.trainer import train_model, detect_and_categorical_anomalies
        from approaches.subAdjacentEmbed.configClass import Config
        train_func = train_model
        detect_func = detect_and_categorical_anomalies
        config = Config()

    elif args.approach == 'surpriseTransformer':
        from approaches.surpriseTransformer.trainer import train_function, detect_function
        from approaches.surpriseTransformer.configClass import Config
        train_func = train_function
        detect_func = detect_function
        config = Config()

    elif args.approach == 'sslgad':
        from approaches.sslgad.trainer import train_sslgad, detect_sslgad
        from approaches.sslgad.configClass import Config
        train_func = train_sslgad
        detect_func = detect_sslgad
        config = Config()

    elif args.approach == 'tranad':
        from approaches.tranAD.trainer import train_tranad, detect_tranad
        from approaches.tranAD.configClass import Config
        train_func = train_tranad
        detect_func = detect_tranad
        config = Config()    

    return config, train_func, detect_func