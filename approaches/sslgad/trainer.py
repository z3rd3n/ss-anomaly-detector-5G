import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto  # if needed elsewhere
from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

########################################
# Helper functions for checkpointing
########################################

def save_checkpoint_sslgad(model, optimizer, epoch, best_accuracy, params, filepath):
    """Save model and optimizer state along with training metadata."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy,
    }
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved to {filepath}.")

def load_checkpoint_sslgad(model, optimizer, filepath, device):
    """Load model and optimizer state from a checkpoint file."""
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Loaded checkpoint from {filepath} (epoch {checkpoint['epoch']}).")
    else:
        logging.warning(f"No checkpoint found at {filepath}.")

########################################
# CSV-based Dataset for Validation/Testing
########################################

class CSVValidationDataset(Dataset):
    """
    A dataset for loading validation data from a CSV file.
    Expected CSV columns (required):
       - "timestamp_str" (timestamp)
       - "SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI" (features)
    For simplicity, each row is treated as a sequence with length 1.
    """
    def __init__(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        self.df = pd.read_csv(csv_path)
        required_columns = ["timestamp_str", "SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"]
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' is missing from {csv_path}!")
        # Extract features and timestamps
        self.features = self.df[["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"]].values.astype(np.float32)
        self.timestamps = self.df["timestamp_str"].tolist()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Each row is a sequence of length 1
        feature = self.features[idx]  # shape: (num_features,)
        # Convert to tensor and add a sequence dimension.
        feature_seq = torch.tensor(feature, dtype=torch.long).unsqueeze(0)  # (1, num_features)
        # Wrap the timestamp in a list to maintain a sequence-like structure.
        timestamp = self.timestamps[idx]
        return {'features': feature_seq, 'timestamps': [timestamp]}

########################################
# Contrastive Loss and Local Score Functions
########################################

def compute_contrastive_loss(z, temperature=0.1):
    """
    Compute a simple contrastive loss (e.g., InfoNCE) on a batch of latent representations.
    
    Args:
        z: Tensor of shape (batch_size, seq_len, d_model).
        temperature: Temperature scaling parameter.
        
    Returns:
        Scalar loss.
        
    Note:
        For simplicity, we create positive pairs by pairing each z_t with z_{t+1} (for t < seq_len-1),
        and treat all other pairs as negatives.
    """
    batch_size, seq_len, d_model = z.shape
    if seq_len < 2:
        # No valid positive pairs if sequence length is less than 2.
        return torch.tensor(0.0, device=z.device)
    
    # Positive pairs: (z[:, t, :], z[:, t+1, :]) for t=0,...,seq_len-2
    z1 = z[:, :-1, :]   # shape: (batch_size, seq_len-1, d_model)
    z2 = z[:, 1:, :]    # shape: (batch_size, seq_len-1, d_model)
    z1 = z1.contiguous().view(-1, d_model)  # (B*(seq_len-1), d_model)
    z2 = z2.contiguous().view(-1, d_model)
    
    # Normalize representations
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Similarity matrix
    sim = torch.matmul(z1, z2.T)  # (N, N)
    sim = sim / temperature
    
    # Create labels (each element i should match with i)
    N = sim.shape[0]
    labels = torch.arange(N).to(z.device)
    
    loss = F.cross_entropy(sim, labels)
    return loss

def compute_local_anomaly_scores(seq_z, window_size=3):
    """
    Compute a local anomaly score for a sequence of latent representations.
    
    Args:
        seq_z: numpy array of shape (seq_len, d_model) for one sequence.
        window_size: number of neighbors to consider on each side.
        
    Returns:
        List of anomaly scores (length seq_len).
        
    For each time step, the score is the average Euclidean distance between the current latent
    vector and its neighbors within ±window_size (excluding itself).
    """
    seq_len, d = seq_z.shape
    scores = []
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        # Exclude the current point from the neighbors
        neighbors = np.delete(seq_z[start:end], i - start, axis=0)
        if len(neighbors) > 0:
            score = np.mean(np.linalg.norm(seq_z[i] - neighbors, axis=1))
        else:
            score = 0.0
        scores.append(score)
    return scores

########################################
# Detection Functions
########################################

def detect_sslgad(model, dataloader, params):
    """
    Perform anomaly detection on a given dataloader using the SSLGAD method.
    The procedure computes latent representations using the encoder, estimates a local score and a global score
    (via a density estimator), then flags timestamps whose combined score exceeds a threshold.
    
    Returns:
        anomalies_df: DataFrame with columns ['timestamp', 'anomaly_score'].
        fig_path: Path to the summary plot of anomaly scores.
    """
    device = params.device
    model.eval()
    
    all_latents = []      # Latent representations for global density estimation.
    all_timestamps = []   # Corresponding timestamps.
    all_local_scores = [] # Local anomaly scores computed for each latent vector.
    
    logging.info("Starting detection on validation/test data...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Detecting anomalies"):
            features = batch['features'].to(device)  # shape: (B, seq_len, num_features)
            timestamps_batch = batch['timestamps']     # list of list of timestamp strings
            # Get latent representations via model.encode
            z = model.encode(features)  # shape: (B, seq_len, d_model)
            z_np = z.cpu().numpy()
            B, seq_len, d_model = z_np.shape
            # Process each sequence in the batch
            for b in range(B):
                seq_z = z_np[b]  # (seq_len, d_model)
                seq_timestamps = timestamps_batch[b]  # list of length seq_len
                local_scores = compute_local_anomaly_scores(seq_z, window_size=params.window_size)
                all_latents.append(seq_z)         # later stack them (each row corresponds to one time step)
                all_timestamps.extend(seq_timestamps)
                all_local_scores.extend(local_scores)
    
    # Stack latent representations from all sequences: shape (N, d_model)
    all_latents_np = np.vstack(all_latents)
    
    # Global density estimation via a Gaussian Mixture Model.
    gmm = GaussianMixture(n_components=params.gmm_components, covariance_type='full', random_state=42)
    gmm.fit(all_latents_np)
    # Global anomaly score: negative log-likelihood.
    global_nll = -gmm.score_samples(all_latents_np)  # shape: (N,)
    
    # Combine local and global scores.
    beta = params.beta  # Weight for the local score.
    local_scores_np = np.array(all_local_scores)
    final_scores = beta * local_scores_np + (1 - beta) * global_nll
    
    # Determine a threshold (e.g., mean + k * std)
    k = params.anomaly_threshold_std  # Hyperparameter (e.g., 3.0)
    threshold = final_scores.mean() + k * final_scores.std()
    
    anomalies = []
    for ts, score in zip(all_timestamps, final_scores):
        if score > threshold:
            anomalies.append((ts, score))
    
    anomalies_df = pd.DataFrame(anomalies, columns=['timestamp', 'anomaly_score'])
    anomalies_df = anomalies_df.sort_values('anomaly_score', ascending=False)
    
    # Save top anomalies to CSV.
    anomalies_csv_path = os.path.join(params.output_dir, "anomalies.csv")
    anomalies_df.head(1000).to_csv(anomalies_csv_path, index=False)
    logging.info(f"Saved top anomalies to {anomalies_csv_path}")
    
    # Plot anomaly scores over time.
    plt.figure(figsize=(12, 6))
    plt.plot(final_scores, label='Final Anomaly Score', alpha=0.7)
    plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
    plt.xlabel("Time step index (ordered as processed)")
    plt.ylabel("Anomaly Score")
    plt.title("Anomaly Scores from SSLGAD")
    plt.legend()
    fig_path = os.path.join(params.output_dir, "anomaly_scores.png")
    plt.savefig(fig_path)
    plt.close()
    
    logging.info(f"Saved anomaly score plot to {fig_path}")
    return anomalies_df, fig_path

def validate_sslgad_csv(model, params):
    """
    Validation routine for SSLGAD.
    Loads the validation CSV, runs detection to flag anomalies, and compares the detected
    timestamps with those in the ground truth CSV.
    
    Returns:
        accuracy: The fraction of detected anomalies that match the ground truth.
    """
    logging.info("Starting validation procedure...")
    # Create a dataset and dataloader for validation data.
    val_dataset = CSVValidationDataset(params.validation_csv_path)
    val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False)
    
    anomalies_df, _ = detect_sslgad(model, val_loader, params)
    
    if anomalies_df.empty:
        logging.warning("No anomalies detected during validation.")
        return 0.0
    
    # Load ground truth CSV (assumes column "timestamp_str" or "timestamp")
    df_gt = pd.read_csv(params.ground_truth_csv_path)
    gt_col = "timestamp_str" if "timestamp_str" in df_gt.columns else "timestamp"
    gt_timestamps = set(df_gt[gt_col].unique())
    
    # Mark each anomaly as correct if its timestamp is found in the ground truth.
    anomalies_df["is_accurate"] = anomalies_df["timestamp"].apply(lambda ts: ts in gt_timestamps)
    num_accurate = anomalies_df["is_accurate"].sum()
    total_anoms = len(anomalies_df)
    accuracy = (num_accurate / total_anoms) if total_anoms > 0 else 0.0
    
    logging.info(f"[RESULT] Validation anomaly detection accuracy: {accuracy*100:.2f}% "
                 f"({num_accurate} correct out of {total_anoms} anomalies).")
    return accuracy

########################################
# Training Function for SSLGAD with Validation and Early Stopping
########################################

def train_sslgad(params, model, optimizer, scheduler, train_loader, val_loader=None):
    """
    Train the model using a contrastive SSLGAD loss and periodically run a validation routine.
    The validation step runs the detection procedure on a CSV-based validation set and compares with
    ground truth. Early stopping and checkpointing are applied based on the validation anomaly detection accuracy.
    
    Args:
        params: Object with training parameters.
        model: The model to train (must implement an .encode() method).
        optimizer: Optimizer.
        scheduler: Learning rate scheduler (can be None).
        train_loader: DataLoader for training (expects batches with key 'features').
        val_loader: (Optional) DataLoader for validation. If None, CSV-based validation will be used.
        
    Returns:
        train_losses: List of average training losses per epoch.
        best_accuracy: Best validation accuracy achieved.
    """
    device = params.device
    model = model.to(device)
    best_accuracy = -1.0  # Use -inf if you prefer; here we maximize accuracy.
    patience = params.early_stopping_patience
    min_delta = params.early_stopping_min_delta
    wait = 0
    train_losses = []

    logging.info("Starting SSLGAD training...")
    for epoch in range(1, params.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # tqdm progress bar for training batches.
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{params.num_epochs} [Train]", leave=False)
        for batch in pbar:
            features = batch['features'].to(device)  # shape: (B, seq_len, num_features)
            z = model.encode(features)  # get latent representations
            loss = compute_contrastive_loss(z, temperature=params.temperature)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        train_losses.append(avg_loss)
        logging.info(f"Epoch {epoch}: Average Training Loss = {avg_loss:.4f}")
        
        if scheduler is not None:
            scheduler.step(avg_loss)
        
        # Validation (using CSV-based validation)
        val_accuracy = validate_sslgad_csv(model, params)
        logging.info(f"Epoch {epoch}: Validation Accuracy = {val_accuracy:.4f}")
        
        # Early stopping & checkpointing (maximize accuracy)
        if val_accuracy > best_accuracy + min_delta:
            best_accuracy = val_accuracy
            wait = 0
            checkpoint_path = os.path.join(params.output_dir, 'checkpoint_best.pt')
            save_checkpoint_sslgad(model, optimizer, epoch, best_accuracy, params, checkpoint_path)
            logging.info(f"Epoch {epoch}: New best accuracy, checkpoint saved.")
        else:
            wait += 1
            logging.info(f"Epoch {epoch}: No improvement. Early stopping counter: {wait}/{patience}")
            if wait >= patience:
                logging.info("Early stopping triggered.")
                break

    logging.info(f"Training complete. Best Validation Accuracy = {best_accuracy:.6f}")
    return train_losses, best_accuracy