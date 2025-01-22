# trainer.py
import math
import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm  # <-- ADDED
from utils import save_checkpoint, unscale_features  # Make sure this is available
import numpy as np
import pandas as pd

def train_function(params, model, optimizer, scheduler, train_loader, val_loader):
    """
    A training routine that:
      - For each batch in train_loader:
        * Compute memory assoc_loss & surprise
        * Compute gating alpha_t = sigma(gamma * surprise)
        * Update memory w/ momentum, scaled by (1 - alpha_t)
        * Compute MSE recon => final_loss 
        * Backprop & step optimizer
      - After each epoch, do a validation pass (val_loader) computing recon + assoc_loss
        * No memory update in validation, but we do compute alpha_t for the final loss
        * Track the best validation performance and save checkpoint
        * Implement early stopping based on patience
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    num_epochs = params.num_epochs

    # For early stopping
    patience = getattr(params, 'early_stopping_patience', 5)  # default = 5 if not in params
    min_delta = getattr(params, 'early_stopping_min_delta', 1e-4)  # minimal required improvement
    wait = 0  # counter for how many epochs have passed with no improvement

    # Keep track of losses
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(num_epochs):
        ##################################################
        # 1) Training
        ##################################################
        model.train()
        total_loss = 0.0
        nbatches = 0

        # Wrap the dataloader with tqdm
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Train", leave=False)

        for batch_idx, batch in enumerate(train_pbar):
            x = batch['features'].to(device)  # [B, seq_len, d_in]
            nbatches += 1
            optimizer.zero_grad()
            x_recon, recon_loss, assoc_loss, surprise_score = model.update_memory(x)
            recon_loss.backward()
            optimizer.step()

            total_loss += recon_loss.item()
            train_pbar.set_postfix({'loss': f"{recon_loss.item():.4f}", 'surprise': f"{surprise_score.item():.4f}, assoc_loss: {assoc_loss.item():.4f}"})

        avg_train_loss = total_loss / max(1, nbatches)
        train_losses.append(avg_train_loss)

        # Step scheduler
        if scheduler is not None:
            scheduler.step(avg_train_loss)

        ##################################################
        # 2) Validation
        ##################################################
        model.eval()
        total_val_loss = 0
        total_surprise = 0
        total_assoc_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                x = batch['features'].to(device)
                
                # Get reconstruction
                _, recon_loss, assoc_loss, surprise_score = model.update_memory(x, is_training=False)
                            
                total_val_loss += recon_loss .item()
                total_surprise += surprise_score.item()
                total_assoc_loss += assoc_loss.item()
                num_batches += 1
        
        avg_val_loss = total_val_loss / num_batches
        avg_surprise = total_surprise / num_batches
        avg_assoc_loss = total_assoc_loss / num_batches

        ##################################################
        # 3) Logging & Checkpoints
        ##################################################
        logging.info((
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Val Assoc Loss: {avg_assoc_loss:.6f} | "
            f"Val Surprise: {avg_surprise:.6f}"

        ))

        # Check if this is the best validation so far; if yes, save model/optimizer
        if avg_val_loss < (best_val_loss - min_delta):
            best_val_loss = avg_val_loss
            best_epoch = epoch
            wait = 0  # reset counter

            # Save a checkpoint of the best model so far
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch+1,
                loss=best_val_loss,
                params=params
            )
        else:
            wait += 1

        # Early stopping check
        if wait >= patience:
            logging.info(f"Early stopping triggered! No improvement in val loss for {patience} consecutive epochs.")
            break

    logging.info(f"Training complete. Best Val Loss = {best_val_loss:.6f} at epoch {best_epoch+1}.")

    return train_losses, val_losses

def detect_function(model, val_loader, params, threshold=None):
    """Detect anomalies using both reconstruction and surprise metrics"""
    model.eval()
    device = params.device
    
    # Prepare DataFrame for anomalies
    anomalies = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Detecting anomalies")):
            x = batch['features'].to(device)
            timestamps = batch['timestamps']
            
            # Flatten timestamps if they're nested
            timestamps_flat = [item for sublist in timestamps for item in sublist]
            
            # Get reconstruction and attention outputs
            x_recon, recon_loss, assoc_loss, surprise_score = model.update_memory(x, is_training=False)
            
            
            # Combined anomaly score (can be adjusted based on your needs)
            lambda_mse = getattr(params, 'lambda_mse', 1.0)
            lambda_assoc = getattr(params, 'lambda_assoc', 1.0)
            lambda_surprise = getattr(params, 'lambda_surprise', 1.0)
            
            anomaly_scores = lambda_mse * recon_loss + lambda_assoc * assoc_loss + lambda_surprise * surprise_score 
            
            # Move to CPU for processing
            anomaly_scores = anomaly_scores.cpu().numpy()
            x_np = x.cpu().numpy()
            x_recon_np = x_recon.cpu().numpy()
                      

            # Unscale features if needed (assuming you have the function)
            batch_features = x_np.reshape(-1, x.shape[-1])
            batch_preds = x_recon_np.reshape(-1, x.shape[-1])
            
            unscaled = unscale_features(batch_features)
            predicted = unscale_features(batch_preds)
            
            anomalies.append({
                'timestamp': timestamps_flat,
                'batch_idx': batch_idx,
                'anomaly_score': anomaly_scores,
                'mse_loss': recon_loss.item(),
                'assoc_loss': assoc_loss.item(),
                'surprise_score': surprise_score.item(),
                **{f'feature_{i}': unscaled[:, i] for i in range(unscaled.shape[1])},
                **{f'predicted_{i}': predicted[:, i] for i in range(predicted.shape[1])}
            })
    
    # Create DataFrame
    anomalies_df = pd.DataFrame(anomalies)
    
    if anomalies_df.empty:
        logging.warning("No anomalies found (empty DataFrame).")
    else:
        # Sort by anomaly score
        anomalies_df = anomalies_df.sort_values('anomaly_score', ascending=False)
        logging.info(f"Found {len(anomalies_df)} anomalies in total.")
    
    # Save results
    save_path = f"anomalies_th{threshold:.4f}.csv"
    anomalies_df.to_csv(save_path, index=False)
    logging.info(f"Saved anomalies to {save_path}")
    
    return anomalies_df, None