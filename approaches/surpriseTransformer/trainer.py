import numpy as np
import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import save_checkpoint, load_checkpoint, unscale_features
import os
import pandas as pd
from scipy.stats import genpareto

import torch

# ----- POT–based threshold helper -----
def compute_pot_threshold(anomaly_scores, base_quantile=0.95, tail_probability=0.99):
    """
    Given a 1D numpy array of anomaly_scores, compute a POT-based threshold.
    1. Compute the base threshold u as the base_quantile (e.g., 95th percentile).
    2. Fit a Generalized Pareto Distribution (GPD) to the exceedances above u.
    3. Return the threshold at the desired tail probability.
    """
    u = np.percentile(anomaly_scores, base_quantile * 100)
    # Select exceedances above u
    exceedances = anomaly_scores[anomaly_scores > u] - u
    if len(exceedances) < 10:
        # Not enough tail samples: fall back to a simple threshold
        return u + np.std(anomaly_scores)
    shape, loc, scale = genpareto.fit(exceedances, floc=0)
    threshold = u + genpareto.ppf(tail_probability, shape, loc=0, scale=scale)
    return threshold

def validate_csv_file(model, params, anomaly_threshold):
    """
    Loads validation CSV, computes anomaly scores, flags anomalies using the provided anomaly_threshold,
    and compares their timestamp_str values with those in ground_truth_csv_path.
    """

    # Load validation CSV
    df = pd.read_csv(params.validation_csv_path)
    required_columns = ["timestamp_str", "SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from {params.validation_csv_path}!")
    
    # Prepare features: only 7 features are used.
    features_np = df[["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"]].values.astype(np.int64)
    # Add a sequence dimension: each row is treated as a sequence of length 1.
    features_np = np.expand_dims(features_np, axis=1)  # shape: [N, 1, 7]
    features_tensor = torch.tensor(features_np, dtype=torch.long)
    
    # Also prepare the timestamps (each as a list of length 1)
    timestamps = df["timestamp_str"].tolist()
    timestamps = [[ts] for ts in timestamps]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features_tensor = features_tensor.to(device)
    model = model.to(device)
    model.eval()
    
    # Get model submodules.
    embed_multi   = model.embed_multi
    memory_module = model.memory_module
    reconstructor = model.reconstructor
    
    with torch.no_grad():
        x_up = embed_multi(features_tensor)
    
    k = x_up.detach().clone().requires_grad_(True)
    v_hat = memory_module(k)
    assoc_loss = torch.nn.functional.mse_loss(v_hat, k, reduction='none').mean(dim=-1)  # [N,1]
    loss_sum = assoc_loss.sum()
    
    grad = torch.autograd.grad(loss_sum, k, retain_graph=True)[0]
    N, S, _ = features_tensor.shape
    grad_norm = grad.view(N, S, -1).norm(p=2, dim=-1)  # [N,1]
    surprise = grad_norm.detach()
    
    logits = reconstructor(x_up, memory_module)
    recon_loss = compute_multi_feature_ce_loss_per_sample(logits, features_tensor, params.cardinalities)  # [N,1]
    
    # Compute anomaly score without the surprise term:
    anomaly_score = assoc_loss + recon_loss  # [N,1]
    anomaly_score = anomaly_score.squeeze(1)  # shape: [N]
    
    # Add computed loss values back to the dataframe.
    df["anomaly_score"] = anomaly_score.detach().cpu().numpy()
    df["assoc_loss"] = assoc_loss.squeeze(1).detach().cpu().numpy()
    df["recon_loss"] = recon_loss.squeeze(1).detach().cpu().numpy()
    df["surprise"]   = surprise.squeeze(1).detach().cpu().numpy()
    
    # Flag anomalies based on the provided threshold.
    anomalies_df = df[df["anomaly_score"] > anomaly_threshold].copy()
    
    anomalies_csv_path = os.path.join(params.output_dir, "anomalies.csv")
    anomalies_df.to_csv(anomalies_csv_path, index=False)
    
    # Load ground truth CSV and compare timestamp_str.
    df_gt = pd.read_csv(params.ground_truth_csv_path)
    if "timestamp_str" not in df_gt.columns:
        raise ValueError(f"'timestamp_str' column missing from {params.ground_truth_csv_path}!")
    gt_timestamps = set(df_gt["timestamp_str"].unique())
    
    # For each anomaly, mark whether its timestamp_str exists in the ground truth.
    anomalies_df["is_accurate"] = anomalies_df["timestamp_str"].apply(lambda ts: ts in gt_timestamps)
    num_accurate = anomalies_df["is_accurate"].sum()
    total_anoms  = len(anomalies_df)
    accuracy = (num_accurate / total_anoms) if total_anoms > 0 else 0.0
    print(f"[RESULT] Anomaly detection accuracy: {accuracy*100:.2f}% ({num_accurate} out of {total_anoms} anomalies)")
    
    return accuracy

def compute_rule_violations_in_sequence(x_int, max_retx=4):
    """
    x_int: [B, S, 7] where features are:
         [SFN, Slot, HARQ, MCS, CRC, ReTx, NDI]
    Returns:
       rule_mask: [B, S] with count of rule violations.
       rule_flags: nested list with triggered rule names per sample.
    """
    B, S, _ = x_int.shape
    SFN_idx, Slot_idx, HARQ_idx = 0, 1, 2
    MCS_idx, CRC_idx, ReTx_idx  = 3, 4, 5
    NDI_idx = 6

    rule_flags = [[[] for _ in range(S)] for _ in range(B)]
    rule_mask  = torch.zeros((B, S), dtype=torch.float, device=x_int.device)

    for b in range(B):
        last_harq_sample = {}
        for s_idx in range(S):
            HARQ_val = x_int[b, s_idx, HARQ_idx].item()
            ReTx_val = x_int[b, s_idx, ReTx_idx].item()
            NDI_val  = x_int[b, s_idx, NDI_idx].item()
            triggered_rules = []

            # Rule 1: High reTx
            if ReTx_val >= max_retx:
                triggered_rules.append("high_retx")

            # Compare with previous sample in same HARQ if available
            if HARQ_val in last_harq_sample:
                prev_idx = last_harq_sample[HARQ_val]
                prev_CRC = x_int[b, prev_idx, CRC_idx].item()
                prev_NDI = x_int[b, prev_idx, NDI_idx].item()

                # Rule 2: Unnecessary reTx
                if (ReTx_val > 0) and (prev_CRC == 1):
                    triggered_rules.append("unnecessary_retx")
                # Rule 3: Missing reTx
                if (ReTx_val == 0) and (prev_CRC == 0) and (NDI_val == prev_NDI):
                    triggered_rules.append("missing_retx")
                # Rule 4: New data with no reTx after CRC failure
                if (prev_CRC == 0) and (NDI_val != prev_NDI):
                    triggered_rules.append("new_data_no_retx")
            last_harq_sample[HARQ_val] = s_idx

            if triggered_rules:
                rule_flags[b][s_idx] = triggered_rules
                rule_mask[b, s_idx] = len(triggered_rules)
    return rule_mask, rule_flags

def compute_multi_feature_ce_loss(logits, x_int, cardinalities):
    """
    logits: [B, S, sum(cardinalities)]
    x_int:  [B, S, 7] ground-truth (integer) labels.
    Returns the average cross entropy loss across all features.
    """
    B, S, _ = logits.shape
    total_ce = 0.0
    offset = 0
    for i, card in enumerate(cardinalities):
        logits_i = logits[..., offset:offset+card]  # [B, S, card]
        offset += card
        target_i = x_int[..., i]  # [B, S]
        ce_i = F.cross_entropy(
            logits_i.view(-1, card),
            target_i.view(-1),
            reduction='mean'
        )
        total_ce += ce_i
    loss = total_ce / len(cardinalities)
    return loss

def compute_multi_feature_ce_loss_per_sample(logits, x_int, cardinalities):
    """
    Computes cross entropy loss per sample (per time step).
    Returns a tensor of shape [B, S].
    """
    B, S, _ = logits.shape
    losses = []
    offset = 0
    for i, card in enumerate(cardinalities):
        logits_i = logits[..., offset:offset+card]  # [B, S, card]
        offset += card
        target_i = x_int[..., i]  # [B, S]
        ce_i = F.cross_entropy(
            logits_i.view(-1, card),
            target_i.view(-1),
            reduction='none'
        )
        ce_i = ce_i.view(B, S)
        losses.append(ce_i)
    loss_per_sample = sum(losses) / len(cardinalities)
    return loss_per_sample


def train_function(params, model, optimizer, scheduler, train_loader, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    memory_module = model.memory_module
    reconstructor = model.reconstructor
    surprise_gate = model.surprise_gate  # now returns binary gate
    embed_multi   = model.embed_multi

    optimizer = torch.optim.AdamW(
        [
            {"params": memory_module.parameters(), "lr": params.lr_memory},
            {
                "params": list(reconstructor.parameters()) 
                        + list(embed_multi.parameters()) 
                        + list(surprise_gate.parameters()),
                "lr": params.learning_rate,
            },
        ],
        weight_decay=params.weight_decay,
    )

    best_accuracy = -1.0
    patience = params.early_stopping_patience
    min_delta = params.early_stopping_min_delta
    wait = 0
    train_losses = []

    # --- Initialize moving average estimates for anomaly scores ---
    threshold_ema_mean = None
    threshold_ema_var = None
    ema_alpha = 0.1  # Smoothing factor for exponential moving average

    for epoch in range(params.num_epochs):
        model.train()
        epoch_loss = 0.0
        train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{params.num_epochs} [Train]")
        for batch_idx, batch in enumerate(train_pbar):
            x = batch['features'].to(device)  # Shape: [B, S, 7]
            optimizer.zero_grad()

            # 1. ---- Up-project input ----
            x_up = embed_multi(x)

            # 2. ---- Association Loss with Surprise ----
            # Compute association loss and its gradients
            k = x_up.detach().clone().requires_grad_(True)
            v_hat = memory_module(k)
            assoc_loss_tensor = F.mse_loss(v_hat, k, reduction='none')  # [B, S, d_model]
            assoc_loss_per_sample = assoc_loss_tensor.mean(dim=-1)        # [B, S]
            loss_sum = assoc_loss_per_sample.sum()
            
            grad = torch.autograd.grad(loss_sum, k, retain_graph=True)[0]
            grad_norm = grad.view(k.size(0), k.size(1), -1).norm(p=2, dim=-1)  # [B, S]

            # 3. Normalize gradients (compute per-batch z-scores)
            batch_mean = grad_norm.mean(dim=1, keepdim=True)  # [B, 1]
            batch_std = grad_norm.std(dim=1, keepdim=True) + 1e-6  # [B, 1]
            surprise_z = (grad_norm - batch_mean) / batch_std  # [B, S]

            # 4. Gate out high-surprise samples
            T = model.surprise_gate.threshold  # learnable threshold (scalar)
            gate = (surprise_z <= T).float()  # [B, S]
            weighted_assoc_loss = (assoc_loss_per_sample * gate).sum() / (gate.sum() + 1e-6)

            # 5. ---- Reconstruction Loss ----
            logits = reconstructor(x_up, memory_module)
            recon_loss_ce = compute_multi_feature_ce_loss(logits, x, params.cardinalities)

            # 6. Surprise Regularization: Penalize excess surprise over the threshold.
            surprise_penalty = torch.clamp(surprise_z - T, min=0).pow(2).mean()

            # Total loss (note: no surprise term in the loss)
            total_loss = recon_loss_ce + params.assoc_loss_weight * weighted_assoc_loss \
                         + params.surprise_penalty_weight * surprise_penalty
            total_loss.backward()
            optimizer.step()

            # ---- Compute Anomaly Score per Sample ----
            # Now anomaly_score is computed only as: assoc_loss + recon_loss
            recon_loss_per_sample = compute_multi_feature_ce_loss_per_sample(logits, x, params.cardinalities)
            anomaly_score_batch = assoc_loss_per_sample + recon_loss_per_sample  # [B, S]

            # Update moving average estimates based on current batch anomaly scores:
            batch_mean = anomaly_score_batch.mean().item()
            batch_var = anomaly_score_batch.var(unbiased=False).item()  # population variance

            if threshold_ema_mean is None:
                threshold_ema_mean = batch_mean
                threshold_ema_var = batch_var
            else:
                threshold_ema_mean = ema_alpha * batch_mean + (1 - ema_alpha) * threshold_ema_mean
                threshold_ema_var = ema_alpha * batch_var + (1 - ema_alpha) * threshold_ema_var

            # Derive a threshold from the moving averages (e.g., mean + 5*std)
            current_threshold = threshold_ema_mean + 5 * (threshold_ema_var ** 0.5)
            
            train_batches += 1
            epoch_loss += total_loss.item()
            
            train_pbar.set_postfix({
                'total_loss': f"{total_loss.item():.4f}",
                'recon_loss': f"{recon_loss_ce.item():.4f}",
                'assoc_loss': f"{(params.assoc_loss_weight * weighted_assoc_loss).item():.4f}",
                'surprise': f"{surprise_penalty.mean().item():.4f}",
                'ano_score': f"{anomaly_score_batch.mean().item():.4f}",
                'threshold': f"{current_threshold:.4f}"
            })
            
        avg_train_loss = epoch_loss / train_batches
        train_losses.append(avg_train_loss)

        logging.info(f"[Epoch {epoch+1}] Moving average anomaly threshold: {current_threshold:.4f}")
        
        # ---- Validation ----
        model.eval()
        accuracy = validate_csv_file(model, params, current_threshold)
        logging.info(f"Epoch {epoch+1}: Accuracy = {accuracy:.4f}")
        
        # --- Early stopping / checkpointing ---
        if accuracy < best_accuracy - min_delta:
            best_accuracy = accuracy
            wait = 0
            save_checkpoint(model, optimizer, epoch+1, best_accuracy, params,
                            filepath=os.path.join(params.output_dir, 'checkpoint_best.pt'))
        else:
            wait += 1
            logging.info(f"Early stopping counter: {wait}/{patience}")
            if wait >= patience:
                logging.info("Early stopping triggered.")
                break

    logging.info(f"Training complete. Best Accuracy = {best_accuracy:.6f}")
    return train_losses, best_accuracy


########################################
# Detection Function
########################################

def detect_function(model, test_loader, params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    memory_module = model.memory_module
    reconstructor = model.reconstructor
    embed_multi = model.embed_multi
    pos_embedding = model.pos_embedding

    checkpoint_path = os.path.join(params.output_dir, 'checkpoint_best.pt')
    load_checkpoint(model, None, checkpoint_path)
    model.eval()
    anomalies_records = []

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Detecting anomalies")):
        x = batch['features'].to(device)  # [B, S, 7]
        timestamps = batch.get('timestamps', None)
        B, S, D_in = x.shape
        x_up = embed_multi(x)

        k = x_up.detach().clone().requires_grad_(True)
        v_hat = model.memory_module(k)
        assoc_loss = F.mse_loss(v_hat, k, reduction='none').mean(dim=-1)  # [B, S]
        loss_sum = assoc_loss.sum()
        grad = torch.autograd.grad(loss_sum, k, retain_graph=True)[0]
        grad_norm = grad.view(B, S, -1).norm(p=2, dim=-1)  # [B, S]
        
        # Compute normalized surprise z-scores
        batch_mean = grad_norm.mean(dim=1, keepdim=True)
        batch_std = grad_norm.std(dim=1, keepdim=True) + 1e-6
        surprise_z = (grad_norm - batch_mean) / batch_std  # [B, S]
        
        # Reconstruction loss per time step
        logits = model.reconstructor(x_up, model.memory_module)
        recon_loss = compute_multi_feature_ce_loss_per_sample(logits, x, params.cardinalities)

        # Compute anomaly score as assoc_loss + recon_loss (without surprise term)
        anomaly_score = assoc_loss + recon_loss + params.surprise_penalty_weight * torch.clamp(surprise_z - model.surprise_gate.threshold, min=0)

        for b in range(B):
            for s in range(S):
                record = {
                    'batch_idx': batch_idx,
                    'timestamp': (timestamps[b][s].item() if timestamps is not None and hasattr(timestamps[b][s], 'item') 
                                  else (timestamps[b][s] if timestamps is not None else None)),
                    'anomaly_score': anomaly_score[b, s].item(),
                    'assoc_loss': assoc_loss[b, s].item(),
                    'recon_loss': recon_loss[b, s].item(),
                    'surprise': anomaly_score[b, s].item()
                }
                anomalies_records.append(record)

    anomalies_df = pd.DataFrame(anomalies_records)
    if anomalies_df.empty:
        logging.warning("No data found in test set.")
        return anomalies_df
    anomalies_df = anomalies_df.sort_values('anomaly_score', ascending=False)
    save_path = os.path.join(params.output_dir, "anomalies.csv")
    anomalies_df.head(1000).to_csv(save_path, index=False)
    logging.info(f"Saved top anomalies to {save_path}")
    return anomalies_df
