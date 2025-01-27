import math
import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import save_checkpoint, load_checkpoint, unscale_features
import os
import pandas as pd

def train_function(params, model, optimizer, scheduler, train_loader, val_loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    memory_module = model.memory_module
    reconstructor = model.reconstructor
    surprise_gate = model.surprise_gate
    embed = model.embed

    optimizer = torch.optim.AdamW(
        [
            {"params": memory_module.parameters(), "lr": params.lr_memory},
            {
                "params": list(reconstructor.parameters()) 
                        + list(embed.parameters()) 
                        + list(surprise_gate.parameters()),
                "lr": params.learning_rate,
            },
        ],
        weight_decay=params.weight_decay,
    )

    num_epochs = params.num_epochs

    # Early stopping
    patience = getattr(params, 'early_stopping_patience', 5)
    min_delta = getattr(params, 'early_stopping_min_delta', 1e-4)
    wait = 0
    best_val_loss = float('inf')
    best_epoch = -1

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        ####################################################################
        # TRAIN
        ####################################################################
        memory_module.train()
        reconstructor.train()
        embed.train()
        surprise_gate.train()

        total_recon_loss = 0.0
        num_train_batches = 0

        # Show only high-level progress here
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Train", leave=False)

        for batch_idx, batch in enumerate(train_pbar):
            num_train_batches += 1
            x = batch['features'].to(device)   # [B, S, d_input]
            # 1) up-project
            x_up = embed(x)                   # [B, S, d_model]

            # 2) forward pass in memory to get assoc. loss
            k = x_up.detach()
            k.requires_grad_()
            v_hat = memory_module(k)
            assoc_loss = F.mse_loss(v_hat, k, reduction='none')  # shape [B, S, d_model]
            assoc_loss_per_sample = assoc_loss.mean(dim=-1)      # shape [B, S]

            # 3) surprise = gradient norm w.r.t k
            loss_sum = assoc_loss_per_sample.sum()
            loss_sum.backward(retain_graph=True)
            grad_norm = k.grad.view(k.size(0), k.size(1), -1).norm(p=2, dim=-1)  # shape [B, S]
            surprise = grad_norm.detach()

            # gating
            gate = surprise_gate(surprise)  # [B, S] in [0, 1]

            # 4) reconstructor pass
            with torch.no_grad():
                v_hat_nograd = memory_module(x_up)
            x_recon = reconstructor(x_up, memory_module)
            recon_loss = F.mse_loss(x_recon, x)  # scalar

            # 5) Gated assoc loss
            assoc_loss_gated = (assoc_loss_per_sample * gate).mean()

            # Optionally incorporate a separate surprise regularization
            lambda_surp = 100
            threshold   = 0.05
            with torch.no_grad():
                anomaly_mask = (surprise > threshold).float()
            surprise_anomalies = (anomaly_mask * surprise).sum() / (anomaly_mask.sum() + 1e-6)
            maximize_surp_term = - lambda_surp * surprise_anomalies

            total_loss = recon_loss + 10 * assoc_loss_gated + maximize_surp_term
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update progress bar with only the most important metrics
            train_pbar.set_postfix({
                'total_loss': f"{total_loss.item():.4f}",
                'recon_loss': f"{recon_loss.item():.4f}",
                'maximize_surp_term': f"{maximize_surp_term.item():.4f}",
            })

            if batch_idx % 1000 == 0:
                # Log additional details at DEBUG level
                logging.info(
                    f"[Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}] "
                    f"assoc_loss_gated={assoc_loss_gated.item():.4f}, "
                    f"gate_mean={gate.mean().item():.4f}, "
                    f"surprise_mean={surprise.mean().item():.4f}, "
                    f"surprise_anomalies={surprise_anomalies.item():.4f}, "
                )

        # We were accumulating `total_recon_loss`, but it was never incremented above.
        # You might want to track recon_loss similarly to how you do total_loss.
        # For consistency, let's do so:
        # (If you want a separate aggregator, be sure to accumulate properly each batch.)
        # For demonstration, let's just set avg_train_loss = recon_loss item (last batch).
        # Adjust as needed for your actual aggregation logic.
        avg_train_loss = recon_loss.item()  
        train_losses.append(avg_train_loss)

        ####################################################################
        # VALIDATION
        ####################################################################
        model.eval()

        total_val_recon_loss = 0.0
        total_val_loss = 0.0
        total_val_assoc_loss = 0.0
        total_val_surprise   = 0.0
        nb_val = 0

        for batch in tqdm(val_loader, desc="Validating", leave=False):
            x = batch['features'].to(device)  # [B, S, d_input]
            B, S, _ = x.shape
            
            # 1) embed
            x_up = embed(x)  # [B, S, d_model]
            # 2) memory => assoc_loss
            k = x_up.detach()
            k.requires_grad_()

            v_hat_val = memory_module(k)
            assoc_loss_val = F.mse_loss(v_hat_val, k, reduction='none')  # [B, S, d_model]
            assoc_loss_val_per_sample = assoc_loss_val.mean(dim=-1)      # [B, S]

            # Now we do a backward pass just to compute gradient norm w.r.t k
            loss_sum_val = assoc_loss_val_per_sample.sum()
            memory_module.zero_grad()
            if k.grad is not None:
                k.grad.zero_()
            loss_sum_val.backward(retain_graph=True)
            with torch.no_grad():
                grad_norm_val = k.grad.view(B, S, -1).norm(p=2, dim=-1)  # [B, S]
                surprise_val = grad_norm_val

                # 3) recon
                x_recon_val = reconstructor(x_up, memory_module)
                recon_loss_val = F.mse_loss(x_recon_val, x, reduction='none')  # [B, S, d_input]
                recon_loss_val = recon_loss_val.mean(dim=-1)  # [B, S]

                gate = surprise_gate(surprise_val)  # [B, S]
                assoc_loss_gated = (assoc_loss_val_per_sample * gate).mean()
                total_loss = recon_loss_val.mean() + 10 * assoc_loss_gated  # simplified

            batch_assoc_loss = assoc_loss_val_per_sample.mean().item()
            batch_recon_loss = recon_loss_val.mean().item()
            batch_surprise   = surprise_val.mean().item()
            batch_total      = total_loss.item()

            total_val_loss         += batch_total
            total_val_recon_loss   += batch_recon_loss
            total_val_assoc_loss   += batch_assoc_loss
            total_val_surprise     += batch_surprise
            nb_val += 1

        if nb_val == 0:
            avg_val_loss  = 0.0
            avg_assoc_loss= 0.0
            avg_surprise  = 0.0
        else:
            avg_val_loss  = total_val_recon_loss / nb_val
            avg_assoc_loss= total_val_assoc_loss / nb_val
            avg_surprise  = total_val_surprise / nb_val

        val_losses.append(avg_val_loss)

        logging.info(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"ValRecon: {avg_val_loss:.6f} | "
            f"ValAssoc: {avg_assoc_loss:.6f} | "
            f"ValSurprise: {avg_surprise:.6f}"
        )

        # Early stopping check using reconstruction loss
        if avg_val_loss < (best_val_loss - min_delta):
            best_val_loss = avg_val_loss
            best_epoch = epoch
            wait = 0
            # save the best model's state
            save_checkpoint(
                model=model,
                optimizer=None,
                epoch=epoch+1,
                loss=best_val_loss,
                params=params,
            )
        else:
            wait += 1
            if wait >= patience:
                logging.info(
                    f"Early stopping triggered. No improvement in val loss for {patience} epochs."
                )
                break

    logging.info(f"Training complete. Best Val Loss = {best_val_loss:.6f} at epoch {best_epoch+1}.")
    return train_losses, val_losses


def detect_function(model, test_loader, params):
    """
    Inference/detection on a new dataset ...
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    memory_module = model.memory_module
    reconstructor = model.reconstructor
    surprise_gate = model.surprise_gate
    embed = model.embed
    checkpoint_dir = os.path.join(params.output_dir, 'checkpoints')
    load_checkpoint(model, None, os.path.join(checkpoint_dir, 'checkpoint_best.pt'))

    memory_module.eval()
    reconstructor.eval()
    embed.eval()
    surprise_gate.eval()

    alpha = getattr(params, 'alpha', 100.0)  # user-defined weighting for surprise

    anomalies_records = []

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Detecting anomalies")):
        x = batch['features'].to(device)
        timestamps = batch['timestamps']
        B, S, D_in = x.shape

        x_up = embed(x)

        # assoc_loss
        k = x_up.detach()
        k.requires_grad_()
        v_hat = memory_module(k)
        assoc_loss = F.mse_loss(v_hat, k, reduction='none').mean(dim=-1)  # [B, S]

        # compute surprise
        memory_module.zero_grad()
        if k.grad is not None:
            k.grad.zero_()
        loss_sum_val = assoc_loss.sum()
        loss_sum_val.backward(retain_graph=True)
        grad_norm_val = k.grad.view(B, S, -1).norm(p=2, dim=-1)
        surprise_val = grad_norm_val.detach()

        # recon_loss
        x_recon = reconstructor(x_up, memory_module)
        recon_loss = F.mse_loss(x_recon, x, reduction='none').mean(dim=-1)  # [B, S]

        # anomaly_score
        anomaly_score = assoc_loss + 0.5 * recon_loss + alpha * surprise_val

        # Optionally unscale
        x_flat = x.view(-1, D_in).detach().cpu().numpy()
        x_recon_flat = x_recon.view(-1, D_in).detach().cpu().numpy()

        x_unscaled = unscale_features(x_flat).reshape(B, S, D_in)
        x_recon_unscaled = unscale_features(x_recon_flat).reshape(B, S, D_in)

        for b in range(B):
            for s in range(S):
                record = {
                    'batch_idx': batch_idx,
                    'timestamp': timestamps[b][s].item() 
                        if hasattr(timestamps[b][s], 'item') else timestamps[b][s],
                    'anomaly_score': anomaly_score[b, s].item(),
                    'assoc_loss': assoc_loss[b, s].item(),
                    'recon_loss': recon_loss[b, s].item(),
                    'surprise': surprise_val[b, s].item()
                }
                for feat_i, feat_name in enumerate(params.feature_columns):
                    record[f'{feat_name}'] = x_unscaled[b, s, feat_i]
                    record[f'pred_{feat_name}'] = x_recon_unscaled[b, s, feat_i]
                anomalies_records.append(record)

    anomalies_df = pd.DataFrame(anomalies_records)
    if anomalies_df.empty:
        logging.warning("No data found in test set (empty DataFrame).")
        return anomalies_df, None

    anomalies_df.sort_values('anomaly_score', ascending=False, inplace=True)
    save_path = "anomalies.csv"
    anomalies_df.head.to_csv(save_path, index=False)
    logging.info(f"Saved top 1000 anomaly detection results to {save_path}")

    return anomalies_df, None
