import torch
from tqdm import tqdm
import logging

def train_one_epoch(model, dataloader, optimizer, device, 
                    criterion_mse, lambda_sacon=10.0):
    """
    Changes:
      - total_loss = rec_loss - lambda_sacon * sacon_loss
      - We skip "2.0 * rec_loss - k * loss_attn" approach.
    """
    model.train()
    total_loss = 0.0
    batch_count = 0

    train_pbar = tqdm(dataloader, desc="Training", total=len(dataloader))
    for batch_idx, batch in enumerate(train_pbar):
        features = batch['features'].to(device)  # [B, seq_len, D]
        enc_out, queries_list, keys_list = model(features)

        # Reconstruction loss
        rec_loss = criterion_mse(enc_out, features)

        # Compute sub-adj attention contribution (SACon) from last layer 
        # or average across layers. We'll do the average across layers:
        sacon_all_layers = 0.0
        for (q, k_) in zip(queries_list, keys_list):
            sacon_all_layers += model.compute_sub_adj_contrib(q, k_)
        sacon_all_layers /= len(queries_list)  # shape [B, L]

        # For training, we might want a single scalar => average over batch & seq
        sacon_mean = sacon_all_layers.mean()

        # total_loss = rec_loss.mean() - lambda_sacon * sacon_mean
        # (assuming rec_loss is also an average scalar)
        # If rec_loss is shape [B, seq_len, D], let's do mean over B, seq_len, D:
        rec_loss_scalar = rec_loss.mean()

        # Final total loss
        loss = rec_loss_scalar - lambda_sacon * sacon_mean

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        # Logging
        if (batch_idx + 1) % 50 == 0:
            logging.info(
                f"Batch {batch_idx+1} => "
                f"RecLoss: {rec_loss_scalar.item():.4f}, "
                f"SACon: {sacon_mean.item():.4f}, "
                f"TotalLoss: {loss.item():.4f}"
            )

        train_pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'rec': f"{rec_loss_scalar.item():.4f}",
            'SACon': f"{sacon_mean.item():.4f}"
        })

    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    return avg_loss


def validate_one_epoch(model, dataloader, device, criterion_mse, lambda_sacon=10.0):
    """
    Validation with the same objective. We return just the rec_loss or total_loss
    for logging.
    """
    model.eval()
    total_rec_loss = 0.0
    total_loss = 0.0
    batch_count = 0

    val_pbar = tqdm(dataloader, desc="Validation", total=len(dataloader), leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_pbar):
            features = batch['features'].to(device)
            enc_out, queries_list, keys_list = model(features)

            # reconstruction loss
            rec_loss = criterion_mse(enc_out, features).mean()

            # SACon
            sacon_all_layers = 0.0
            for (q, k_) in zip(queries_list, keys_list):
                sacon_all_layers += model.compute_sub_adj_contrib(q, k_)
            sacon_all_layers /= len(queries_list)
            sacon_mean = sacon_all_layers.mean()

            # total loss
            loss_val = rec_loss - lambda_sacon * sacon_mean

            total_rec_loss += rec_loss.item()
            total_loss += loss_val.item()
            batch_count += 1

            val_pbar.set_postfix({
                'val_rec': f"{rec_loss.item():.4f}",
                'val_loss': f"{loss_val.item():.4f}"
            })

    avg_rec_loss = total_rec_loss / batch_count if batch_count > 0 else 0.0
    avg_total_loss = total_loss / batch_count if batch_count > 0 else 0.0
    return avg_rec_loss, avg_total_loss