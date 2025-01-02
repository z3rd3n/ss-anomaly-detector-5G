# subAdjacentLSTM/run_epoch.py
import torch
from tqdm import tqdm
from subAdjacentLSTM.model.subAdjacentLSTMLoss import SubAdjacentLSTMLoss  


def train_one_epoch(model, dataloader, optimizer, 
                    criterion_mse, params):
    """
    Changes:
      - total_loss = rec_loss - lambda_sacon * sacon_loss
      - We skip "2.0 * rec_loss - k * loss_attn" approach.
    """
    model.train()
    total_loss = 0.0
    batch_count = 0
    criterion = SubAdjacentLSTMLoss(params.k1, params.k2, params.alpha)

    train_pbar = tqdm(dataloader, desc="Training", total=len(dataloader))
    for batch_idx, batch in enumerate(train_pbar):
        features = batch['features'].to(params.device)  # [B, seq_len, D]

        recon = model(features) # [B, seq_len, D], 

        loss, rec, sacon = criterion(
            features,
            recon,
            model.hidden_states
        )

        optimizer.zero_grad()

        loss.backward()
        
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        train_pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'rec': f"{rec.item():.4f}",
            'SACon': f"{sacon.item():.4f}"
        })

    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    return avg_loss


def validate_one_epoch(model, dataloader, criterion_mse, params):
    """
    Validation with the same objective. We return just the rec_loss or total_loss
    for logging.
    """
    model.eval()
    total_rec_loss = 0.0
    total_loss = 0.0
    batch_count = 0

    criterion = SubAdjacentLSTMLoss(params.k1, params.k2, params.alpha)

    val_pbar = tqdm(dataloader, desc="Validation", total=len(dataloader), leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_pbar):
            features = batch['features'].to(params.device)  # [B, seq_len, D]

            recon, att_weights = model(features) # [B, seq_len, D], [B, L, seq_len, seq_len]

            recon = model(features) # [B, seq_len, D], [B, L, seq_len, seq_len]

            loss, rec, sacon = criterion(
                features,
                recon,
                model.stored_states['hidden_states'],
                model.stored_states['cell_states']
            )

            val_pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'rec': f"{rec.item():.4f}",
            'SACon': f"{sacon.item():.4f}"
        })

    avg_rec_loss = total_rec_loss / batch_count if batch_count > 0 else 0.0
    avg_total_loss = total_loss / batch_count if batch_count > 0 else 0.0
    return avg_rec_loss, avg_total_loss