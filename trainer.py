# trainer.py
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import logging

from data.dataLoader import ParquetSequenceDataset, custom_collate_fn
from subAdjacent.anomalyTransformer import AnomalyTransformer
from utils import Hyperparameters, start_logging, save_checkpoint, load_checkpoint
import torch.nn.functional as F
from tqdm import tqdm

# Import scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def lossSAcon(queries, keys, span=None, one_side=True):
    """
    This version computes an attention matrix from queries/keys,
    sums diagonals within a certain span, and returns B x L.
    """
    # queries, keys: [B, L, H, D]
    L = queries.shape[1]
    if span is None:
        span = [20, 30]

    # Build attention matrix: (B,H,L,L)
    attnMatrix = torch.einsum("b l h e, b s h e -> b h l s", queries, keys)
    attnMatrix = attnMatrix / (attnMatrix.sum(dim=-1, keepdim=True) + 1e-6)

    # Sum diagonals in range(span[0], span[1])
    lossMat = None
    for k in range(-span[1], span[1] + 1):
        if one_side and k < span[0]:
            # skip negative offsets if one_side is True
            continue

        diag1 = torch.diagonal(attnMatrix, offset=k, dim1=-2, dim2=-1)
        if k >= 0:
            diag1 = F.pad(diag1, (k, 0))
        else:
            diag1 = F.pad(diag1, (0, abs(k)))

        if lossMat is None:
            lossMat = diag1
        else:
            lossMat += diag1

    # mean over heads dimension => [B, L]
    lossMat = torch.mean(lossMat, dim=1)
    return lossMat  # shape [B, L]


#####################################
# Training loop
#####################################
def train_one_epoch(model, dataloader, optimizer, device, 
                    criterion_mse, k=0.5):
    """
    Train for one epoch, returning average loss.
    """
    model.train()
    total_loss = 0
    batch_count = 0

    # Wrap dataloader with tqdm for a nice progress bar
    train_pbar = tqdm(dataloader, desc="Training", total=len(dataloader))
    for batch_idx, batch in enumerate(train_pbar):
        features = batch['features'].to(device)  # shape [B, seq_len, input_dim]
        
        # Forward pass => model returns: (enc_out, queries_list, keys_list)
        enc_out, queries_list, keys_list = model(features)
        
        # Reconstruction loss
        rec_loss = criterion_mse(enc_out, features)

        # Compute attention-based “association discrepancy”
        loss_attn = 0.0
        for (q, k_) in zip(queries_list, keys_list):
            loss_attn += lossSAcon(q, k_).mean()
        loss_attn = loss_attn / len(queries_list)

        # Combine losses
        loss = 2.0 * rec_loss - k * loss_attn
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        if (batch_idx + 1) % 50 == 0:
            logging.info(
                f"Batch {batch_idx+1} => rec_loss: {rec_loss.item():.4f}, "
                f"attn_loss: {loss_attn.item():.4f}, total_loss: {loss.item():.4f}"
            )

        # Update the tqdm postfix so you can see current loss, etc.
        train_pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'rec_loss': f"{rec_loss.item():.4f}",
            'attn_loss': f"{loss_attn.item():.4f}"
        })

    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    return avg_loss


def validate_one_epoch(model, dataloader, device, criterion_mse, k=0.5):
    """
    Validate for one epoch, returning (avg_reconstruction_loss, avg_total_loss).
    """
    model.eval()
    total_rec_loss = 0
    total_loss = 0
    batch_count = 0

    val_pbar = tqdm(dataloader, desc="Validation", total=len(dataloader), leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_pbar):
            features = batch['features'].to(device)
            
            enc_out, queries_list, keys_list = model(features)
            rec_loss = criterion_mse(enc_out, features)
            
            loss_attn = 0.0
            for (q, k_) in zip(queries_list, keys_list):
                loss_attn += lossSAcon(q, k_).mean()
            loss_attn = loss_attn / len(queries_list)
            
            total_loss_val = 2.0 * rec_loss - k * loss_attn

            total_rec_loss += rec_loss.item()
            total_loss += total_loss_val.item()
            batch_count += 1

            val_pbar.set_postfix({
                'batch': batch_idx + 1,
                'val_loss': f"{total_loss_val.item():.4f}"
            })

    avg_rec_loss = total_rec_loss / batch_count if batch_count > 0 else 0.0
    avg_total_loss = total_loss / batch_count if batch_count > 0 else 0.0
    return avg_rec_loss, avg_total_loss


def train_model():
    # ====== Initialize hyperparameters, logging, etc. ====== #
    params = Hyperparameters()
    start_logging(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("===== Hyperparameters =====")
    for attr, value in vars(params).items():
        logging.info(f"{attr}: {value}")

    # ====== Create train/validation datasets and loaders ====== #
    train_dataset, val_dataset = ParquetSequenceDataset.create_train_val_splits(
        parquet_path=params.parquet_path,
        feature_columns=params.feature_columns,
        seq_len=params.seq_len,
        validation_ratio=params.validation_ratio,
        seed=params.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=params.pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=params.pin_memory,
        drop_last=True
    )

    # ====== Initialize model, loss function, optimizer ====== #
    model = AnomalyTransformer(
        enc_in=len(params.feature_columns),  # input dim
        c_out=len(params.feature_columns),   # reconstruct same dim
        d_model=params.model_dim,
        n_heads=params.n_heads,
        e_layers=params.e_layers,
        d_ff=params.d_ff,
        dropout=params.dropout,
        activation=params.activation,
        output_attention=True,
    ).to(device)

    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # -- LR Scheduler (ReduceLROnPlateau) --
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,      # reduce LR by half
        patience=3,      # wait 'patience' epochs with no improvement
        min_lr=1e-6,     # do not reduce below this learning rate
        verbose=True
    )

    # Optionally load checkpoint
    start_epoch = 0
    if params.checkpoint_path:
        start_epoch, prev_loss = load_checkpoint(
            model,
            optimizer,
            params.checkpoint_path,
            device
        )
        logging.info(f"Resuming training from epoch {start_epoch + 1}")
        start_epoch += 1

    # ====== Start Training ====== #
    best_val_loss = float('inf')
    not_improved_count = 0  # For early stopping
    early_stop_patience = 5

    for epoch in range(start_epoch, params.num_epochs):
        logging.info(f"===== Epoch {epoch+1}/{params.num_epochs} =====")

        # --- Train ---
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion_mse,
            k=params.k_value
        )

        # --- Validate ---
        val_rec_loss, val_total_loss = validate_one_epoch(
            model,
            val_loader,
            device,
            criterion_mse,
            k=params.k_value
        )

        logging.info(
            f"Epoch [{epoch+1}/{params.num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Recons Loss: {val_rec_loss:.4f} | "
            f"Val Total Loss: {val_total_loss:.4f}"
        )

        # Step the scheduler based on the validation loss
        scheduler.step(val_total_loss)

        # Early Stopping and best checkpoint saving
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            not_improved_count = 0
            save_checkpoint(model, optimizer, epoch, val_total_loss, params)
        else:
            not_improved_count += 1
            logging.info(f"No improvement. Early stopping count: {not_improved_count}/{early_stop_patience}")
            if not_improved_count >= early_stop_patience:
                logging.info("Early stopping triggered.")
                break

    logging.info("Training finished.")


if __name__ == "__main__":
    train_model()
