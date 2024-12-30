# subAdjacent/trainer.py
import logging
import torch
from subAdjacent.run_epoch import train_one_epoch, validate_one_epoch
from utils import *
from tqdm import tqdm


def train_model(params, model, optimizer, scheduler, train_loader, val_loader):
    start_epoch = 0
    if params.checkpoint_path:
        start_epoch, prev_loss = load_checkpoint(
            model,
            optimizer,
            params.checkpoint_path,
            params.device
        )
        logging.info(f"Resuming training from epoch {start_epoch + 1} and from loss {prev_loss:.4f}")
        start_epoch += 1

    # 4) training
    best_val_loss = float('inf')
    not_improved_count = 0
    early_stop_patience = 5

    all_train_losses = []
    all_val_losses = []

    criterion_mse = torch.nn.MSELoss(reduction='none')

    for epoch in range(start_epoch, params.num_epochs):
        logging.info(f"===== Epoch {epoch+1}/{params.num_epochs} =====")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            params.device,
            criterion_mse,
            params.span,
            params.one_side,
            lambda_sacon=params.k_value  # or some param name
        )
        val_rec_loss, val_total_loss = validate_one_epoch(
            model,
            val_loader,
            params.device,
            criterion_mse,
            params.span,
            params.one_side,
            lambda_sacon=params.k_value
        )

        all_train_losses.append(train_loss)
        all_val_losses.append(val_total_loss)

        logging.info(
            f"Epoch [{epoch+1}/{params.num_epochs}] | "
            f"TrainLoss: {train_loss:.4f} | ValRec: {val_rec_loss:.4f} | ValTotal: {val_total_loss:.4f}"
        )

        #scheduler.step(val_total_loss)

        # Early stopping
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

    plot_training_curves(all_train_losses, all_val_losses, params.output_dir)
    logging.info("Training finished.")



def detect_anomalies(params, model, optimizer, val_loader):

    load_last_checkpoint(params, model, optimizer)
    
    all_scores = []
    all_rec = []
    all_sacon = []
    all_features = []
    all_timestamps = []

    criterion_mse = torch.nn.MSELoss(reduction='none')

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Detecting anomalies")):
            if params.debug and i >= 50:
                break
            features = batch['features'].to(params.device)
            timestamps_list = batch['timestamps']

            enc_out, queries_list, keys_list = model(features)
            # rec_loss shape = [B, seq_len, D], we reduce feature-dim => [B, seq_len]
            rec_loss = criterion_mse(enc_out, features).mean(dim=-1) #it means over features

            # SACon (averaged across layers)
            sacon_all_layers = torch.zeros_like(rec_loss)
            for (q, k_) in zip(queries_list, keys_list):
                sacon_all_layers += model.compute_sub_adj_contrib(q, k_, params.span, params.one_side)
            sacon_all_layers /= len(queries_list)

            # flatten
            rec_loss_flat = rec_loss.view(-1).cpu().numpy() # shape [B,L] => [B*L]
            sacon_flat = sacon_all_layers.view(-1).cpu().numpy() # shape [B,L] => [B*L]

            # store
            all_rec.append(rec_loss_flat)
            all_sacon.append(sacon_flat)
            all_features.append(features.view(-1, features.shape[-1]).cpu().numpy())
            all_timestamps.extend([t for sublist in timestamps_list for t in sublist])

    # concat
    all_rec = np.concatenate(all_rec, axis=0)
    all_sacon = np.concatenate(all_sacon, axis=0)
    all_features = np.concatenate(all_features, axis=0)

    # anomaly_score = rec_loss * softmax(-sacon)
    # We'll do a single global softmax across all points for simplicity:
    negative_sacon = -all_sacon
    # be mindful of big shape => do it with stable code
    sacon_weights = np.exp(negative_sacon - negative_sacon.max())
    sacon_weights /= (sacon_weights.sum() + 1e-6)

    all_scores = all_rec * sacon_weights  # multiply elementwise

    # threshold
    threshold = calculate_threshold_evt(all_scores)
    anomalies_mask = (all_scores > threshold)

    # Save to CSV (unscale -> int)
    unscale_and_save_anomalies(
        timestamps=all_timestamps,
        features=all_features,
        anomaly_scores=all_scores,
        threshold=threshold,
        output_csv=os.path.join(params.output_dir, "detected_anomalies.csv")
    )

    # Plot
    plot_anomalies(all_scores, anomalies_mask, threshold, params.output_dir)
    plot_attention_matrices(model, val_loader, params.device, params.output_dir)

    logging.info("Anomaly detection complete.")
