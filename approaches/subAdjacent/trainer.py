# subAdjacent/trainer.py
import logging
import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
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
        )
        logging.info(f"Resuming training from epoch {start_epoch + 1} and from loss {prev_loss:.4f}")
        start_epoch += 1

    # 4) training
    best_val_loss = float('inf')
    not_improved_count = 0
    early_stop_patience = 3

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
            lamda_rec=params.lamda_rec,
            lambda_sacon=params.k_value,  # or some param name
            max_grad_norm=params.max_grad_norm
        )
        val_rec_loss, val_total_loss = validate_one_epoch(
            model,
            val_loader,
            params.device,
            criterion_mse,
            params.span,
            params.one_side,
            lamda_rec=params.lamda_rec,
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


def detect_anomalies(params, model, val_loader):
    
    model.eval()
    all_preds= []
    all_features = []
    all_timestamps = []
    criterion_mse = torch.nn.MSELoss(reduction='none')
    softmax = torch.nn.Softmax(dim=-1)

    with torch.no_grad():
        train_energy = []
        for batch in tqdm(val_loader, desc="Detecting anomalies"):
            features = batch['features'].to(params.device)
            timestamps = batch['timestamps']
            
            # Get model outputs
            enc_out, queries_list, keys_list = model(features)
            
            # Per-window reconstruction loss 
            rec_loss = criterion_mse(enc_out, features).mean(dim=-1)
            loss_attn = 0.0
            # Calculate SACon from all layers
            for q, k in zip(queries_list, keys_list):
                loss_attn += model.compute_sub_adj_contrib(q, k, params.span, params.one_side)
            loss_attn /= len(queries_list)
            
            train_score = softmax(-loss_attn) * rec_loss
            train_energy.append(train_score.cpu().numpy())
            all_features.append(features.cpu().numpy())
            all_preds.append(enc_out.cpu().numpy())
            all_timestamps.extend([t for sublist in timestamps for t in sublist])

    all_scores = np.concatenate(train_energy, axis=0).reshape(-1)
    all_features = np.concatenate(all_features, axis=0).reshape(-1, len(params.feature_columns))
    all_preds = np.concatenate(all_preds, axis=0).reshape(-1, len(params.feature_columns))
   
    # Calculate threshold using EVT
    threshold = calculate_threshold_evt(all_scores, params.q, params.p)
    logging.info(f"Threshold derived from q={params.q}, p={params.p} => {threshold:.4f}")
    
    anomalies_mask = all_scores > threshold
    # Build a DataFrame of anomalies
    anomalies_list = []
    for idx, is_anom in enumerate(anomalies_mask):
        if is_anom:
            row_dict = {
                "timestamp_str": all_timestamps[idx],
                "anomaly_score": all_scores[idx],
                "distance_from_threshold": all_scores[idx] - threshold,
                "threshold": threshold,
            }

            unscaled = unscale_features(all_features[idx:idx+1, :])[0]
            unscaled_p = unscale_features(all_preds[idx:idx+1, :])[0]
            # Just label them as col, col_p
            for i, feat_name in enumerate(params.feature_columns):
                row_dict[f"{feat_name}"] = unscaled[i]
                row_dict[f"{feat_name}_p"] = unscaled_p[i]
            anomalies_list.append(row_dict)

    anomalies_df = pd.DataFrame(anomalies_list)
    # Sort by anomaly_score desc
    anomalies_df = anomalies_df.sort_values("anomaly_score", ascending=False)
    logging.info(f"Found {len(anomalies_df)} anomalies out of {len(all_scores)} data points.")
    return anomalies_df
