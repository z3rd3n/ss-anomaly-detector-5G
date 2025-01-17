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

    #load_checkpoint(model, None, "/workspaces/thesis/detect/reportable/results/checkpoints/checkpoint_best.pt")
    
    model.eval()
    train_energy = []
    all_preds= []
    all_features = []
    all_timestamps = []
    criterion_mse = torch.nn.MSELoss(reduction='none')
    softmax = torch.nn.Softmax(dim=-1)


    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Detecting anomalies")):
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

    train_energy = np.concatenate(train_energy, axis=0).reshape(-1)
    print(f"train_energy has length {len(train_energy)} and stats: "
             f"min={train_energy.min() if len(train_energy) else None}, "
             f"max={train_energy.max() if len(train_energy) else None}, "
             f"mean={train_energy.mean() if len(train_energy) else None}")
    all_features = np.concatenate(all_features, axis=0).reshape(-1, len(params.feature_columns))
    all_preds = np.concatenate(all_preds, axis=0).reshape(-1, len(params.feature_columns))
   
    # Calculate threshold using EVT
    threshold = calculate_threshold_evt(train_energy, params.q, params.p)
    print(f"Threshold derived from q={params.q}, p={params.p} => {threshold:.4f}")
    
    anomaly_indices = np.where(train_energy >= threshold)[0]
    logging.info(f"Total anomaly count: {len(anomaly_indices)} from {len(train_energy)} ratio {len(anomaly_indices)/len(train_energy)} (threshold={threshold})")

    if len(anomaly_indices) > 0:
        unscaled = unscale_features(all_features[anomaly_indices,:])
        predicted = unscale_features(all_preds[anomaly_indices,:])
        anomalies = pd.DataFrame({
            'timestamp_str': [all_timestamps[i] for i in anomaly_indices],
            'SFN': unscaled[:, 0].astype(int),
            'SFN_p': predicted[:, 0].astype(int),
            'Slot': unscaled[:, 1].astype(int),
            'Slot_p': predicted[:, 1].astype(int),
            'CC': unscaled[:, 2].astype(int),
            'CC_p': predicted[:, 2].astype(int),
            'HARQ': unscaled[:, 3].astype(int),
            'HARQ_p': predicted[:, 3].astype(int),
            'MCS': unscaled[:, 4].astype(int),
            'MCS_p': predicted[:, 4].astype(int),
            'CRC': unscaled[:, 5].astype(int),
            'CRC_p': predicted[:, 5].astype(int),
            'ReTx': unscaled[:, 6].astype(int),
            'ReTx_p': predicted[:, 6].astype(int),
            'NDI': unscaled[:, 7].astype(int),
            'NDI_p': predicted[:, 7].astype(int),
            'threshold': threshold,
            'anomaly_score': train_energy[anomaly_indices],  # Add anomaly scores,
            'distance_from_threshold': (train_energy[anomaly_indices] - threshold)
        })
        
        # Sort by anomaly score in descending order
        anomalies_df = anomalies.sort_values('anomaly_score', ascending=False)
        logging.info("Anomalies detected.")
        #anomalies_df.to_csv(f"{params.output_dir}_p{params.p}q{str(params.q)[2:]}.csv", index=False)

    if anomalies_df.empty:
        logging.warning("No anomalies found (empty DataFrame).")
        assert len(train_energy) == 0, "If anomalies_df is empty, all_scores should be empty too."
    else:
        anomalies_df = anomalies_df.sort_values("anomaly_score", ascending=False)
        logging.info(f"Found {len(anomalies_df)} anomalies.")
    # Sort by anomaly_score desc
    anomalies_df = anomalies_df.sort_values("anomaly_score", ascending=False)
    logging.info(f"Found {len(anomalies_df)} anomalies out of {len(train_energy)} data points.")

    fig, ax = plt.subplots()
    ax.plot(train_energy, label="scores")
    ax.axhline(threshold, color='red', label=f"Threshold (p={params.p}, q={params.q})")
    fig_path = os.path.join(params.output_dir, f"anomalies_p{params.p}q{str(params.q)[-2:]}_val{params.validation_ratio*100}.png")
    fig.savefig(fig_path)
    plt.close(fig)
    return anomalies_df, fig_path

def detect_anomalies_from_threshold(params, model, val_loader, threshold=0.05):

    model.eval()
    criterion_mse = torch.nn.MSELoss(reduction='none')
    softmax = torch.nn.Softmax(dim=-1)

    # Prepare an empty anomalies DataFrame up front
    anomalies_df = pd.DataFrame(
        columns=[
            'timestamp_str', 'SFN', 'SFN_p', 'Slot', 'Slot_p', 'CC', 'CC_p', 
            'HARQ', 'HARQ_p', 'MCS', 'MCS_p', 'CRC', 'CRC_p', 'ReTx', 
            'ReTx_p', 'NDI', 'NDI_p', 'threshold', 'anomaly_score', 
            'distance_from_threshold'
        ]
    )

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Detecting anomalies")):
            features = batch['features'].to(params.device)
            timestamps = batch['timestamps']
            timestamps_flat = [item for sublist in timestamps for item in sublist]
            # Forward pass
            enc_out, queries_list, keys_list = model(features)

            # Per-window reconstruction loss
            rec_loss = criterion_mse(enc_out, features).mean(dim=-1)

            # Calculate SACon from all layers
            loss_attn = 0.0
            for q, k in zip(queries_list, keys_list):
                loss_attn += model.compute_sub_adj_contrib(q, k, params.span, params.one_side)
            loss_attn /= len(queries_list)

            # Combined score
            train_score = softmax(-loss_attn) * rec_loss
            train_score_cpu = train_score.cpu().numpy().reshape(-1)

            # Identify indices above threshold for this batch
            anomaly_indices = np.where(train_score_cpu >= threshold)[0]
            if len(anomaly_indices) > 0:
                # Extract anomalies from CPU side
                batch_features = features.cpu().numpy().reshape(-1, features.shape[-1])
                batch_preds = enc_out.cpu().numpy().reshape(-1, features.shape[-1])

                # Unscale only the anomalies
                unscaled = unscale_features(batch_features[anomaly_indices, :])
                predicted = unscale_features(batch_preds[anomaly_indices, :])

                # Build per-batch anomalies DataFrame
                local_anomalies = pd.DataFrame({
                    'timestamp_str': [timestamps_flat[row] for row in anomaly_indices],
                    'SFN':  unscaled[:, 0].astype(int),
                    'SFN_p': predicted[:, 0].astype(int),
                    'Slot': unscaled[:, 1].astype(int),
                    'Slot_p': predicted[:, 1].astype(int),
                    'CC': unscaled[:, 2].astype(int),
                    'CC_p': predicted[:, 2].astype(int),
                    'HARQ': unscaled[:, 3].astype(int),
                    'HARQ_p': predicted[:, 3].astype(int),
                    'MCS': unscaled[:, 4].astype(int),
                    'MCS_p': predicted[:, 4].astype(int),
                    'CRC': unscaled[:, 5].astype(int),
                    'CRC_p': predicted[:, 5].astype(int),
                    'ReTx': unscaled[:, 6].astype(int),
                    'ReTx_p': predicted[:, 6].astype(int),
                    'NDI': unscaled[:, 7].astype(int),
                    'NDI_p': predicted[:, 7].astype(int),
                    'threshold': threshold,
                    'anomaly_score': train_score_cpu[anomaly_indices],
                    'distance_from_threshold': train_score_cpu[anomaly_indices] - threshold
                })

                # Append to global anomalies_df
                anomalies_df = pd.concat([anomalies_df, local_anomalies], ignore_index=True)

    # If no anomalies found, anomalies_df will be empty
    if anomalies_df.empty:
        logging.warning("No anomalies found (empty DataFrame).")
    else:
        # Sort final anomalies in descending order by anomaly_score
        anomalies_df = anomalies_df.sort_values("anomaly_score", ascending=False)
        logging.info(f"Found {len(anomalies_df)} anomalies in total.")

    anomalies_df.to_csv(f"temp_mlflow/anomalies_p{params.p}q{str(params.q)[2:]}v{int(params.validation_ratio * 100)}_th{threshold:.4f}.csv", index=False)

    return anomalies_df, None

