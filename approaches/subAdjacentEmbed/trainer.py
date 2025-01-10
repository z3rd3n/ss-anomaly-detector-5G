# subAdjacent/trainer.py
import logging
import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from subAdjacentEmbed.run_epoch import train_one_epoch, validate_one_epoch
from subAdjacentEmbed.model.dataEmbedding import compute_categorical_reconstruction_loss
from subAdjacentEmbed.model.anomalyTransformer import CategoricalDecoder
from utils import *
from tqdm import tqdm
import torch.nn.functional as F


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
    criterion = compute_categorical_reconstruction_loss


    for epoch in range(start_epoch, params.num_epochs):
        logging.info(f"===== Epoch {epoch+1}/{params.num_epochs} =====")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            params.device,
            criterion,
            params.feature_config,
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
            criterion,
            params.feature_config,
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
        #anomalies_df.to_csv(f"{params.output_dir}_p{params.p}q{str(params.q)[-2:]}.csv", index=False)

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
    fig_path = os.path.join(params.output_dir, f"anomalies_p{params.p}q{str(params.q)[-2:]}.png")
    fig.savefig(fig_path)
    plt.close(fig)
    return anomalies_df, fig_path

def detect_and_categorical_anomalies(params, model, val_loader):
    """
    Enhanced anomaly detection and reporting for categorical features
    """
    model.eval()
    all_scores = []
    all_predictions = {feat: [] for feat in params.feature_config.keys()}
    all_originals = {feat: [] for feat in params.feature_config.keys()}
    all_timestamps = []
    
    decoder = CategoricalDecoder(params.feature_config, params.model_dim).to(params.device)
    softmax = torch.nn.Softmax(dim=-1)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Detecting anomalies"):
            features = batch['features'].to(params.device)
            timestamps = batch['timestamps']
            
            # Get model outputs
            enc_out, queries_list, keys_list = model(features)
            
            # Decode predictions back to categorical probabilities
            decoded_preds = decoder(enc_out)
            
            # Calculate reconstruction loss per feature
            rec_loss = 0
            for feat_name, pred_probs in decoded_preds.items():
                true_indices = features[..., params.feature_config[feat_name]["feature_idx"]].long()
                feat_loss = F.cross_entropy(
                    pred_probs.reshape(-1, pred_probs.size(-1)),
                    true_indices.reshape(-1),
                    weight=torch.tensor([
                        params.feature_config[feat_name]["class_weights"].get(str(i), 1.0)
                        for i in range(len(params.feature_config[feat_name]["value_to_index"]))
                    ]).to(params.device),
                    reduction='none'
                ).reshape(features.shape[0], -1)
                rec_loss += feat_loss
            
            # Calculate attention-based anomaly contribution
            loss_attn = 0.0
            for q, k in zip(queries_list, keys_list):
                loss_attn += model.compute_sub_adj_contrib(q, k, params.span, params.one_side)
            loss_attn /= len(queries_list)
            
            # Compute final anomaly scores
            anomaly_scores = softmax(-loss_attn) * rec_loss
            all_scores.append(anomaly_scores.cpu().numpy())
            
            # Store predictions and original values
            for feat_name, pred_probs in decoded_preds.items():
                # Get most likely class
                pred_classes = pred_probs.argmax(dim=-1)
                true_classes = features[..., params.feature_config[feat_name]["feature_idx"]].long()
                
                # Convert indices back to original values
                idx_to_value = params.feature_config[feat_name]["index_to_value"]
                pred_values = torch.tensor([float(idx_to_value[str(idx.item())]) for idx in pred_classes.flatten()])
                true_values = torch.tensor([float(idx_to_value[str(idx.item())]) for idx in true_classes.flatten()])
                
                all_predictions[feat_name].append(pred_values.numpy())
                all_originals[feat_name].append(true_values.numpy())
            
            all_timestamps.extend([t for sublist in timestamps for t in sublist])
    
    # Concatenate all scores and values
    all_scores = np.concatenate(all_scores, axis=0).reshape(-1)
    for feat_name in all_predictions:
        all_predictions[feat_name] = np.concatenate(all_predictions[feat_name])
        all_originals[feat_name] = np.concatenate(all_originals[feat_name])
    
    # Calculate threshold using EVT
    threshold = calculate_threshold_evt(all_scores, params.q, params.p)
    print(f"Threshold derived from q={params.q}, p={params.p} => {threshold:.4f}")
    
    # Find anomalous points
    anomaly_indices = np.where(all_scores >= threshold)[0]
    logging.info(f"Total anomaly count: {len(anomaly_indices)} from {len(all_scores)} "
                f"ratio {len(anomaly_indices)/len(all_scores)} (threshold={threshold})")
    
    if len(anomaly_indices) > 0:
        # Create anomaly report DataFrame
        anomaly_data = {
            'timestamp_str': [all_timestamps[i] for i in anomaly_indices],
            'threshold': threshold,
            'anomaly_score': all_scores[anomaly_indices],
            'distance_from_threshold': (all_scores[anomaly_indices] - threshold)
        }
        
        # Add original and predicted values for each feature
        for feat_name in params.feature_config:
            anomaly_data[feat_name] = all_originals[feat_name][anomaly_indices]
            anomaly_data[f"{feat_name}_p"] = all_predictions[feat_name][anomaly_indices]
            
            # Add confidence scores for predictions
            if f"{feat_name}_conf" not in anomaly_data:
                anomaly_data[f"{feat_name}_conf"] = np.max(
                    F.softmax(torch.tensor(all_predictions[feat_name][anomaly_indices]), dim=-1).numpy(),
                    axis=-1
                )
        
        anomalies_df = pd.DataFrame(anomaly_data)
        
        # Sort by anomaly score in descending order
        anomalies_df = anomalies_df.sort_values('anomaly_score', ascending=False)
        
        # Save results
        output_path = os.path.join(
            params.output_dir, 
            f"anomalies_p{params.p}q{str(params.q)[-2:]}_val{params.validation_ratio*100}.csv"
        )
        anomalies_df.to_csv(output_path, index=False)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot anomaly scores
        ax1.plot(all_scores, label="Anomaly Scores")
        ax1.axhline(threshold, color='red', linestyle='--', label=f"Threshold (p={params.p}, q={params.q})")
        ax1.set_title("Anomaly Scores Over Time")
        ax1.legend()
        
        # Plot feature-wise anomaly distribution
        feature_anomaly_counts = {
            feat: np.sum(np.abs(all_originals[feat][anomaly_indices] - 
                               all_predictions[feat][anomaly_indices]) > 0)
            for feat in params.feature_config
        }
        ax2.bar(feature_anomaly_counts.keys(), feature_anomaly_counts.values())
        ax2.set_title("Anomaly Distribution Across Features")
        ax2.set_xticklabels(feature_anomaly_counts.keys(), rotation=45)
        
        plt.tight_layout()
        fig_path = os.path.join(
            params.output_dir,
            f"anomalies_p{params.p}q{str(params.q)[-2:]}_val{params.validation_ratio*100}.png"
        )
        fig.savefig(fig_path)
        plt.close(fig)
        
        return anomalies_df, fig_path
    else:
        logging.warning("No anomalies found.")
        return pd.DataFrame(), None