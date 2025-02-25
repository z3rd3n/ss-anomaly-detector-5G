import os
import logging
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import SequenceStateCacheDataset, custom_collate_fn
from utils import EPS

def run_inference(model, test_data_path, params, device):
    """
    Run inference on test data and return anomaly predictions.
    
    Args:
        model: Trained model
        test_data_path: Path to test data parquet file
        params: Dictionary of parameters
        device: Device to run inference on
        
    Returns:
        DataFrame with original data and anomaly predictions
    """
    logging.info(f"Running inference on {test_data_path}")
    
    # Load normalization stats
    if 'features_stats_json' in params and os.path.exists(params['features_stats_json']):
        import json
        with open(params['features_stats_json'], 'r') as f:
            stats = json.load(f)
        means = torch.tensor(stats['means'], dtype=torch.float32)
        variances = torch.tensor(stats['variances'], dtype=torch.float32)
    else:
        logging.warning("No feature statistics found. Using zeros for means and ones for variances.")
        means = torch.zeros(len(params['feature_columns']), dtype=torch.float32)
        variances = torch.ones(len(params['feature_columns']), dtype=torch.float32)
    
    # Create test dataset
    test_dataset = SequenceStateCacheDataset(
        parquet_path=test_data_path,
        feature_columns=params['feature_columns'],
        seq_len=params['seq_len'],
        stride=params.get('stride', None),
        ratio=1.0,  # Use all test data
        seed=params['seed'],
        skip_anomalies=False,
        normalization_stats={'means': means, 'variances': variances},
        use_state_cache=params.get('use_state_cache', True),
        overlap_ratio=params.get('overlap_ratio', 0.5)
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=params.get('num_workers', 4),
        pin_memory=True
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dictionaries to store predictions by timestamp
    pred_labels_by_ts = {}
    anomaly_scores_by_ts = {}
    anomaly_probs_by_ts = {}
    
    # Inference loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference", unit="batch"):
            features = batch['features'].to(device)
            timestamps = batch['timestamps']  # List of lists of timestamps
            
            # Get model predictions and anomaly scores
            class_logits = model(features)
            
            # For anomaly score, use the get_anomaly_score method if available
            if hasattr(model, 'get_anomaly_score'):
                anomaly_scores, _ = model.get_anomaly_score(features)
                anomaly_scores = anomaly_scores.cpu()
            else:
                # Otherwise, use the negative probability of the normal class
                class_probs = F.softmax(class_logits, dim=-1)
                anomaly_scores = -torch.log(class_probs[:, :, 0] + EPS).cpu()
            
            # Get class predictions
            pred_labels = torch.argmax(class_logits, dim=-1).cpu()
            
            # Calculate probability of being an anomaly (any non-zero class)
            class_probs = F.softmax(class_logits, dim=-1)
            normal_probs = class_probs[:, :, 0]
            anomaly_probs = 1.0 - normal_probs
            
            # Store predictions by timestamp
            for b in range(len(timestamps)):
                for t in range(len(timestamps[b])):
                    ts = timestamps[b][t]
                    pred_labels_by_ts[ts] = pred_labels[b, t].item()
                    anomaly_scores_by_ts[ts] = anomaly_scores[b, t].item()
                    anomaly_probs_by_ts[ts] = anomaly_probs[b, t].cpu().item()
    
    # Load original test data to add predictions
    test_df = pd.read_parquet(test_data_path)
    
    # Map timestamp strings to rows
    if 'timestamp_str' in test_df.columns:
        ts_column = 'timestamp_str'
    else:
        logging.warning("No timestamp_str column found. Creating a synthetic one.")
        test_df['timestamp_str'] = [f"ts_{i}" for i in range(len(test_df))]
        ts_column = 'timestamp_str'
    
    # Create reverse mapping from anomaly class to label
    anomaly_mapping = {v: k for k, v in test_dataset.anomaly_mapping.items()}
    anomaly_mapping[0] = "normal"
    
    # Add predictions to dataframe
    test_df['predicted_label'] = test_df[ts_column].map(
        lambda ts: pred_labels_by_ts.get(ts, 0))
    
    test_df['predicted_anomaly'] = test_df['predicted_label'].map(
        lambda label: anomaly_mapping.get(label, "unknown"))
    
    test_df['anomaly_score'] = test_df[ts_column].map(
        lambda ts: anomaly_scores_by_ts.get(ts, 0.0))
    
    test_df['anomaly_probability'] = test_df[ts_column].map(
        lambda ts: anomaly_probs_by_ts.get(ts, 0.0))
    
    # Add flag for any anomaly
    test_df['is_anomaly'] = test_df['predicted_label'] > 0
    
    # Log summary of predictions
    anomaly_counts = test_df['predicted_anomaly'].value_counts()
    logging.info("Prediction summary:")
    for anomaly_type, count in anomaly_counts.items():
        logging.info(f"  {anomaly_type}: {count} instances ({count/len(test_df)*100:.2f}%)")
    
    logging.info(f"Total anomalies detected: {test_df['is_anomaly'].sum()} "
                f"({test_df['is_anomaly'].sum()/len(test_df)*100:.2f}%)")
    
    return test_df

def run_inference_with_states(model, test_data_path, params, device):
    """
    Run inference on test data with stateful processing for HARQ IDs.
    
    This version maintains state between non-overlapping chunks of data
    for better handling of temporal dependencies.
    
    Args:
        model: Trained model
        test_data_path: Path to test data parquet file
        params: Dictionary of parameters
        device: Device to run inference on
        
    Returns:
        DataFrame with original data and anomaly predictions
    """
    logging.info(f"Running stateful inference on {test_data_path}")
    
    # Load normalization stats
    if 'features_stats_json' in params and os.path.exists(params['features_stats_json']):
        import json
        with open(params['features_stats_json'], 'r') as f:
            stats = json.load(f)
        means = torch.tensor(stats['means'], dtype=torch.float32)
        variances = torch.tensor(stats['variances'], dtype=torch.float32)
    else:
        logging.warning("No feature statistics found. Using zeros for means and ones for variances.")
        means = torch.zeros(len(params['feature_columns']), dtype=torch.float32)
        variances = torch.ones(len(params['feature_columns']), dtype=torch.float32)
    
    # Load test data
    test_df = pd.read_parquet(test_data_path)
    
    # Ensure timestamp column exists
    if 'timestamp_str' not in test_df.columns:
        logging.warning("No timestamp_str column found. Creating a synthetic one.")
        test_df['timestamp_str'] = [f"ts_{i}" for i in range(len(test_df))]
    
    # Track HARQ states
    harq_states = {}
    
    # Create predictions dataframe
    predictions = []
    
    # Process data in chunks by HARQ ID
    harq_groups = test_df.groupby('HARQ')
    
    for harq_id, group in tqdm(harq_groups, desc="Processing HARQ groups"):
        # Extract features
        features = torch.tensor(group[params['feature_columns']].values, dtype=torch.float32)
        
        # Normalize features
        std = torch.sqrt(variances + EPS)
        features = (features - means) / std
        
        # Process in chunks of seq_len
        for start_idx in range(0, len(features), params['seq_len']):
            end_idx = min(start_idx + params['seq_len'], len(features))
            chunk_features = features[start_idx:end_idx]
            
            # Pad if needed
            if chunk_features.size(0) < params['seq_len']:
                padding = torch.zeros(params['seq_len'] - chunk_features.size(0), 
                                    chunk_features.size(1), 
                                    dtype=chunk_features.dtype)
                chunk_features = torch.cat([chunk_features, padding], dim=0)
            
            # Add batch dimension
            chunk_features = chunk_features.unsqueeze(0).to(device)
            
            # Get initial state for this HARQ ID if available
            prev_state = harq_states.get(harq_id, None)
            
            # Forward pass with state tracking
            with torch.no_grad():
                if hasattr(model, 'inference') and callable(model.inference):
                    # Use stateful inference if available
                    output, new_state = model.inference(chunk_features, prev_state)
                    class_logits = output
                else:
                    # Fallback to standard forward pass
                    class_logits = model(chunk_features)
                    new_state = None
            
            # Update state
            if new_state is not None:
                harq_states[harq_id] = new_state
            
            # Get predictions
            pred_labels = torch.argmax(class_logits, dim=-1).cpu().numpy()[0]
            class_probs = F.softmax(class_logits, dim=-1).cpu().numpy()[0]
            
            # Store predictions only for actual data (not padding)
            actual_length = min(params['seq_len'], end_idx - start_idx)
            chunk_indices = group.iloc[start_idx:end_idx].index
            
            for i, idx in enumerate(chunk_indices):
                if i >= actual_length:
                    break
                    
                predictions.append({
                    'index': idx,
                    'predicted_label': pred_labels[i],
                    'normal_prob': class_probs[i, 0],
                    'anomaly_prob': 1.0 - class_probs[i, 0]
                })
    
    # Convert predictions to dataframe and merge with test data
    pred_df = pd.DataFrame(predictions)
    pred_df.set_index('index', inplace=True)
    
    # Merge with original data
    result_df = test_df.copy()
    result_df = result_df.join(pred_df, how='left')
    
    # Fill missing predictions (if any)
    result_df['predicted_label'] = result_df['predicted_label'].fillna(0).astype(int)
    result_df['normal_prob'] = result_df['normal_prob'].fillna(1.0)
    result_df['anomaly_prob'] = result_df['anomaly_prob'].fillna(0.0)
    
    # Create anomaly type mapping
    anomaly_mapping = {0: "normal"}
    for anomaly_name, anomaly_id in test_dataset.anomaly_mapping.items():
        anomaly_mapping[anomaly_id] = anomaly_name
    
    # Add predicted anomaly type
    result_df['predicted_anomaly'] = result_df['predicted_label'].map(
        lambda x: anomaly_mapping.get(x, "unknown"))
    
    # Add anomaly score (negative log probability of normal class)
    result_df['anomaly_score'] = -np.log(result_df['normal_prob'] + EPS)
    
    # Add flag for any anomaly
    result_df['is_anomaly'] = result_df['predicted_label'] > 0
    
    # Log summary of predictions
    anomaly_counts = result_df['predicted_anomaly'].value_counts()
    logging.info("Prediction summary:")
    for anomaly_type, count in anomaly_counts.items():
        logging.info(f"  {anomaly_type}: {count} instances ({count/len(result_df)*100:.2f}%)")
    
    logging.info(f"Total anomalies detected: {result_df['is_anomaly'].sum()} "
                f"({result_df['is_anomaly'].sum()/len(result_df)*100:.2f}%)")
    
    return result_df