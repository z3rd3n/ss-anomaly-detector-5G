import os
import argparse
import logging
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import TwoPhaseModel
from dataset import ParquetSequenceDataset, custom_collate_fn
from torch.utils.data import DataLoader
from utils import start_logging

def load_model(model_path, config_path=None, device=None):
    """
    Load a trained model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to model configuration (optional)
        device: Device to load model to
    
    Returns:
        model: Loaded model
        config: Model configuration
    """
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Try to extract config from checkpoint
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        
        if 'params' in checkpoint:
            config = checkpoint['params']
        else:
            raise ValueError("No configuration found. Please provide a config file.")
    
    # Create model
    feature_ranges = config.get('feature_ranges', [1023, 30, 15, 32, 1, 8, 1])
    model = TwoPhaseModel(feature_ranges, config)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config

def run_inference(model, data_path, config, output_path=None, device=None):
    """
    Run inference on new data
    
    Args:
        model: Trained model
        data_path: Path to data file
        config: Model configuration
        output_path: Path to save results (optional)
        device: Device to run inference on
    
    Returns:
        results: Dictionary of results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f"Running inference on {data_path}")
    
    # Load feature statistics
    stats_file = config.get('features_stats_json', 'features_stats.json')
    
    if not os.path.exists(stats_file):
        logging.warning(f"Feature statistics file {stats_file} not found.")
        stats = None
    else:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            means = torch.tensor(stats['means'], dtype=torch.float32)
            variances = torch.tensor(stats['variances'], dtype=torch.float32)
            stats = {'means': means, 'variances': variances}
    
    # Create dataset
    dataset = ParquetSequenceDataset(
        parquet_path=data_path,
        feature_columns=config['feature_columns'],
        seq_len=config['seq_len'],
        stride=config.get('inference_stride', config['seq_len']),
        ratio=1.0,
        seed=config.get('seed', 42),
        skip_anomalies=False,
        normalization_stats=stats
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.get('inference_batch_size', 32),
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    # Run inference
    results = {'timestamps': [], 'predictions': [], 'confidence': [], 'reconstruction_error': []}
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference", unit="batch"):
            features = batch['features'].to(device)
            timestamps = batch['timestamps']
            
            # Get model outputs
            class_logits = model(features)
            
            # Get predictions and confidence
            probs = torch.softmax(class_logits, dim=-1)
            preds = torch.argmax(class_logits, dim=-1)
            confidence, _ = torch.max(probs, dim=-1)
            
            # Get reconstruction error (if available)
            try:
                embedded = model.feature_embedding(features)
                reconstructed, _ = model.autoencoder(embedded)
                recon_error = torch.mean(torch.abs(embedded - reconstructed), dim=-1)
            except:
                recon_error = torch.zeros_like(preds, dtype=torch.float32)
            
            # Store results
            for ts_list, pred_list, conf_list, err_list in zip(
                timestamps, preds.cpu(), confidence.cpu(), recon_error.cpu()
            ):
                for ts, pred, conf, err in zip(ts_list, pred_list, conf_list, err_list):
                    if ts:  # Skip empty timestamps (padding)
                        results['timestamps'].append(ts)
                        results['predictions'].append(pred.item())
                        results['confidence'].append(conf.item())
                        results['reconstruction_error'].append(err.item())
    
    # Map predictions to anomaly names
    anomaly_mapping = {
        0: "normal",
        1: "unnecessary_retx",
        2: "missing_retx",
        3: "new_data_no_retx",
        4: "max_retx_achieved",
        5: "none_of_them"
    }
    
    results['prediction_names'] = [anomaly_mapping.get(p, f"unknown_{p}") for p in results['predictions']]
    
    # Save results if output path provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': results['timestamps'],
            'prediction': results['predictions'],
            'prediction_name': results['prediction_names'],
            'confidence': results['confidence'],
            'reconstruction_error': results['reconstruction_error']
        })
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logging.info(f"Results saved to {output_path}")
        
        # Generate summary
        summary = df['prediction_name'].value_counts().to_dict()
        
        # Log summary
        logging.info("Inference summary:")
        for name, count in summary.items():
            logging.info(f"  {name}: {count}")
    
    return results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PDSCH Anomaly Detection Inference')
    
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, help='Path to model configuration')
    parser.add_argument('--input', type=str, required=True, help='Path to input data file')
    parser.add_argument('--output', type=str, help='Path to save results')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--log_dir', type=str, default='./inference_logs', help='Log directory')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.log_dir, 'inference.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.model, args.config, device)
    logging.info(f"Model loaded from {args.model}")
    
    # Set output path
    output_path = args.output
    if output_path is None:
        output_dir = os.path.join(args.log_dir, 'results')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'predictions.csv')
    
    # Run inference
    results = run_inference(model, args.input, config, output_path, device)
    
    # Print summary
    anomaly_count = sum(1 for p in results['predictions'] if p > 0)
    total_count = len(results['predictions'])
    anomaly_percentage = (anomaly_count / total_count) * 100 if total_count > 0 else 0
    
    logging.info(f"Inference complete. Found {anomaly_count} anomalies out of {total_count} instances ({anomaly_percentage:.2f}%)")
    
    return 0

if __name__ == "__main__":
    main()