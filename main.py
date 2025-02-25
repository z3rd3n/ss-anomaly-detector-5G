import os
import sys
import argparse
import logging
import torch
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Import components
from dataset import BinaryAnomalyDataset, create_binary_dataloader
from model import TimeSeriesAnomalyDetector
from trainer import BinaryAnomalyTrainer, train_binary_model, load_and_evaluate_model

def start_logging(params=None):
    """
    Initialize logging.
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = params['output_dir'] if params and 'output_dir' in params else 'output'
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'log_training_{current_time}.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    if params:
        logging.info("Hyperparameters and settings:")
        for key, value in params.items():
            logging.info(f"{key}: {value}")
    return log_file

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PDSCH Binary Anomaly Detection')
    
    # Data paths
    parser.add_argument('--train_path', type=str, default='unscaled_pdsch_val.parquet',
                        help='Path to training parquet file')
    parser.add_argument('--val_path', type=str, default='unscaled_pdsch_val_min.parquet',
                        help='Path to validation parquet file (optional)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output files')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation set ratio if no validation file provided')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension of the model')
    parser.add_argument('--embedding_dim', type=int, default=8,
                        help='Embedding dimension for categorical features')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers in the model')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--focal_alpha', type=float, default=0.75,
                        help='Alpha parameter for focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss')
    
    # Sequence parameters
    parser.add_argument('--seq_len', type=int, default=100,
                        help='Sequence length')
    parser.add_argument('--stride', type=int, default=50,
                        help='Stride between sequences')
    parser.add_argument('--overlap_ratio', type=float, default=0.5,
                        help='Overlap ratio between sequences')
    
    # Load/save model
    parser.add_argument('--load_model', type=str, default='',
                        help='Path to load a trained model (skip training)')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate the model, do not train')
    
    # Debug/testing options
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use (for testing/debugging)')
    
    # Misc
    parser.add_argument('--use_cuda', action='store_true',
                        help='Use CUDA if available')
    
    return parser.parse_args()

def get_default_params():
    """Get default parameters"""
    return {
        'parquet_path': 'unscaled_pdsch_val.parquet',
        'validation_parquet_path': 'unscaled_pdsch_val_min.parquet',
        'output_dir': 'output',
        'feature_columns': ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"],
        'numerical_features': ['SFN', 'Slot', 'MCS', 'ReTx'],
        'categorical_features': ['HARQ', 'CRC', 'NDI'],
        'seq_len': 100,
        'stride': 50,
        'overlap_ratio': 0.5,
        'train_ratio': 1.0,
        'val_ratio': 1.0,
        'seed': 42,
        'batch_size': 64,
        'num_epochs': 20,
        'lr': 5e-4,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'embedding_dim': 8,
        'hidden_dim': 128,
        'latent_dim': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'focal_alpha': 0.75,
        'focal_gamma': 2.0,
        'use_balanced_sampling': True,
        'use_state_cache': True,
        'pre_normalize': True,
        'num_workers': 4,
        'checkpoint_interval': 5,
        'features_stats_json': 'features_stats.json',
        'anomaly_threshold': 0.5,
        'max_samples': None
    }


def ensure_feature_stats(params):
    """
    Ensure feature statistics file exists or create dummy stats
    """
    stats_file = params.get('features_stats_json', 'feature_stats.json')
    
    if not os.path.exists(stats_file):
        logging.warning(f"Feature statistics file {stats_file} not found, creating default stats")
        
        # Create default stats based on numerical features
        numerical_features = params.get('numerical_features', ['SFN', 'Slot', 'MCS', 'ReTx'])
        
        # Default stats (means=0, stds=1)
        stats = {
            'means': [0.0] * len(numerical_features),
            'stds': [1.0] * len(numerical_features)
        }
        
        # Save to file
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logging.info(f"Created default feature statistics at {stats_file}")

def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Get default parameters and update with arguments
    params = get_default_params()
    
    # Set up logging
    start_logging(params)
    
    # Ensure feature statistics file exists
    ensure_feature_stats(params)
    
    # Set random seeds for reproducibility
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    
    # Set device
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed(params['seed'])
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}")
    
    # Load model or train a new one
    if args.load_model or args.eval_only:
        # Load existing model
        model_path = args.load_model
        if not model_path and args.eval_only:
            # Look for best_model.pt in output directory
            model_path = os.path.join(params['output_dir'], 'best_model.pt')
            if not os.path.exists(model_path):
                logging.error(f"No model found at {model_path} and --eval_only specified")
                return
                
        logging.info(f"Loading model from {model_path}")
        model, trainer, _ = load_and_evaluate_model(model_path, params, device)
        
    else:
        # Train a new model
        logging.info("Training new model...")
        model, trainer, best_metrics = train_binary_model(params)
        logging.info(f"Training complete. Best metrics: {best_metrics}")
    
    logging.info("Process complete!")

if __name__ == "__main__":
    main()