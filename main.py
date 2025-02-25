import os
import argparse
import logging
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from model import TwoPhaseModel
from trainer import main_training_pipeline
from utils import start_logging, load_checkpoint
from inference import run_inference

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PDSCH Telecommunication Anomaly Detection')
    
    # General settings
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'inference'],
                        help='Operation mode: train, evaluate, or inference')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save outputs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Data settings
    parser.add_argument('--train_data', type=str, default='data/unscaled_pdsch_train.parquet',
                        help='Path to training data parquet file')
    parser.add_argument('--val_data', type=str, default='data/unscaled_pdsch_val.parquet',
                        help='Path to validation data parquet file')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Path to test data parquet file (for inference mode)')
    parser.add_argument('--feature_columns', type=str, nargs='+', 
                        default=["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"],
                        help='Feature columns to use from parquet files')
    parser.add_argument('--feature_ranges', type=int, nargs='+', 
                        default=[1023, 30, 15, 32, 1, 8, 1],
                        help='Value ranges for each feature')
    
    # Model parameters
    parser.add_argument('--seq_len', type=int, default=100,
                        help='Sequence length for temporal modeling')
    parser.add_argument('--stride', type=int, default=50,
                        help='Stride between sequences (use with overlap_ratio)')
    parser.add_argument('--overlap_ratio', type=float, default=0.5,
                        help='Ratio of overlap between consecutive sequences')
    parser.add_argument('--embedding_dim', type=int, default=8,
                        help='Dimension for feature embedding')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for model')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Latent dimension for VAE')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers in RNN components')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping threshold')
    parser.add_argument('--focal_alpha', type=float, default=0.75,
                        help='Alpha parameter for focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss')
    parser.add_argument('--vae_loss_weight', type=float, default=0.2,
                        help='Weight for VAE loss component')
    
    # Balance strategies
    parser.add_argument('--use_balanced_sampling', action='store_true',
                        help='Use balanced sampling of classes')
    parser.add_argument('--oversample_ratio', type=float, default=0.8,
                        help='Ratio for oversampling minority classes')
    parser.add_argument('--undersample_ratio', type=float, default=0.3,
                        help='Ratio for undersampling majority class')
    parser.add_argument('--use_state_cache', action='store_true',
                        help='Use state caching between sequences for same HARQ ID')
    
    # Model loading
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load a pretrained model checkpoint')
    
    return parser.parse_args()

def setup_environment(args):
    """Set up environment for training/evaluation."""
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Start logging
    start_logging({key: value for key, value in vars(args).items()})
    
    return args

def create_model(args, device):
    """Create model based on args."""
    model = TwoPhaseModel(
        feature_ranges=args.feature_ranges,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_classes=5,  # Fixed for this problem
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    # Load pretrained model if specified
    if args.load_model and os.path.exists(args.load_model):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        _, _, _ = load_checkpoint(args.load_model, model, optimizer)
        logging.info(f"Loaded pretrained model from {args.load_model}")
    
    model.to(device)
    return model

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Set up environment
    args = setup_environment(args)
    
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create model
    model = create_model(args, device)
    
    # Convert args to params dictionary for compatibility with existing code
    params = {key: value for key, value in vars(args).items()}
    params['parquet_path'] = args.train_data
    params['validation_parquet_path'] = args.val_data
    params['features_stats_json'] = os.path.join(args.output_dir, 'features_stats.json')
    
    # Execute based on mode
    if args.mode == 'train':
        model, params = main_training_pipeline(params, model, train=True)
        logging.info("Training completed.")
        
    elif args.mode == 'evaluate':
        model, params = main_training_pipeline(params, model, train=False)
        logging.info("Evaluation completed.")
        
    elif args.mode == 'inference':
        if args.test_data is None:
            logging.error("Test data path must be provided for inference mode.")
            return
            
        # Run inference
        results = run_inference(model, args.test_data, params, device)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(args.output_dir, f'inference_results_{timestamp}.csv')
        results.to_csv(results_file, index=False)
        logging.info(f"Inference results saved to {results_file}")
    
    logging.info("Process completed successfully.")

if __name__ == "__main__":
    main()