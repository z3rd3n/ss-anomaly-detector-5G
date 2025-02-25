import os
import argparse
import logging
import json
from trainer import run_training_pipeline
from model import TwoPhaseModel
from utils import start_logging, validate_csv
import torch

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PDSCH Anomaly Detection Training')
    
    # Basic options
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--validate', action='store_true', help='Validate the model')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    
    # Data options
    parser.add_argument('--parquet_path', type=str, help='Path to training parquet file')
    parser.add_argument('--validation_parquet_path', type=str, help='Path to validation parquet file')
    parser.add_argument('--seq_len', type=int, help='Sequence length')
    parser.add_argument('--stride', type=int, help='Stride for sequence generation')
    
    # Training options
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--ae_epochs', type=int, help='Number of autoencoder training epochs')
    parser.add_argument('--cls_epochs', type=int, help='Number of classifier training epochs')
    parser.add_argument('--ae_lr', type=float, help='Autoencoder learning rate')
    parser.add_argument('--cls_lr', type=float, help='Classifier learning rate')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    
    # Model options
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension size')
    parser.add_argument('--latent_dim', type=int, help='Latent dimension size')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    
    # Sampling options
    parser.add_argument('--no_oversampling', action='store_true', help='Disable oversampling')
    
    return parser.parse_args()

def load_config(config_path, args):
    """
    Load configuration from file and override with command line arguments
    """
    # Default configuration
    config = {
        # Data paths
        'parquet_path': 'unscaled_pdsch_train.parquet',
        'validation_parquet_path': 'unscaled_pdsch_val.parquet',
        'output_dir': './output',
        'features_stats_json': './output/features_stats.json',
        
        # Feature configuration
        'feature_columns': ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"],
        'feature_ranges': [1023, 30, 15, 32, 1, 8, 1],
        
        # Dataset parameters
        'seq_len': 100,
        'stride': 50,  # Overlapping sequences to catch boundary anomalies
        'val_stride': 100,
        'train_ratio': 1.0,
        'val_ratio': 1.0,
        'seed': 42,
        'batch_size': 32,
        'num_workers': 4,
        
        # Oversampling configuration
        'use_oversampling': True,
        'oversample_factors': {
            1: 5.0,   # unnecessary_retx: moderate oversampling
            2: 20.0,  # missing_retx: heavy oversampling (very rare)
            3: 5.0,   # new_data_no_retx: moderate oversampling
            4: 20.0,  # max_retx_achieved: heavy oversampling (very rare)
        },
        
        # Model parameters
        'embedding_dim': 16,
        'hidden_dim': 128,
        'latent_dim': 64,
        'dropout': 0.3,
        'num_anomaly_classes': 5,  # 4 types + "none of them"
        
        # Autoencoder training parameters
        'ae_epochs': 15,
        'ae_lr': 1e-3,
        'ae_weight_decay': 1e-6,
        'use_l1_reg': True,
        'l1_lambda': 1e-5,
        
        # Classifier training parameters
        'cls_epochs': 25,
        'cls_lr': 5e-4,
        'cls_weight_decay': 1e-6,
        'focal_gamma': 2.5,
        'focal_alpha': 0.25,
        
        # Fine-tuning parameters
        'do_fine_tuning': True,
        'ft_epochs': 5,
        'ft_lr': 1e-5,
        'ft_weight_decay': 1e-6,
        
        # Training control
        'use_lr_scheduler': True,
        'clip_grad': True,
        'max_grad_norm': 1.0,
        'train': True,
        'use_cuda': True,
    }
    
    # Load from config file if exists
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # Override with command line arguments
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if value is not None and key in config:
            if key == 'no_cuda' and value:
                config['use_cuda'] = False
            elif key == 'no_oversampling' and value:
                config['use_oversampling'] = False
            elif value is not None:
                config[key] = value
    
    # Ensure output directories exist
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Update features_stats_json path to be in output_dir
    if 'features_stats_json' not in arg_dict or arg_dict['features_stats_json'] is None:
        config['features_stats_json'] = os.path.join(config['output_dir'], 'features_stats.json')
    
    return config

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config, args)
    
    # Start logging
    start_logging(config)
    
    # Log configuration
    logging.info("Configuration:")
    for key, value in config.items():
        logging.info(f"  {key}: {value}")
    
    # Set training mode
    if args.train:
        config['train'] = True
    elif args.validate:
        config['train'] = False
    
    # Run the training pipeline
    model, metrics = run_training_pipeline(config)
    
    # Log results
    logging.info("Final results:")
    logging.info(f"Metrics: {metrics}")
    
    return 0

if __name__ == "__main__":
    main()