import torch
import numpy as np
import os
import logging

class Hyperparameters:
    def __init__(self):
        self.seed=42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.train = True
        self.detect_anomalies = False
        self.num_heads = 4
        self.num_layers = 2
        self.dropout = 0.1
        self.seq_len = 32
        self.stride = 32
        self.batch_size = 64
        self.num_epochs = 50
        self.validation_ratio=0.2
        self.learning_rate = 1e-4
        self.parquet_path = "data/scaled_pdsch.parquet"
        self.pretrain= None
        self.feature_columns = [
            'SFN', 'Slot', 'CC', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI',
        ]
        self.output_dir = "results"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_path = None  # Path to load checkpoint from

        # Model architecture params
        self.model_dim = 512       # d_model
        self.n_heads = 8          # number of attention heads
        self.e_layers = 3         # number of encoder layers
        self.d_ff = 512          # dimension of feed-forward network
        self.activation = 'gelu'  # activation function
        self.k_value = 0.5       # trade-off parameter for loss

        # Training specific params
        self.shuffle_files = True
        self.output_attention = True
        self.detect_anomalies = False

        # System params
        self.num_workers = 4
        self.pin_memory = True

def start_logging(params):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_file = 'results/logTraining.log'
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a' if params.train and params.pretrain is not None else 'w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def save_checkpoint(model, optimizer, epoch, loss, params):
    """Save model checkpoint to the output directory."""
    checkpoint_path = os.path.join(params.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    logging.info(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint from file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    logging.info(f"Loaded checkpoint from epoch {epoch+1} with loss {loss:.4f}")
    return epoch, loss