import os
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

# For PCA, t-SNE, and plotting.
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# A constant used by the rule–based flag (if needed)
MAX_RETX = 5

###################################
# 1. HELPER FUNCTIONS (logging, checkpointing, rule–based flags)
###################################

def rule_based_flags(sequence: torch.Tensor) -> torch.Tensor:
    """
    Apply rule-based checks to flag anomalous timesteps in the sequence.
    Returns a boolean tensor (shape [seq_len]) with True indicating a rule violation.
    
    The following rules are applied:
      - "Unnecessary ReTx": If ReTx > 0 and the previous CRC was 1.
      - "Missing ReTx": If ReTx == 0 and the previous CRC was 0 and the current NDI equals the previous NDI.
      - "New Data but No ReTx": If the previous CRC is 0 and the current NDI is different from the previous NDI.
      - "Max ReTx Achieved": If ReTx >= MAX_RETX.
    """
    seq = sequence.cpu().numpy()  # shape (seq_len, 7)
    seq_len = seq.shape[0]
    flags = np.zeros(seq_len, dtype=bool)
    for t in range(seq_len):
        current_harq = seq[t, 2]
        t_prev = None
        for candidate in range(t - 1, -1, -1):
            if seq[candidate, 2] == current_harq:
                t_prev = candidate
                break
        if t_prev is None:
            continue
        prev_crc = seq[t_prev, 4]
        curr_crc = seq[t, 4]
        curr_ret = seq[t, 5]
        curr_ndi = seq[t, 6]
        prev_ndi = seq[t_prev, 6]
        if (curr_ret > 0 and prev_crc == 1) or \
           (curr_ret == 0 and prev_crc == 0 and (curr_ndi == prev_ndi)) or \
           (prev_crc == 0 and (curr_ndi != prev_ndi)) or \
           (curr_ret >= MAX_RETX):
            flags[t] = True
    return torch.tensor(flags, dtype=torch.bool)

def start_logging(params=None):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = params['output_dir'] if params and 'output_dir' in params else 'output'
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'logTraining_{current_time}.log')
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

def save_checkpoint(state: dict, filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    logging.info(f"Checkpoint saved to {filename}")

def load_checkpoint(filename: str, model: nn.Module, optimizer: torch.optim.Optimizer):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        best_f1 = checkpoint.get('best_f1', 0.0)
        params = checkpoint.get('params', {})
        logging.info(f"Loaded checkpoint '{filename}' (epoch {epoch}) with best F1: {best_f1:.4f}")
        return epoch, best_f1, params
    else:
        logging.warning(f"No checkpoint found at '{filename}'")
        return 0, 0.0, {}

###################################
# 2. MODEL DEFINITION (VARIATIONAL AUTOENCODER)
###################################

class VariationalCategoricalAutoencoder(nn.Module):
    def __init__(self, vocab_sizes, embedding_dims, hidden_dim, num_layers, dropout=0.1):
        """
        vocab_sizes: list of ints, e.g. [1024, 31, 16, 33, 2, 9, 2]
        embedding_dims: list of ints, e.g. [16, 8, 4, 8, 2, 4, 2]
        hidden_dim: hidden dimension for the LSTM layers (and latent code)
        num_layers: number of LSTM layers
        dropout: dropout probability for LSTM layers
        """
        super().__init__()
        assert len(vocab_sizes) == len(embedding_dims), "Mismatch between vocab sizes and embedding dims"
        self.num_features = len(vocab_sizes)
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim)
            for vocab_size, emb_dim in zip(vocab_sizes, embedding_dims)
        ])
        self.total_emb_dim = sum(embedding_dims)
        # Encoder LSTM processes the concatenated embeddings.
        self.encoder = nn.LSTM(
            input_size=self.total_emb_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout
        )
        # For VAE: layers to compute the latent distribution parameters.
        self.fc_mu = nn.Linear(hidden_dim, hidden_dim)
        self.fc_logvar = nn.Linear(hidden_dim, hidden_dim)
        # Decoder LSTM: it will decode from the latent code.
        # We feed the latent code (repeated over seq_len) as input.
        self.decoder = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout
        )
        # Output heads: one per feature to predict the categorical distribution.
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size) for vocab_size in vocab_sizes
        ])

    def forward(self, x, return_latents=False):
        """
        x: tensor of shape [batch, seq_len, num_features] containing integer values.
        If return_latents is True, returns a tuple (outputs, (z, mu, logvar)),
          where outputs is a list of tensors (one per feature) of shape [batch, seq_len, vocab_size],
          and (z, mu, logvar) are the latent code, mean and log-variance.
        Otherwise, returns outputs only.
        """
        batch_size, seq_len, num_features = x.size()
        embedded_feats = []
        for i in range(num_features):
            emb = self.embeddings[i](x[:, :, i])
            embedded_feats.append(emb)
        # Concatenate embeddings along the last dimension: shape [batch, seq_len, total_emb_dim]
        x_emb = torch.cat(embedded_feats, dim=-1)
        # Pass through the encoder LSTM.
        encoded, _ = self.encoder(x_emb)
        # Use the encoder's last time-step representation to compute latent distribution parameters.
        latent_input = encoded[:, -1, :]  # shape: [batch, hidden_dim]
        mu = self.fc_mu(latent_input)
        logvar = self.fc_logvar(latent_input)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon  # Reparameterization trick.
        # Create decoder input: repeat z for each time step.
        dec_input = z.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(dec_input)
        # For each feature, produce logits for each time step.
        outputs = [head(decoded) for head in self.output_heads]
        if return_latents:
            return outputs, (z, mu, logvar)
        return outputs

###################################
# 3. ANOMALY SCORING, PCA/t-SNE, AND VALIDATION FUNCTIONS
###################################

def compute_anomaly_scores(model, batch, vocab_sizes, device, kl_weight=1.0):
    """
    Given a batch (with key 'features') and the trained model,
    returns a tensor of anomaly scores of shape [batch, seq_len].
    The anomaly score for each timestep is the sum of the per-feature cross-entropy losses
    plus a contribution from the KL divergence computed from the latent distribution.
    """
    features = batch['features'].to(device)  # [batch, seq_len, num_features]
    outputs, (z, mu, logvar) = model(features, return_latents=True)
    batch_size, seq_len, _ = features.shape
    scores = torch.zeros(batch_size, seq_len, device=device)
    for i in range(len(vocab_sizes)):
        logits = outputs[i]  # Shape: [batch, seq_len, vocab_sizes[i]]
        logits_flat = logits.reshape(-1, vocab_sizes[i])
        targets_flat = features[:, :, i].reshape(-1)
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        ce_loss = ce_loss.reshape(batch_size, seq_len)
        scores += ce_loss
    # Compute KL divergence per sample (scalar per sequence)
    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # shape: [batch]
    # Expand KL divergence to add to each timestep in the sequence.
    kl_div = kl_div.unsqueeze(1).expand(-1, seq_len)
    kl_score = kl_weight * kl_div
    scores += kl_score
    return scores, kl_score

def plot_latent_space(model, val_loader, params, device):
    """
    Extracts latent representations from the validation loader and plots them using PCA.
    Uses the sampled latent vector 'z' for visualization.
    """
    model.eval()
    all_latents = []
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            _, latents = model(features, return_latents=True)
            z, mu, logvar = latents  # Use z for plotting.
            all_latents.append(z.cpu().numpy())
    if not all_latents:
        logging.warning("No latent representations found for plotting.")
        return
    all_latents = np.concatenate(all_latents, axis=0)
    # PCA
    n_components = params.get('pca_n_components', 3)
    pca = PCA(n_components=n_components)
    latent_pca = pca.fit_transform(all_latents)
    explained_variance = pca.explained_variance_ratio_
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(latent_pca[:, 0], latent_pca[:, 1], latent_pca[:, 2],
                    c=latent_pca[:, 0], cmap='viridis', s=3)
    ax.set_title("Latent Space PCA (3 Components)")
    ax.set_xlabel(f"PC1 ({explained_variance[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained_variance[1]*100:.1f}% var)")
    ax.set_zlabel(f"PC3 ({explained_variance[2]*100:.1f}% var)")
    plt.savefig(os.path.join(params['output_dir'], 'latent_space_plots.png'))
    plt.close()
    logging.info("Saved latent space plots (PCA).")

def produce_anomalies(model, val_loader, params, device, vocab_sizes):
    """
    Iterates through the validation loader, computing anomaly scores per timestep.
    Returns a DataFrame of anomalies (with timestamps, anomaly scores, true and predicted features)
    and the threshold used.
    """
    model.eval()
    all_scores = []
    anomaly_candidates = []
    kl_weight = params.get('kl_beta', 1.0)  # Use the same weight as during training.
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f"Validation", unit="batch") as val_bar:
            for batch in val_loader:
                recon_scores, kl_score = compute_anomaly_scores(model, batch, vocab_sizes, device, kl_weight=kl_weight)
                scores = recon_scores

                batch_timestamps = batch['timestamps']
                features_tensor = batch['features'].to(device)
                outputs = model(features_tensor)
                predicted_list = []
                for i in range(len(vocab_sizes)):
                    pred_i = torch.argmax(outputs[i], dim=-1)
                    predicted_list.append(pred_i.detach().cpu().numpy())
                predicted_array = np.stack(predicted_list, axis=-1)
                features_np = features_tensor.cpu().numpy()
                scores_np = scores.detach().cpu().numpy()
                kl_scores_np = kl_score.detach().cpu().numpy()

                for i in range(scores_np.shape[0]):
                    for j in range(scores_np.shape[1]):
                        score = scores_np[i, j]
                        kl = kl_scores_np[i,j]
                        candidate = {
                            'timestamp': batch_timestamps[i][j],
                            'anomaly_score': score,
                            'kl': kl,
                            'true_features': features_np[i, j].tolist(),
                            'predicted_features': predicted_array[i, j].tolist()
                        }
                        all_scores.append(score)
                        anomaly_candidates.append(candidate)
                val_bar.update(1)
    if len(all_scores) == 0:
        threshold = 0.0
    else:
        threshold = np.percentile(all_scores, params['percentile'])
    anomalies = [candidate for candidate in anomaly_candidates if candidate['anomaly_score'] > threshold]
    anomalies_df = pd.DataFrame(anomalies)
    plot_latent_space(model, val_loader, params, device)
    return anomalies_df, threshold

def validate_csv(model, params, device, vocab_sizes):
    """
    Validates the model on CSV data by computing anomaly scores and comparing against ground truth.
    Also extracts and plots the latent representations.
    """
    logging.info("Starting CSV validation procedure...")
    val_dataset = CSVValidationDataset(
        csv_path=params['validation_csv_path'],
        seq_len=params['seq_len'],
        stride=params['stride']
    )
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=custom_collate_fn)

    anomalies_df, _ = produce_anomalies(model, val_loader, params, device, vocab_sizes)
    if anomalies_df.empty:
        logging.warning("No anomalies detected during CSV validation!")
        return 0.0, 0.0, 0.0
    anomalies_df = anomalies_df.drop_duplicates(subset=['timestamp'])
    df_gt = pd.read_csv(params['ground_truth_csv_path'])
    gt_col = "timestamp_str" if "timestamp_str" in df_gt.columns else "timestamp"
    gt_timestamps = set(df_gt[gt_col].unique())
    anomalies_df["is_accurate"] = anomalies_df["timestamp"].apply(lambda ts: ts in gt_timestamps)
    num_accurate = anomalies_df["is_accurate"].sum()
    total_anoms = len(anomalies_df)
    total_gt = len(gt_timestamps)
    precision = (num_accurate / total_anoms) if total_anoms > 0 else 0.0
    recall = (num_accurate / total_gt) if total_gt > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    logging.info(f"Total anomalies reported by model: {total_anoms}")
    logging.info(f"Total ground truth anomalies: {total_gt}")
    logging.info(f"Total accurate detections: {num_accurate}")
    logging.info(f"[RESULT] CSV Validation - Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1-Score: {f1_score*100:.2f}%")
    anomalies_df = anomalies_df.sort_values(by='anomaly_score', ascending=False)
    anomalies_csv_path = os.path.join(params['output_dir'], 'sorted_anomalies.csv')
    anomalies_df.to_csv(anomalies_csv_path, index=False)
    logging.info(f"Sorted anomalies saved to {anomalies_csv_path}")
    return precision, recall, f1_score

###################################
# 4. TRAINING FUNCTION (VARIATIONAL AUTOENCODER)
###################################

def train_anomaly_detector(model, train_loader, optimizer, device, params, vocab_sizes, checkpoint_path='best_model.pt'):
    num_epochs = params.get('num_epochs', 10)
    best_f1 = 0.0
    kl_beta = params.get('kl_beta', 1.0)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs}", unit="batch") as pbar:
            for batch in train_loader:
                # --- Check for anomalies in the batch ---
                skip_batch = False
                for sample in batch['features']:
                    # rule_based_flags returns a boolean tensor (shape: [seq_len])
                    if rule_based_flags(sample).any():
                        skip_batch = True
                        break
                if skip_batch:
                    pbar.update(1)
                    continue
                # --- End anomaly check ---
                optimizer.zero_grad()
                features = batch['features'].to(device)
                # Forward pass with latent outputs.
                outputs, (z, mu, logvar) = model(features, return_latents=True)
                recon_loss = 0.0
                # Compute reconstruction loss for each feature.
                for i in range(len(vocab_sizes)):
                    logits = outputs[i]
                    targets = features[:, :, i]
                    loss_i = F.cross_entropy(
                        logits.reshape(-1, vocab_sizes[i]),
                        targets.reshape(-1),
                        reduction='mean'
                    )
                    recon_loss += loss_i
                # Compute KL divergence loss (averaged over the batch).
                kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                loss = recon_loss + kl_beta * kl_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", kl_loss=f"{kl_loss.item():.4f}")
                pbar.update(1)
        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch}/{num_epochs}: Average Loss = {avg_loss:.4f}")
        
        # Perform validation (e.g., on CSV data)
        precision, recall, f1 = validate_csv(model, params, device, vocab_sizes)
        logging.info(f"Epoch {epoch}: Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'params': params,
            }
            filename = os.path.join(params['output_dir'], checkpoint_path)
            save_checkpoint(checkpoint_state, filename)
    return model, best_f1

###################################
# 5. DATASET CLASSES
###################################

class ParquetSequenceDataset(IterableDataset):
    def __init__(self,
                 parquet_path: str,
                 feature_columns: list,
                 seq_len: int,
                 stride: int = None,
                 shuffle_files: bool = True,
                 split: str = 'train',
                 validation_ratio: float = 0.2,
                 seed: int = 42):
        super().__init__()
        self.parquet_path = Path(parquet_path)
        self.feature_columns = feature_columns
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.shuffle_files = shuffle_files
        self.split = split
        self.validation_ratio = validation_ratio
        self.seed = seed
        try:
            parquet_file = pd.read_parquet(self.parquet_path, columns=['file_id'])
            file_info = parquet_file.groupby('file_id').size().reset_index(name='counts')
            total_lines = file_info['counts'].sum()
            val_target = total_lines * self.validation_ratio
            rng = np.random.RandomState(self.seed)
            files_with_sizes = list(zip(file_info['file_id'], file_info['counts']))
            rng.shuffle(files_with_sizes)
            val_file_ids = []
            train_file_ids = []
            cumsum = 0
            for f_id, size in files_with_sizes:
                if cumsum < val_target:
                    val_file_ids.append(f_id)
                    cumsum += size
                else:
                    train_file_ids.append(f_id)
            self.file_ids = train_file_ids if self.split == 'train' else val_file_ids
            self.num_files = len(self.file_ids)
            logging.info(f"Using {self.num_files} files for the {split} split.")
            file_counts_map = dict(zip(file_info['file_id'], file_info['counts']))
            total_sequences = 0
            for fid in self.file_ids:
                n_lines = file_counts_map.get(fid, 0)
                possible_sequences = (n_lines - self.seq_len) // self.stride
                if possible_sequences > 0:
                    total_sequences += possible_sequences
            self._length = max(total_sequences, 0)
        except Exception as e:
            logging.error(f"Error reading Parquet file {self.parquet_path}: {e}")
            self.file_ids = []
            self.num_files = 0
            self._length = 0

    def _get_sequences_from_chunk(self, chunk: pd.DataFrame):
        try:
            features = torch.tensor(chunk[self.feature_columns].values, dtype=torch.long)
            timestamps = chunk['timestamp_str'].tolist()
            n_sequences = (len(features) - self.seq_len) // self.stride
            sequences = []
            sequence_timestamps = []
            for i in range(n_sequences):
                start_idx = i * self.stride
                end_idx = start_idx + self.seq_len
                if end_idx > len(features):
                    break
                sequences.append(features[start_idx:end_idx])
                sequence_timestamps.append(timestamps[start_idx:end_idx])
            if sequences:
                return torch.stack(sequences), sequence_timestamps
            return None, None
        except Exception as e:
            logging.error(f"Error extracting sequences from chunk: {e}")
            return None, None

    def _process_file_id(self, file_id: int):
        try:
            df = pd.read_parquet(self.parquet_path, filters=[('file_id', '=', file_id)])
            if df.isna().any().any():
                raise ValueError("Found NaNs!")
            return self._get_sequences_from_chunk(df)
        except Exception as e:
            logging.error(f"Error processing file_id {file_id}: {e}")
            return None, None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        file_ids = self.file_ids.copy()
        if self.shuffle_files:
            np.random.shuffle(file_ids)
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            file_ids = np.array_split(file_ids, num_workers)[worker_id]
        for file_id in file_ids:
            sequences, timestamps = self._process_file_id(file_id)
            if sequences is not None and timestamps is not None:
                for seq, ts in zip(sequences, timestamps):
                    yield {'features': seq, 'timestamps': ts}

    def __len__(self):
        return self._length

class CSVValidationDataset(Dataset):
    def __init__(self, csv_path, seq_len=32, stride=16):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.stride = stride
        required_columns = ["timestamp_str", "SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"]
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' is missing from {csv_path}!")
        self.features = self.df[["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"]].values.astype(np.int64)
        self.timestamps = self.df["timestamp_str"].tolist()
        self.num_sequences = max(0, (len(self.features) - self.seq_len) // self.stride + 1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_len
        if end_idx > len(self.features):
            end_idx = len(self.features)
            start_idx = end_idx - self.seq_len
        feature_seq = torch.tensor(self.features[start_idx:end_idx], dtype=torch.long)
        timestamp_seq = self.timestamps[start_idx:end_idx]
        return {'features': feature_seq, 'timestamps': timestamp_seq}

def custom_collate_fn(batch):
    try:
        features = torch.stack([item['features'] for item in batch])
        timestamps = [item['timestamps'] for item in batch]
        return {'features': features, 'timestamps': timestamps}
    except Exception as e:
        logging.error(f"Error in collate_fn: {e}")
        return {}

###################################
# 6. MAIN TRAINING PIPELINE
###################################

def main_training_pipeline(params, model, vocab_sizes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create the base dataset.
    train_dataset = ParquetSequenceDataset(
        parquet_path=params['parquet_path'],
        feature_columns=params['feature_columns'],
        seq_len=params['seq_len'],
        stride=params['stride'],
        split='train',
        validation_ratio=params['validation_ratio'],
        seed=params['seed']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model has {total_params / 1e3:.2f}K parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 1e-3))
    model, best_f1 = train_anomaly_detector(model, train_loader, optimizer, device, params, vocab_sizes)
    logging.info(f"Training complete. Best validation F1: {best_f1:.4f}")
    return model, params

###################################
# 7. MAIN
###################################

if __name__ == '__main__':
    # Hyperparameters and settings.
    params = {
        'parquet_path': 'unscaled_pdsch.parquet',
        'ground_truth_csv_path': 'test_gt.csv',
        'validation_csv_path': 'test.csv',
        'output_dir': './output',
        'feature_columns': ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"],
        'seq_len': 200,
        'stride': 100,
        'validation_ratio': 0.95,
        'seed': 42,
        'batch_size': 16,
        'num_epochs': 1,
        'lr': 5e-4,
        'hidden_dim': 128,
        'percentile': 97,
        'num_layers': 1,
        'dropout': 0.0,
        'vocab_sizes': [1024, 31, 16, 33, 2, 9, 2],
        'embedding_dims': [8, 4, 4, 4, 2, 4, 2],
        'pca_n_components': 3,
        'kl_beta': 0.1  # Weight for the KL divergence term.
    }

    start_logging(params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VariationalCategoricalAutoencoder(
        vocab_sizes=params['vocab_sizes'],
        embedding_dims=params['embedding_dims'],
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    )
    model.to(device)
    model, params = main_training_pipeline(params, model, params['vocab_sizes'])
