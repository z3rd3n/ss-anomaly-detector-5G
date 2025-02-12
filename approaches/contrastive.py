import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
import random
from tqdm import tqdm
from datetime import datetime

###############################
# 1. LOGGING SETUP
###############################
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

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_embeddings(embeddings, labels, epoch, output_dir):
    """
    embeddings: NumPy array of shape [N, D]
    labels: Optional, a list/array of labels or flags for coloring
    """
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    plt.title(f"Embedding Space at Epoch {epoch}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    # Save the plot instead of displaying it
    plot_path = os.path.join(output_dir, f"embeddings_epoch_{epoch}.png")
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"PCA plot for epoch {epoch} saved to {plot_path}")


###############################
# 2. DATA AUGMENTATION FUNCTIONS
###############################
# The feature ranges (the same order as in your original [SFN, Slot, HARQ, MCS, CRC, ReTx, NDI])
FEATURE_RANGES = {
    0: (0, 1023),  # SFN
    1: (0, 30),    # Slot
    2: (0, 15),    # HARQ
    3: (0, 32),    # MCS
    4: (0, 1),     # CRC (binary)
    5: (0, 8),     # ReTx
    6: (0, 1)      # NDI (binary)
}
MAX_RETX = 4  # For example

def augment_positive(sequence: torch.Tensor) -> torch.Tensor:
    seq = sequence.clone()
    # Reduce perturbation probability
    for idx in [3, 5]:  # MCS and ReTx
        if random.random() < 0.3:  # Lower probability
            offset = random.choice([-1, 1])
            seq[:, idx] = torch.clamp(
                seq[:, idx] + offset, 
                min=FEATURE_RANGES[idx][0], 
                max=FEATURE_RANGES[idx][1]
            )
    return seq

def rule_based_flags(sequence: torch.Tensor) -> torch.Tensor:
    """
    Check each timestep in the sequence (assumed shape [seq_len, 7]) against rule-based conditions.
    Returns a boolean tensor of shape [seq_len] where True indicates an anomalous line.
    Rules (per timestep):
      - "unnecessary_retx": if ReTx > 0 and the previous CRC in the same HARQ equals 1.
      - "missing_retx": if ReTx == 0, previous CRC == 0 and current NDI equals previous NDI.
      - "new_data_no_retx": if previous CRC == 0 and current NDI is not equal to previous NDI.
      - "max_retx_achieved": if ReTx >= MAX_RETX.
    """
    seq = sequence.cpu().numpy()  # shape (seq_len, 7)
    seq_len = seq.shape[0]
    flags = np.zeros(seq_len, dtype=bool)
    for t in range(seq_len):
        current_harq = seq[t, 2]
        # Look for the most recent timestep with the same HARQ
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

def inject_negative(sequence: torch.Tensor) -> torch.Tensor:
    """
    Create a negative (anomalous) example from the sequence with improved injection strategy.
    Now ensures at least one rule-based anomaly is always present.
    """
    seq = sequence.clone()
    flags = rule_based_flags(seq)  # Boolean tensor of shape [seq_len]
    
    # If no rule-based anomalies present, explicitly create them
    if not flags.sum() > 0:
        # Select random positions for injection (between 1-3 positions)
        num_injections = random.randint(1, seq.shape[0]//4)
        positions = random.sample(range(seq.shape[0]), num_injections)
        
        for pos in positions:
            # Get HARQ context
            current_harq = seq[pos, 2]
            prev_idx = None
            for i in range(pos - 1, -1, -1):
                if seq[i, 2] == current_harq:
                    prev_idx = i
                    break
                    
            if prev_idx is not None:
                # Choose one of several rule-based anomaly types
                anomaly_type = random.choice([
                    'unnecessary_retx',
                    'missing_retx',
                    'new_data_no_retx',
                    'max_retx'
                ])
                
                if anomaly_type == 'unnecessary_retx':
                    # Create unnecessary retransmission when previous CRC was good
                    seq[prev_idx, 4] = 1  # Set previous CRC to success
                    seq[pos, 5] = 1 # Retransmit
                    
                elif anomaly_type == 'missing_retx':
                    # Missing retransmission when needed
                    seq[prev_idx, 4] = 0  # Set previous CRC to failure
                    seq[pos, 5] = 0  # No retransmission
                    seq[pos, 6] = seq[prev_idx, 6]  # Same NDI
                    
                elif anomaly_type == 'new_data_no_retx':
                    # New data when should be retransmitting
                    seq[prev_idx, 4] = 0  # Set previous CRC to failure
                    seq[pos, 6] = 1 - seq[prev_idx, 6]  # Toggle NDI
                    
                elif anomaly_type == 'max_retx':
                    # Exceed max retransmissions
                    seq[pos, 5] = MAX_RETX + random.randint(0, 3)
    
    # Additional perturbations on existing anomalies
    flags = rule_based_flags(seq)  # Recompute flags after injections
    for t in range(seq.shape[0]):
        if flags[t]:
            # Add extra perturbation on MCS for anomalous timesteps
            mcs_change = random.randint(2, 8) * random.choice([-1, 1])
            new_mcs = seq[t, 3].item() + mcs_change
            seq[t, 3] = torch.clamp(torch.tensor(new_mcs), FEATURE_RANGES[3][0], FEATURE_RANGES[3][1])
            # Add extra perturbation to SFN, Slot, HARQ
            for idx in [0, 1]:
                offset = random.choice([-1, 1])
                seq[t, idx] = torch.clamp(
                    seq[t, idx] + offset, 
                    min=FEATURE_RANGES[idx][0], 
                    max=FEATURE_RANGES[idx][1])
    return seq

###############################
# 3. THE MODEL: CNN FEATURE EXTRACTOR WITH TEMPORAL (CAUSAL) TRANSFORMER
###############################
class CNNFeatureExtractor(nn.Module):
    """
    Enhanced CNN Feature Extractor that learns feature relationships automatically.
    Now includes proper type handling for input tensors.
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Initial feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Multi-scale temporal convolutions
        self.temporal_convs = nn.ModuleList([
            # Short-range patterns
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim // 4, kernel_size=1),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.GELU()
            ),
            # Local patterns
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim // 4, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.GELU()
            ),
            # Medium-range patterns
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim // 4, kernel_size=5, padding=4, dilation=2),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.GELU()
            ),
            # Long-range patterns with dilation
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim // 4, kernel_size=7, padding=6, dilation=2),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.GELU()
            )
        ])
        
        # Self-attention for learning feature relationships
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Final processing
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        x: [B, seq_len, 7]
        Returns: [B, seq_len, hidden_dim]
        """
        # Convert input to float and normalize
        x = x.float()  # Convert from Long to Float
        
        B, seq_len, _ = x.size()
        
        # Initial projection
        x = self.input_proj(x)  # [B, seq_len, hidden_dim]
        identity = x
        
        # Multi-scale temporal convolutions
        x_conv = x.transpose(1, 2)  # [B, hidden_dim, seq_len]
        conv_outputs = []
        for conv in self.temporal_convs:
            conv_outputs.append(conv(x_conv))
        
        # Concatenate all conv outputs
        x_multi = torch.cat(conv_outputs, dim=1)  # [B, hidden_dim, seq_len]
        x_multi = x_multi.transpose(1, 2)  # [B, seq_len, hidden_dim]
        
        # Self-attention for feature relationship learning
        x_att, _ = self.feature_attention(x, x, x)
        
        # Combine convolution and attention features
        combined = torch.cat([x_multi, x_att], dim=-1)
        output = self.output_proj(combined)
        
        # Skip connection and normalization
        output = self.layer_norm(output + identity)
        output = self.dropout(output)
        
        return F.normalize(output, p=2, dim=-1)


###############################
# 4. CONTRASTIVE TRAINING SETUP (DATASET WRAPPER)
###############################
class ContrastiveTrainingWrapper(IterableDataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __iter__(self):
        for sample in self.base_dataset:
            anchor = sample['features']
            positive = augment_positive(anchor)
            negative = inject_negative(anchor)
            yield {
                'anchor': anchor,
                'positive': positive,
                'negative': negative,
                'timestamps': sample['timestamps']
            }
    def __len__(self):
        return len(self.base_dataset)

def custom_triplet_collate_fn(batch):
    try:
        anchors = torch.stack([item['anchor'] for item in batch])
        positives = torch.stack([item['positive'] for item in batch])
        negatives = torch.stack([item['negative'] for item in batch])
        timestamps = [item['timestamps'] for item in batch]
        return {
            'anchor': anchors,
            'positive': positives,
            'negative': negatives,
            'timestamps': timestamps,
        }
    except Exception as e:
        logging.error(f"Error in triplet collate_fn: {e}")
        return {}

###############################
# 5. TRAINING AND VALIDATION FUNCTIONS
###############################
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

def compute_normality_centroid_timestep(model, loader, device):
    """
    Compute the centroid (mean) of embeddings from training data (per timestep) and re-normalize it.
    """
    model.eval()
    total_sum = None
    total_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing centroid", unit="batch"):
            feats = batch['features'].to(device)
            emb_seq = model(feats)  # [B, seq_len, hidden_dim]
            batch_sum = emb_seq.sum(dim=(0, 1))
            count = emb_seq.shape[0] * emb_seq.shape[1]
            total_sum = batch_sum if total_sum is None else total_sum + batch_sum
            total_count += count
    centroid = total_sum / total_count
    centroid = centroid / centroid.norm(p=2)
    return centroid

def validate_csv(model, params, device):
    logging.info("Starting CSV validation procedure...")
    val_dataset = CSVValidationDataset(
        csv_path=params['validation_csv_path'],
        seq_len=params['seq_len'],
        stride=params['stride']
    )
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=custom_collate_fn)
    anomalies_df, _ = produce_anomalies(model, val_loader, params, device)
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

def produce_anomalies(model, val_loader, params, device, threshold=None, checkpoint_path='best_model.pt'):
    """
    Produce anomaly records from the validation loader. For each timestep,
    compute an anomaly score that is a combination of:
      - The Euclidean distance from a normality centroid.
      - An extra term based on the jump (difference) between consecutive embeddings.
    Also, the output CSV now includes the original feature values.
    """
    model.to(device)
    model.eval()
    anomalies = []
    all_scores = []
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Producing anomalies", unit="batch"):
            feats = batch['features'].to(device)           # [B, seq_len, 7]
            emb_seq = model(feats)          # [B, seq_len, hidden_dim]
            B, T, D = emb_seq.size()
            all_embeddings.append(emb_seq.cpu().numpy())
            all_labels.extend([0] * B * T)  # Assuming 0 for normal, modify as needed
            
            # Compute distance from the centroid per timestep.
            centroid = params.get('centroid')
            if centroid is None:
                raise ValueError("Centroid not computed. Please compute it from training data.")
            # Expand centroid to [B, T]
            dists = torch.cdist(emb_seq, centroid.to(device).unsqueeze(0), p=2).squeeze(-1)  # [B, T]
            
            # Final anomaly score: combine the two terms.
            scores = dists  # [B, T]
            all_scores.append(scores.cpu().numpy())
            
            batch_features = batch['features']  # original features, shape: [B, T, 7]
            batch_timestamps = batch['timestamps']  # list of lists
            scores_np = scores.cpu().numpy()
            for b in range(B):
                for t in range(T):
                    score = scores_np[b, t]
                    timestamp = batch_timestamps[b][t]
                    features_list = batch_features[b][t].tolist()
                    anomalies.append({
                        'timestamp': timestamp,
                        'anomaly_score': score,
                        'features': features_list
                    })
    # Flatten scores over the whole dataset to compute threshold if not provided.
    all_scores_flat = np.concatenate(all_scores).flatten()
    if threshold is None:
        threshold = np.percentile(all_scores_flat, 95)
    anomalies = [rec for rec in anomalies if rec['anomaly_score'] > threshold]
    anomalies_df = pd.DataFrame(anomalies)
    anomalies_df = anomalies_df.sort_values(by='anomaly_score', ascending=False)
    anomalies_csv_path = os.path.join(params['output_dir'], 'anomalies.csv')
    anomalies_df.to_csv(anomalies_csv_path, index=False)
    logging.info(f"Anomalies saved to {anomalies_csv_path}")
    
    # Plot embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    plot_embeddings(all_embeddings, all_labels, epoch=0, output_dir=params['output_dir'])
    
    return anomalies_df, threshold

def train_contrastive(model, train_loader, train_plain_loader, optimizer, device, params, checkpoint_path='best_model.pt'):
    model.train()
    margin = params.get('margin', 2.0)
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    best_f1 = 0.0
    num_epochs = params.get('num_epochs', 10)
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs}", unit="batch") as pbar:
            for i, batch in enumerate(train_loader):
                anchor = batch['anchor'].to(device)      # [B, seq_len, 7]
                positive = batch['positive'].to(device)
                negative = batch['negative'].to(device)
                optimizer.zero_grad()
                # Get per-timestep representations
                emb_anchor = model(anchor)   # [B, seq_len, hidden_dim]
                emb_positive = model(positive)
                emb_negative = model(negative)
                B, T, D = emb_anchor.size()
                emb_anchor = emb_anchor.reshape(B * T, D)
                emb_positive = emb_positive.reshape(B * T, D)
                emb_negative = emb_negative.reshape(B * T, D)
                loss = triplet_loss_fn(emb_anchor, emb_positive, emb_negative)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)
        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch}/{num_epochs}: Average Loss = {avg_loss:.4f}")
        # Compute the centroid on plain training data
        centroid = compute_normality_centroid_timestep(model, train_plain_loader, device)
        params['centroid'] = centroid
        precision, recall, f1 = validate_csv(model, params, device)
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

###############################
# 6. ORIGINAL DATASET CLASSES (SIMILAR TO YOUR ORIGINAL CODE)
###############################
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

###############################
# 7. MAIN TRAINING PIPELINE
###############################
def main_training_pipeline(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    # Create the base dataset from the parquet file.
    train_base_dataset = ParquetSequenceDataset(
        parquet_path=params['parquet_path'],
        feature_columns=params['feature_columns'],
        seq_len=params['seq_len'],
        stride=params['stride'],
        split='train',
        validation_ratio=params['validation_ratio'],
        seed=params['seed']
    )
    train_plain_loader = DataLoader(
        train_base_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        collate_fn=custom_collate_fn,
    )
    params['train_plain_loader'] = train_plain_loader
    # Wrap the base dataset with the contrastive training wrapper.
    contrastive_dataset = ContrastiveTrainingWrapper(train_base_dataset)
    train_loader = DataLoader(
        contrastive_dataset,
        batch_size=params['batch_size'],
        collate_fn=custom_triplet_collate_fn,
    )
    # Create the model with CNN+Transformer; hyperparameters can be tuned.
    model = CNNFeatureExtractor(
        hidden_dim=params.get('hidden_dim', 128)
    )
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model has {total_params / 1e3:.2f}K parameters.")
    optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 1e-3))
    model, best_f1 = train_contrastive(model, train_loader, train_plain_loader, optimizer, device, params)
    logging.info(f"Training complete. Best validation F1: {best_f1:.4f}")
    return model, params

###############################
# 8. PARAMETERS AND RUNNING THE PIPELINE
###############################
if __name__ == '__main__':
    start_logging()
    params = {
        # Data paths (update these as needed)
        'parquet_path': 'unscaled_pdsch.parquet',
        'ground_truth_csv_path': 'test_gt.csv',
        'validation_csv_path': 'test.csv',
        'output_dir': './output',
        # Feature and sequence parameters
        'feature_columns': ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"],
        'seq_len': 30,
        'stride': 30,
        'validation_ratio': 0.0,
        'seed': 42,
        # Training parameters
        'batch_size': 64,
        'num_epochs': 1,
        'lr': 1e-4,
        'margin': 1.5,  # Adjust as needed.
        'hidden_dim': 64,
    }
    model, params = main_training_pipeline(params)
