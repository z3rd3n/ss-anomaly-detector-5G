import os
import math
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, IterableDataset

import optuna  # Make sure to install optuna: pip install optuna

# -------------------------- Logging --------------------------
def start_logging(params=None):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = params['output_dir'] if params is not None else 'output'
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

    if params is not None:
        logging.info("Hyperparameters and settings:")
        for key, value in params.items():
            logging.info(f"{key}: {value}")

# -------------------------- Positional Encoding --------------------------
class PositionalEncoding(nn.Module):
    """
    Standard positional encoding module.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape (seq_len, batch_size, d_model) or (batch, seq_len, d_model)
        """
        # We assume batch_first is used for our transformer.
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

# -------------------------- Transformer Autoencoder --------------------------
class TransformerAnomalyDetector(nn.Module):
    """
    Transformer autoencoder for anomaly detection.
    """
    def __init__(self, d_model=128, d_feature=16, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        # Embedding layers for each feature (cardinalities as given)
        self.embed_sfn   = nn.Embedding(1024, d_feature)  # SFN: 0-1023
        self.embed_slot  = nn.Embedding(31,   d_feature)  # Slot: 0-30
        self.embed_harq  = nn.Embedding(16,   d_feature)  # HARQ: 0-15
        self.embed_mcs   = nn.Embedding(33,   d_feature)  # MCS: 0-32
        self.embed_crc   = nn.Embedding(2,    d_feature)  # CRC: 0-1
        self.embed_retx  = nn.Embedding(9,    d_feature)  # ReTx: 0-8
        self.embed_ndi   = nn.Embedding(2,    d_feature)  # NDI: 0-1

        self.num_features = 7
        # Concatenate embeddings (d_feature * 7) and project to d_model
        self.input_proj = nn.Linear(d_feature * self.num_features, d_model)
        
        # Positional encoding (for the temporal axis)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer autoencoder (encoder + decoder)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output heads for each feature: project from d_model to vocab size for that feature.
        self.head_sfn  = nn.Linear(d_model, 1024)
        self.head_slot = nn.Linear(d_model, 31)
        self.head_harq = nn.Linear(d_model, 16)
        self.head_mcs  = nn.Linear(d_model, 33)
        self.head_crc  = nn.Linear(d_model, 2)
        self.head_retx = nn.Linear(d_model, 9)
        self.head_ndi  = nn.Linear(d_model, 2)
        
        # A learnable cluster center (to “pull” normal sequences together)
        self.cluster_center = nn.Parameter(torch.zeros(d_model))

    def forward(self, src):
        """
        Args:
            src: LongTensor of shape (batch, seq_len, num_features)
        Returns:
            A dict with predictions (logits per feature), the decoder output (latent representations)
            and an average latent vector per sequence.
        """
        batch, seq_len, num_features = src.size()
        sfn   = src[:, :, 0]
        slot  = src[:, :, 1]
        harq  = src[:, :, 2]
        mcs   = src[:, :, 3]
        crc   = src[:, :, 4]
        retx  = src[:, :, 5]
        ndi   = src[:, :, 6]
        
        emb_sfn   = self.embed_sfn(sfn)
        emb_slot  = self.embed_slot(slot)
        emb_harq  = self.embed_harq(harq)
        emb_mcs   = self.embed_mcs(mcs)
        emb_crc   = self.embed_crc(crc)
        emb_retx  = self.embed_retx(retx)
        emb_ndi   = self.embed_ndi(ndi)
        
        concat_emb = torch.cat([
            emb_sfn, emb_slot, emb_harq, emb_mcs, emb_crc, emb_retx, emb_ndi
        ], dim=-1)
        
        src_emb = self.input_proj(concat_emb)  # shape: (batch, seq_len, d_model)
        src_emb = self.pos_encoder(src_emb)
        tgt = src_emb
        decoded = self.transformer(src_emb, tgt)  # (batch, seq_len, d_model)
        
        pred_sfn   = self.head_sfn(decoded)
        pred_slot  = self.head_slot(decoded)
        pred_harq  = self.head_harq(decoded)
        pred_mcs   = self.head_mcs(decoded)
        pred_crc   = self.head_crc(decoded)
        pred_retx  = self.head_retx(decoded)
        pred_ndi   = self.head_ndi(decoded)
        
        latent_avg = decoded.mean(dim=1)
        
        return {
            "predictions": {
                "SFN": pred_sfn,
                "Slot": pred_slot,
                "HARQ": pred_harq,
                "MCS": pred_mcs,
                "CRC": pred_crc,
                "ReTx": pred_retx,
                "NDI": pred_ndi,
            },
            "reconstructed_sequence": decoded,
            "latent_avg": latent_avg
        }

# -------------------------- Loss Function with Adaptive Weighting --------------------------
def compute_loss(model_output, target, model, lambda_cluster):
    """
    Compute the total loss as the sum of reconstruction loss plus an adaptive weighted cluster loss.
    """
    batch, seq_len, _ = target.size()
    device = target.device

    rec_error_tokens = torch.zeros(batch, seq_len, device=device)
    
    sfn_loss_tokens = F.cross_entropy(
        model_output["predictions"]["SFN"].view(-1, 1024),
        target[:, :, 0].view(-1),
        reduction='none'
    ).view(batch, seq_len)
    rec_error_tokens += sfn_loss_tokens
    
    slot_loss_tokens = F.cross_entropy(
        model_output["predictions"]["Slot"].view(-1, 31),
        target[:, :, 1].view(-1),
        reduction='none'
    ).view(batch, seq_len)
    rec_error_tokens += slot_loss_tokens
    
    harq_loss_tokens = F.cross_entropy(
        model_output["predictions"]["HARQ"].view(-1, 16),
        target[:, :, 2].view(-1),
        reduction='none'
    ).view(batch, seq_len)
    rec_error_tokens += harq_loss_tokens
    
    mcs_loss_tokens = F.cross_entropy(
        model_output["predictions"]["MCS"].view(-1, 33),
        target[:, :, 3].view(-1),
        reduction='none'
    ).view(batch, seq_len)
    rec_error_tokens += mcs_loss_tokens
    
    crc_loss_tokens = F.cross_entropy(
        model_output["predictions"]["CRC"].view(-1, 2),
        target[:, :, 4].view(-1),
        reduction='none'
    ).view(batch, seq_len)
    rec_error_tokens += crc_loss_tokens
    
    retx_loss_tokens = F.cross_entropy(
        model_output["predictions"]["ReTx"].view(-1, 9),
        target[:, :, 5].view(-1),
        reduction='none'
    ).view(batch, seq_len)
    rec_error_tokens += retx_loss_tokens
    
    ndi_loss_tokens = F.cross_entropy(
        model_output["predictions"]["NDI"].view(-1, 2),
        target[:, :, 6].view(-1),
        reduction='none'
    ).view(batch, seq_len)
    rec_error_tokens += ndi_loss_tokens
    
    recon_loss = rec_error_tokens.mean()
    
    weight = 1.0 - torch.sigmoid(rec_error_tokens)
    
    latent_seq = model_output["reconstructed_sequence"]  # (B, T, d_model)
    cluster_center = model.cluster_center.expand_as(latent_seq)
    
    cluster_loss = (weight.unsqueeze(-1) * (latent_seq - cluster_center).pow(2)).mean()
    
    total_loss = recon_loss + lambda_cluster * cluster_loss
    return total_loss, recon_loss, cluster_loss

# -------------------------- Checkpoint Save/Load Functions --------------------------
def save_checkpoint(model, optimizer, epoch, params, filename="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "params": params,
    }
    torch.save(checkpoint, os.path.join(params['output_dir'], filename))
    logging.info(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename="checkpoint.pth", device="cpu"):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    params = checkpoint["params"]
    logging.info(f"Checkpoint loaded from {filename}, starting at epoch {epoch}")
    return epoch, params

# -------------------------- Adaptive Threshold Update Function --------------------------
def update_threshold(model, dataloader, params, device, k=2.0):
    """
    Update the anomaly threshold adaptively using a streaming algorithm.
    """
    model.eval()
    n = 0
    mean = 0.0
    M2 = 0.0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Updating threshold", leave=False)):
            if i >= 1000:
                break
            features = batch['features'].to(device)
            batch_size, seq_len, _ = features.shape
            output = model(features)
            latent_seq = output["reconstructed_sequence"]
            cluster_center = model.cluster_center.to(device)
            cluster_distance = torch.norm(latent_seq - cluster_center, dim=-1)
            
            rec_error = torch.zeros(batch_size, seq_len, device=device)
            sfn_loss = F.cross_entropy(output["predictions"]["SFN"].view(-1, 1024),
                                       features[:, :, 0].view(-1),
                                       reduction='none').view(batch_size, seq_len)
            rec_error += sfn_loss
            slot_loss = F.cross_entropy(output["predictions"]["Slot"].view(-1, 31),
                                        features[:, :, 1].view(-1),
                                        reduction='none').view(batch_size, seq_len)
            rec_error += slot_loss
            harq_loss = F.cross_entropy(output["predictions"]["HARQ"].view(-1, 16),
                                        features[:, :, 2].view(-1),
                                        reduction='none').view(batch_size, seq_len)
            rec_error += harq_loss
            mcs_loss = F.cross_entropy(output["predictions"]["MCS"].view(-1, 33),
                                       features[:, :, 3].view(-1),
                                       reduction='none').view(batch_size, seq_len)
            rec_error += mcs_loss
            crc_loss = F.cross_entropy(output["predictions"]["CRC"].view(-1, 2),
                                       features[:, :, 4].view(-1),
                                       reduction='none').view(batch_size, seq_len)
            rec_error += crc_loss
            retx_loss = F.cross_entropy(output["predictions"]["ReTx"].view(-1, 9),
                                        features[:, :, 5].view(-1),
                                        reduction='none').view(batch_size, seq_len)
            rec_error += retx_loss
            ndi_loss = F.cross_entropy(output["predictions"]["NDI"].view(-1, 2),
                                       features[:, :, 6].view(-1),
                                       reduction='none').view(batch_size, seq_len)
            rec_error += ndi_loss
            
            batch_anomaly_score = params['alpha'] * rec_error + params['beta'] * cluster_distance  # (B, T)
            scores = batch_anomaly_score.view(-1)
            for score in scores:
                x = score.item()
                n += 1
                delta = x - mean
                mean += delta / n
                delta2 = x - mean
                M2 += delta * delta2
    if n > 1:
        std = (M2 / (n - 1)) ** 0.5
    else:
        std = 0.0
    new_threshold = mean + k * std
    logging.info(f"Updated threshold: {new_threshold:.4f} (mean: {mean:.4f}, std: {std:.4f})")
    return new_threshold

# -------------------------- Inference / Anomaly Production --------------------------
def produce_anomalies(model, dataloader, params, device):
    """
    For each sequence in the dataloader, compute an anomaly score per time step.
    """
    model.eval()
    anomalies = []
    all_scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Producing anomalies", leave=False):
            features = batch['features'].to(device)  # (B, T, num_features)
            timestamps = batch['timestamps']          # list of lists
            output = model(features)
            batch_size, seq_len, _ = features.shape
            
            latent_seq = output["reconstructed_sequence"]  # (B, T, d_model)
            cluster_center = model.cluster_center.to(device)
            cluster_distance = torch.norm(latent_seq - cluster_center, dim=-1)
            
            rec_error = torch.zeros(batch_size, seq_len, device=device)
            sfn_loss = F.cross_entropy(output["predictions"]["SFN"].view(-1, 1024),
                                       features[:, :, 0].view(-1),
                                       reduction='none').view(batch_size, seq_len)
            rec_error += sfn_loss
            slot_loss = F.cross_entropy(output["predictions"]["Slot"].view(-1, 31),
                                        features[:, :, 1].view(-1),
                                        reduction='none').view(batch_size, seq_len)
            rec_error += slot_loss
            harq_loss = F.cross_entropy(output["predictions"]["HARQ"].view(-1, 16),
                                        features[:, :, 2].view(-1),
                                        reduction='none').view(batch_size, seq_len)
            rec_error += harq_loss
            mcs_loss = F.cross_entropy(output["predictions"]["MCS"].view(-1, 33),
                                       features[:, :, 3].view(-1),
                                       reduction='none').view(batch_size, seq_len)
            rec_error += mcs_loss
            crc_loss = F.cross_entropy(output["predictions"]["CRC"].view(-1, 2),
                                       features[:, :, 4].view(-1),
                                       reduction='none').view(batch_size, seq_len)
            rec_error += crc_loss
            retx_loss = F.cross_entropy(output["predictions"]["ReTx"].view(-1, 9),
                                        features[:, :, 5].view(-1),
                                        reduction='none').view(batch_size, seq_len)
            rec_error += retx_loss
            ndi_loss = F.cross_entropy(output["predictions"]["NDI"].view(-1, 2),
                                       features[:, :, 6].view(-1),
                                       reduction='none').view(batch_size, seq_len)
            rec_error += ndi_loss
            
            anomaly_score = params['alpha'] * rec_error + params['beta'] * cluster_distance
            
            pred_sfn   = torch.argmax(output["predictions"]["SFN"], dim=-1)
            pred_slot  = torch.argmax(output["predictions"]["Slot"], dim=-1)
            pred_harq  = torch.argmax(output["predictions"]["HARQ"], dim=-1)
            pred_mcs   = torch.argmax(output["predictions"]["MCS"], dim=-1)
            pred_crc   = torch.argmax(output["predictions"]["CRC"], dim=-1)
            pred_retx  = torch.argmax(output["predictions"]["ReTx"], dim=-1)
            pred_ndi   = torch.argmax(output["predictions"]["NDI"], dim=-1)
            
            for i in range(batch_size):
                ts_list = timestamps[i]
                for j in range(seq_len):
                    score = anomaly_score[i, j].item()
                    if score > params['threshold']:
                        ts = ts_list[j] if j < len(ts_list) else "N/A"
                        anomaly_entry = {
                            "timestamp": ts,
                            "anomaly_score": score,
                            "reconstruction_error": rec_error[i, j].item(),
                            "cluster_distance": cluster_distance[i, j].item(),
                            "threshold": params['threshold'],
                            "SFN_pred": pred_sfn[i, j].item(),
                            "SFN_orig": features[i, j, 0].item(),
                            "Slot_pred": pred_slot[i, j].item(),
                            "Slot_orig": features[i, j, 1].item(),
                            "HARQ_pred": pred_harq[i, j].item(),
                            "HARQ_orig": features[i, j, 2].item(),
                            "MCS_pred": pred_mcs[i, j].item(),
                            "MCS_orig": features[i, j, 3].item(),
                            "CRC_pred": pred_crc[i, j].item(),
                            "CRC_orig": features[i, j, 4].item(),
                            "ReTx_pred": pred_retx[i, j].item(),
                            "ReTx_orig": features[i, j, 5].item(),
                            "NDI_pred": pred_ndi[i, j].item(),
                            "NDI_orig": features[i, j, 6].item(),
                        }
                        anomalies.append(anomaly_entry)
            all_scores.append(anomaly_score.cpu().numpy())
            
    anomalies_df = pd.DataFrame(anomalies)
    if anomalies_df.empty:
        logging.warning("No anomalies detected during inference!")
    else:
        logging.info(f"Detected {len(anomalies_df)} anomalous rows.")
    return anomalies_df, all_scores

# -------------------------- CSVValidationDataset --------------------------
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
        
        return {
            'features': feature_seq,
            'timestamps': timestamp_seq
        }

def validate_csv(model, params, device):
    """
    Runs anomaly detection on the validation CSV.
    """
    logging.info("Starting CSV validation procedure...")
    val_dataset = CSVValidationDataset(
        params['validation_csv_path'], 
        seq_len=params['seq_len'], 
        stride=params['stride']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=params['batch_size'], 
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
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
    
    logging.info(
        f"[RESULT] CSV Validation - Precision: {precision*100:.2f}%, "
        f"Recall: {recall*100:.2f}%, F1-Score: {f1_score*100:.2f}% "
        f"({num_accurate} correct out of {total_anoms} anomalies, {total_gt} ground truth)."
    )
    return precision, recall, f1_score

# -------------------------- (Optional) ParquetSequenceDataset & collate_fn --------------------------
class ParquetSequenceDataset(IterableDataset):
    def __init__(
        self,
        parquet_path: str,
        feature_columns: list,
        seq_len: int,
        stride: int,
        shuffle_files: bool = True,
        split: str = 'train',
        validation_ratio: float = 0.2,
        seed: int = 42
    ):
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
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(self.parquet_path)
            file_info = parquet_file.read(columns=['file_id']).to_pandas()
            file_counts_df = file_info.groupby('file_id').size().reset_index(name='counts')
            total_lines = file_counts_df['counts'].sum()
            val_target = total_lines * self.validation_ratio

            rng = np.random.RandomState(self.seed)
            files_with_sizes = list(zip(file_counts_df['file_id'], file_counts_df['counts']))
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
            logging.info(
                f"Using {self.num_files} files for the {split} split (split by line counts)."
            )

            file_counts_map = dict(zip(file_counts_df['file_id'], file_counts_df['counts']))
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
            features = torch.tensor(
                chunk[self.feature_columns].values,
                dtype=torch.long
            )
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
            df = pd.read_parquet(
                self.parquet_path,
                filters=[('file_id', '=', file_id)]
            )
            assert not df.isna().any().any(), "Found NaNs!"
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
                    yield {
                        'features': seq,
                        'timestamps': ts,
                    }

    def __len__(self):
        return self._length

def custom_collate_fn(batch: list):
    try:
        features = torch.stack([item['features'] for item in batch])
        timestamps = [item['timestamps'] for item in batch]
        return {'features': features, 'timestamps': timestamps}
    except Exception as e:
        logging.error(f"Error in collate_fn: {e}")
        return {}

# -------------------------- Experiment Runner --------------------------
def run_experiment(global_params, hyperparams, device):
    """
    Run training and CSV validation for a given hyperparameter combination.
    Returns a dictionary with the best F1 score, the best epoch, and the final threshold.
    """
    # Merge the global params with the current hyperparameters.
    params = global_params.copy()
    params.update(hyperparams)
    
    # Set stride according to stride_type.
    if params.get("stride_type", "equal") == "half":
        params['stride'] = params['seq_len'] // 2
    elif params.get("stride_type", "equal") == "quarter":
        params['stride'] = params['seq_len'] // 4
    else:
        params['stride'] = params['seq_len']

    params['dim_feedforward'] = 4 * params['d_model']
    logging.info("Starting experiment with hyperparameters:")
    for k, v in hyperparams.items():
        logging.info(f"{k}: {v}")
        
    model = TransformerAnomalyDetector(
        d_model=params['d_model'],
        d_feature=params['d_feature'],
        nhead=params['nhead'],
        num_encoder_layers=params['num_layers'],
        num_decoder_layers=params['num_layers'],
        dim_feedforward=params['dim_feedforward'],
        dropout=params['dropout']
    ).to(device)
    
    # Create training dataset & loader (using Parquet dataset here)
    train_dataset = ParquetSequenceDataset(
        parquet_path=params['train_parquet_path'],
        feature_columns=params['feature_columns'],
        seq_len=params['seq_len'],
        stride=params['stride'],
        shuffle_files=True,
        split='train'
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        collate_fn=custom_collate_fn
    )
    
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    best_f1 = 0.0
    best_epoch = -1
    epochs_no_improve = 0
    
    # For hyperparameter tuning, you might want to use fewer epochs.
    for epoch in range(1, params['num_epochs'] + 1):
        model.train()
        running_loss = 0.0
        num_batches = len(train_loader) if len(train_loader) > 0 else 1
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", dynamic_ncols=True)
        for i, batch in enumerate(progress_bar):
            # For speed during hyperparameter tuning, limit the number of batches.
            if i >= 1000:
                break
            features = batch['features'].to(device)
            optimizer.zero_grad()
            output = model(features)
            loss, recon_loss, cluster_loss = compute_loss(output, features, model, params['lambda_cluster'])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(
                batch_loss=f"{loss.item():.4f}",
                recon_loss=f"{recon_loss.item():.4f}",
                cluster_loss=f"{cluster_loss.item():.4f}"
            )
        avg_loss = running_loss / num_batches
        logging.info(f"Epoch {epoch} training loss: {avg_loss:.4f}")
        # Optionally, you can save checkpoints.
        # save_checkpoint(model, optimizer, epoch, params, filename=f"checkpoint_epoch_{epoch}.pth")
        
        # Update threshold using the CSV validation dataset.
        val_dataset = CSVValidationDataset(params['validation_csv_path'], seq_len=params['seq_len'], stride=params['stride'])
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=custom_collate_fn)
        params['threshold'] = update_threshold(model, val_loader, params, device, k=2.0)
        
        precision, recall, f1 = validate_csv(model, params, device)
        logging.info(f"Epoch {epoch} CSV Validation - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = recall
            best_epoch = epoch
            epochs_no_improve = 0
            # Optionally, save the best model.
            # save_checkpoint(model, optimizer, epoch, params, filename=f"checkpoint_best_{epoch}_f1_{f1:.4f}.pth")
            # logging.info(f"New best model saved with F1-Score: {f1:.4f}")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in F1-Score for {epochs_no_improve} epochs.")
            
        if epochs_no_improve >= params['patience']:
            logging.info("Early stopping triggered.")
            break
            
    return {
        "best_f1": f1,
        "best_epoch": best_epoch,
        "final_threshold": params['threshold'],
        "hyperparams": hyperparams
    }

# -------------------------- Optuna Objective Function --------------------------
def objective(trial):
    # Global parameters (defaults that remain constant)
    global_params = {
        'train_parquet_path': '/workspaces/thesis/data/pdsch_data_romes_clean/processed/unscaled_pdsch.parquet',
        'validation_csv_path': '/workspaces/thesis/data/pdsch_data_romes_clean/test/test.csv',
        'ground_truth_csv_path': '/workspaces/thesis/data/pdsch_data_romes_clean/test/test_gt.csv',
        'output_dir': './output',
        'feature_columns': ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"],
        'batch_size': 64,
        'num_epochs': 1,   # Use fewer epochs for tuning
        'patience': 3,
        # The following will be overwritten by hyperparameters:
        'dim_feedforward': 256,
        'seq_len': 32,
        'stride': 16,  # This will be set based on "stride_type".
        'alpha': 1.0,
        'beta': 3.0,
        'threshold': 5,
    }

    # Sample hyperparameters
    hyperparams = {
        'nhead': trial.suggest_categorical('nhead', [2, 4, 8]),
        'dropout': trial.suggest_categorical('dropout', [0.3, 0.4, 0.5]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True),
        'lambda_cluster': trial.suggest_int('lambda_cluster', 3, 10, step=1),
        'd_model': trial.suggest_categorical('d_model', [16, 32, 64, 128]),
        'd_feature': trial.suggest_categorical('d_feature', [4, 8, 16, 32]),
        'num_layers': trial.suggest_categorical('num_layers', [1, 2, 3]),
        'seq_len': trial.suggest_int('seq_len', 4, 32, step=4),
        'stride_type': trial.suggest_categorical('stride_type', ["equal", "half", "quarter"]),
        'alpha': trial.suggest_float('alpha', 1.0, 5.0, step=0.5),
        'beta': trial.suggest_float('beta', 1.0, 5.0, step=0.5),
    }

    # Check for valid combination: d_model must be divisible by nhead.
    if hyperparams['d_model'] % hyperparams['nhead'] != 0:
        raise optuna.exceptions.TrialPruned("Invalid d_model and nhead combination.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result = run_experiment(global_params, hyperparams, device)
    best_f1 = result["best_f1"]

    # We aim to maximize F1 score, so return it (Optuna minimizes by default,
    # so we return the negative if needed, or change direction in study creation)
    return best_f1

# -------------------------- Main with Optuna --------------------------
def main():
    # Create the output directory and start logging.
    global_params = {
        'output_dir': './output',
    }
    os.makedirs(global_params['output_dir'], exist_ok=True)
    start_logging(global_params)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create an Optuna study. We set direction="maximize" to maximize F1-Score.
    study = optuna.create_study(
        direction="maximize", 
        study_name="anomaly_detector_optuna",
        storage="sqlite:///optuna.db",  # Save study results to a SQLite database.
        load_if_exists=False
    )
    # Adjust number of trials as needed.
    study.optimize(objective, n_trials=30)

    logging.info("Hyperparameter search completed.")
    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best F1-Score: {study.best_trial.value}")
    logging.info("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        logging.info(f"    {key}: {value}")

if __name__ == '__main__':
    main()
