import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, Dataset


EPS = 1e-6
MAX_RETX = 5

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


class ParquetSequenceDataset(IterableDataset):
    def __init__(self,
                 parquet_path: str,
                 feature_columns: list,
                 seq_len: int,
                 stride: int = None,
                 shuffle_files: bool = True,
                 split: str = 'train',
                 validation_ratio: float = 0.2,
                 seed: int = 42,
                 skip_anomalies: bool = True,
                 normalization_stats: dict = None):
        super().__init__()
        self.parquet_path = Path(parquet_path)
        self.feature_columns = feature_columns
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.shuffle_files = shuffle_files
        self.split = split
        self.validation_ratio = validation_ratio
        self.seed = seed
        self.skip_anomalies = skip_anomalies
        self.normalization_stats = normalization_stats  # dict with 'means' and 'variances'
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
            # Extract raw features as integers.
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
                seq = features[start_idx:end_idx]
                # Skip sequences that contain rule-based anomalies.
                if rule_based_flags(seq).any() and self.skip_anomalies:
                    continue
                sequences.append(seq)
                sequence_timestamps.append(timestamps[start_idx:end_idx])
            if sequences:
                return torch.stack(sequences), sequence_timestamps
            return None, None
        except Exception as e:
            logging.error(f"Error extracting sequences from chunk: {e}")
            return None, None

    def _process_file_id(self, file_id: int):
        try:
            df = pd.read_parquet(self.parquet_path, columns=self.feature_columns + ['timestamp_str'], filters=[('file_id', '=', file_id)])
            
            if df.isna().any().any():
                raise ValueError("Found NaNs!")
            return self._get_sequences_from_chunk(df)
        except Exception as e:
            logging.error(f"Error processing file_id {file_id}: {e}")
            return None, None

    def _normalize(self, features: torch.Tensor):
        if self.normalization_stats is None:
            return features
        if isinstance(self.normalization_stats['means'], torch.Tensor):
            means = self.normalization_stats['means'].clone().detach().to(dtype=torch.float32, device=features.device)
        else:
            means = torch.tensor(self.normalization_stats['means'], dtype=torch.float32, device=features.device).clone().detach()

        if isinstance(self.normalization_stats['variances'], torch.Tensor):
            variances = self.normalization_stats['variances'].clone().detach().to(dtype=torch.float32, device=features.device)
        else:
            variances = torch.tensor(self.normalization_stats['variances'], dtype=torch.float32, device=features.device).clone().detach()

        std = torch.sqrt(variances + 1e-6)
        features_norm = (features.float() - means) / std
        return features_norm

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
                    # Normalize before yielding if stats are provided.
                    seq_norm = self._normalize(seq)
                    yield {'features': seq_norm, 'timestamps': ts}

    def __len__(self):
        return self._length

class CSVValidationDataset(Dataset):
    def __init__(self, csv_path, seq_len=32, stride=16, normalization_stats: dict = None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.stride = stride
        required_columns = ["timestamp_str", "SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"]
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' is missing from {csv_path}!")
        self.raw_features = self.df[["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"]].values.astype(np.int64)
        self.timestamps = self.df["timestamp_str"].tolist()
        self.normalization_stats = normalization_stats
        total = len(self.raw_features)
        self.num_sequences = max(0, (total - self.seq_len) // self.stride + 1)

    def _normalize(self, features: torch.Tensor):
        if self.normalization_stats is None:
            return features
        if isinstance(self.normalization_stats['means'], torch.Tensor):
            means = self.normalization_stats['means'].clone().detach().to(dtype=torch.float32, device=features.device)
        else:
            means = torch.tensor(self.normalization_stats['means'], dtype=torch.float32, device=features.device).clone().detach()

        if isinstance(self.normalization_stats['variances'], torch.Tensor):
            variances = self.normalization_stats['variances'].clone().detach().to(dtype=torch.float32, device=features.device)
        else:
            variances = torch.tensor(self.normalization_stats['variances'], dtype=torch.float32, device=features.device).clone().detach()

        std = torch.sqrt(variances + EPS)
        features_norm = (features.float() - means) / std
        return features_norm

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_len
        if end_idx > len(self.raw_features):
            end_idx = len(self.raw_features)
            start_idx = end_idx - self.seq_len
        seq = torch.tensor(self.raw_features[start_idx:end_idx], dtype=torch.long)
        return {'features': self._normalize(seq), 'timestamps': self.timestamps[start_idx:end_idx]}

def custom_collate_fn(batch):
    try:
        features = torch.stack([item['features'] for item in batch])
        timestamps = [item['timestamps'] for item in batch]
        return {'features': features, 'timestamps': timestamps}
    except Exception as e:
        logging.error(f"Error in collate_fn: {e}")
        return {}
