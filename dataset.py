# dataset.py
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, Dataset


EPS = 1e-6
MAX_RETX = 4

def rule_based_flags(sequence: torch.Tensor) -> torch.Tensor:
    """
    Use a Pandas DataFrame and groupby to flag anomalous timesteps in the sequence.
    The rules applied (to match method 2) are:
      - Unnecessary ReTx: If ReTx > 0 and the previous CRC was 1.
      - Missing ReTx: If ReTx == 0 and the previous CRC was 0.
      - New Data but No ReTx: If the previous CRC is 0 and the current NDI is different from the previous NDI.
      - Max ReTx Achieved: If ReTx equals (MAX_RETX + 1).
    
    Assumes the sequence tensor has shape (seq_len, 7) with columns:
      ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"]
    """
    # Convert tensor to numpy array and create DataFrame
    arr = sequence.cpu().numpy()
    columns = ["SFN", "Slot", "HARQ", "MCS", "CRC", "ReTx", "NDI"]
    df = pd.DataFrame(arr, columns=columns)
    
    # Use groupby to simulate SQL's LAG function.
    df['prev_crc'] = df.groupby('HARQ')['CRC'].shift(1)
    df['prev_ndi'] = df.groupby('HARQ')['NDI'].shift(1)
    
    # Compute anomaly flags per the specified rules:
    flag_unnecessary_retx = (df['ReTx'] > 0) & (df['prev_crc'] == 1)
    flag_missing_retx     = (df['ReTx'] == 0) & (df['prev_crc'] == 0)
    flag_new_data_no_retx = (df['prev_crc'] == 0) & (df['NDI'] != df['prev_ndi'])
    flag_max_retx         = (df['ReTx'] == (MAX_RETX))
    
    # Combine all conditions to create the final flag.
    flags = flag_unnecessary_retx | flag_missing_retx | flag_new_data_no_retx | flag_max_retx
    
    # Convert flags back to a Torch tensor and return.
    return torch.tensor(flags.values, dtype=torch.bool)


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

def custom_collate_fn(batch):
    try:
        features = torch.stack([item['features'] for item in batch])
        timestamps = [item['timestamps'] for item in batch]
        return {'features': features, 'timestamps': timestamps}
    except Exception as e:
        logging.error(f"Error in collate_fn: {e}")
        return {}
