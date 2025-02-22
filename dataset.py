#dataset.py 
import random
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

EPS = 1e-8  # small constant

def custom_collate_fn(batch):
    """
    Merges a list of samples into a batch.
    Now also stacks 'labels' (per timestamp) if present.
    """
    try:
        features = torch.stack([item['features'] for item in batch])
        timestamps = [item['timestamps'] for item in batch]
        labels = torch.stack([item['labels'] for item in batch]) if 'labels' in batch[0] else None
        out = {'features': features, 'timestamps': timestamps}
        if labels is not None:
            out['labels'] = labels
        return out
    except Exception as e:
        logging.error(f"Error in collate_fn: {e}")
        return {}

class ParquetSequenceDataset(IterableDataset):
    def __init__(self,
                 parquet_path: str,
                 feature_columns: list,  # e.g., ['SFN', 'Slot', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI']
                 seq_len: int,
                 stride: int = None,
                 shuffle_files: bool = True,
                 ratio: float = 0.2,
                 seed: int = 42,
                 skip_anomalies: bool = False,  # include anomalies for classification
                 normalization_stats: dict = None):
        """
        Loads sequences from a parquet file.
        Each sample now includes a 'labels' tensor computed by rule_based_labels().
        """
        self.parquet_path = parquet_path
        self.feature_columns = feature_columns
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.shuffle_files = shuffle_files
        self.ratio = ratio
        self.seed = seed
        self.skip_anomalies = skip_anomalies
        self.normalization_stats = normalization_stats

        # Pre-read file_id column to compute row counts.
        df_ids = pd.read_parquet(self.parquet_path, columns=['file_id'])
        self.total_rows = len(df_ids)
        file_counts = df_ids['file_id'].value_counts().to_dict()
        self.all_file_ids = sorted(list(file_counts.keys()))
        
        if self.shuffle_files:
            random.seed(self.seed)
            file_ids_shuffled = self.all_file_ids.copy()
            random.shuffle(file_ids_shuffled)
        else:
            file_ids_shuffled = self.all_file_ids

        cumulative = 0
        self.selected_file_ids = []
        threshold = self.ratio * self.total_rows
        for fid in file_ids_shuffled:
            cumulative += file_counts[fid]
            self.selected_file_ids.append(fid)
            if cumulative >= threshold:
                break
        
        logging.info(f"Selected {len(self.selected_file_ids)} files for training.")
        self.file_seq_counts = {}
        total_seq = 0
        for fid in self.selected_file_ids:
            count = file_counts[fid]
            n_seq = max(0, (count - self.seq_len) // self.stride + 1)
            self.file_seq_counts[fid] = n_seq
            total_seq += n_seq
        self._length = total_seq

    def _normalize(self, features: torch.Tensor):
        if not self.normalization_stats:
            return features

        means = self.normalization_stats.get('means')
        variances = self.normalization_stats.get('variances')
        if not torch.is_tensor(means):
            means = torch.tensor(means, dtype=torch.float32, device=features.device)
        if not torch.is_tensor(variances):
            variances = torch.tensor(variances, dtype=torch.float32, device=features.device)

        std = torch.sqrt(variances + EPS)
        return (features.float() - means) / std

    def __iter__(self):
        file_ids = self.selected_file_ids.copy()
        if self.shuffle_files:
            random.shuffle(file_ids)

        for fid in file_ids:
            df = pd.read_parquet(self.parquet_path, filters=[('file_id', '==', fid)])
            df = df.reset_index(drop=True)
            num_rows = len(df)
            if num_rows < self.seq_len:
                continue

            timestamps_all = df['timestamp_str'].tolist() if 'timestamp_str' in df.columns else ["" for _ in range(num_rows)]
            try:
                features_all = df[self.feature_columns].to_numpy()
            except Exception as e:
                logging.error(f"Error extracting features from file_id {fid}: {e}")
                if self.skip_anomalies:
                    continue
                else:
                    raise e

            for start in range(0, num_rows - self.seq_len + 1, self.stride):
                end = start + self.seq_len
                seq_features = features_all[start:end]
                seq_timestamps = timestamps_all[start:end]
                try:
                    seq_tensor = torch.tensor(seq_features, dtype=torch.float32)
                except Exception as e:
                    logging.error(f"Error converting features to tensor for file_id {fid} rows {start}:{end}: {e}")
                    if self.skip_anomalies:
                        continue
                    else:
                        raise e

                if self.skip_anomalies and torch.isnan(seq_tensor).any():
                    continue

                # Compute per-timestamp rule-based labels.
                labels = rule_based_labels(seq_tensor)  # tensor of shape [seq_len]
                
                norm_features = self._normalize(seq_tensor)

                yield {
                    'features': norm_features,
                    'timestamps': seq_timestamps,
                    'labels': labels
                }

    def __len__(self):
        return self._length

def rule_based_labels(sequence: torch.Tensor) -> torch.Tensor:
    """
    Computes rule-based anomaly labels for each time step using tensor operations.
    
    Assumes the input tensor `sequence` has shape [T, 7] with columns:
      0: SFN, 1: Slot, 2: HARQ, 3: MCS, 4: CRC, 5: ReTx, 6: NDI.
    
    Labels:
      0: Normal
      1: Unnecessary ReTx (ReTx increased when previous CRC == 1)
      2: Missing ReTx (for new data events: previous CRC == 0, NDI changed, but ReTx did not increase relative to previous HARQ)
      3: New Data (for new data events: previous CRC == 0, NDI changed, and ReTx increased relative to previous HARQ)
      4: Max ReTx Achieved (ReTx == MAX_RETX)
      5: Likely Anomaly (if needed)
    """
    MAX_RETX = 4
    T = sequence.size(0)
    
    # Extract relevant columns.
    HARQ = sequence[:, 2]
    CRC = sequence[:, 4]
    ReTx = sequence[:, 5]
    NDI = sequence[:, 6]
    
    # Initialize previous values with a placeholder (-1).
    prev_crc = torch.full((T,), -1, dtype=CRC.dtype, device=CRC.device)
    prev_ndi = torch.full((T,), -1, dtype=NDI.dtype, device=NDI.device)
    prev_retx = torch.full((T,), -1, dtype=ReTx.dtype, device=ReTx.device)
    
    # For each HARQ group, shift the previous values.
    unique_harq = torch.unique(HARQ)
    for h in unique_harq:
        indices = (HARQ == h).nonzero(as_tuple=False).squeeze(1)
        if indices.numel() > 1:
            prev_crc[indices[1:]] = CRC[indices[:-1]]
            prev_ndi[indices[1:]] = NDI[indices[:-1]]
            prev_retx[indices[1:]] = ReTx[indices[:-1]]
    
    # Initialize labels as Normal.
    labels = torch.zeros(T, dtype=torch.long, device=sequence.device)
    
    # Rule 4: Max ReTx Achieved.
    cond_max = (ReTx >= MAX_RETX)
    labels[cond_max] = 4
    
    # Rule 1: Unnecessary ReTx (ReTx increased compared to previous HARQ when previous CRC == 1).
    cond_unnecessary = (ReTx > prev_retx) & (prev_crc == 1)
    labels[cond_unnecessary] = 1
    
    # Rule 2: Missing ReTx but no new data
    cond_missing = (prev_crc == 0) & (ReTx == prev_retx) & (NDI == prev_ndi)
    labels[cond_missing] = 2
    
    # Rule 3: New data but no ReTx increase
    cond_new_data_actual = (prev_crc == 0) & (ReTx == prev_retx) & (NDI != prev_ndi)
    labels[cond_new_data_actual] = 3
    
    return labels


