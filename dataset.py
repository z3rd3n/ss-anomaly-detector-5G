import os
import json
import random
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq

EPS = 1e-8  # small constant to avoid divide-by-zero

def custom_collate_fn(batch):
    """
    Custom collate function to merge a list of samples into a batch.
    """
    try:
        features = torch.stack([item['features'] for item in batch])
        timestamps = [item['timestamps'] for item in batch]
        rule_flags = torch.stack([item['rule_flags'] for item in batch])
        return {
            'features': features,
            'timestamps': timestamps,
            'rule_flags': rule_flags
        }
    except Exception as e:
        logging.error(f"Error in collate_fn: {e}")
        return {}

class ParquetSequenceDataset(IterableDataset):
    def __init__(self,
                 parquet_path: str,
                 feature_columns: list,  # expected numeric feature columns (e.g., ['SFN', 'Slot', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI'])
                 seq_len: int,
                 stride: int = None,
                 shuffle_files: bool = True,
                 ratio: float = 0.2,  # ratio of total rows to include
                 seed: int = 42,
                 skip_anomalies: bool = True,
                 normalization_stats: dict = None):
        """
        Args:
            parquet_path: Path to the parquet file.
            feature_columns: List of column names to use as features.
            seq_len: Length of each sequence.
            stride: Step between sequences (if None, defaults to seq_len, i.e. no overlap).
            shuffle_files: Whether to shuffle the files before iterating.
            ratio: Fraction (0-1) of the total rows (across files) to use.
            seed: Random seed for shuffling.
            skip_anomalies: If True, skip sequences that cause errors, contain NaNs, or are flagged by rule_based_flags.
            normalization_stats: A dict containing 'means' and 'variances' for feature normalization.
        """
        self.parquet_path = parquet_path
        self.feature_columns = feature_columns  # e.g., ['SFN', 'Slot', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI']
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.shuffle_files = shuffle_files
        self.ratio = ratio
        self.seed = seed
        self.skip_anomalies = skip_anomalies
        self.normalization_stats = normalization_stats

        # Pre-read only the file_id column to compute per-file row counts efficiently.
        df_ids = pd.read_parquet(self.parquet_path, columns=['file_id'])
        self.total_rows = len(df_ids)
        file_counts = df_ids['file_id'].value_counts().to_dict()
        self.all_file_ids = sorted(list(file_counts.keys()))
        
        # Determine which file_ids to include based on the ratio.
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
        # Precompute number of sequences per file.
        self.file_seq_counts = {}
        total_seq = 0
        for fid in self.selected_file_ids:
            count = file_counts[fid]
            n_seq = max(0, (count - self.seq_len) // self.stride + 1)
            self.file_seq_counts[fid] = n_seq
            total_seq += n_seq
        self._length = total_seq

    def _normalize(self, features: torch.Tensor):
        """
        Normalize features using provided mean and variance statistics.
        """
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

    def rule_based_flags(self, seq_tensor: torch.Tensor) -> bool:
        """
        Apply custom rules to a sequence tensor.
        Return True if the sequence should be skipped.
        Replace the following logic with your own rules.
        Example: Skip if any feature is negative.
        """
        # Example rule: If any element is negative, flag this sequence.
        return (seq_tensor < 0).any().item()

    def __iter__(self):
        # Optionally shuffle the order of file_ids for each epoch.
        file_ids = self.selected_file_ids.copy()
        if self.shuffle_files:
            random.shuffle(file_ids)

        for fid in file_ids:
            # Efficiently read rows corresponding to the current file_id.
            df = pd.read_parquet(self.parquet_path, filters=[('file_id', '==', fid)])
            df = df.reset_index(drop=True)
            num_rows = len(df)
            if num_rows < self.seq_len:
                continue  # Skip files that are too short.

            # Retrieve timestamps (if available) and features.
            timestamps_all = df['timestamp_str'].tolist() if 'timestamp_str' in df.columns else ["" for _ in range(num_rows)]
            try:
                features_all = df[self.feature_columns].to_numpy()
            except Exception as e:
                logging.error(f"Error extracting feature columns from file_id {fid}: {e}")
                if self.skip_anomalies:
                    continue
                else:
                    raise e

            # Create sequences with the specified stride.
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

                # Skip sequence if it contains NaNs.
                if self.skip_anomalies and torch.isnan(seq_tensor).any():
                    continue

                seq_rule_flags = rule_based_flags(seq_tensor)  
                # Apply rule-based flag check.
                if self.skip_anomalies:
                    if seq_rule_flags:
                        continue
        
                norm_features = self._normalize(seq_tensor)
                rule_flags_tensor = seq_rule_flags.to(torch.int8)

                yield {
                    'features': norm_features,
                    'timestamps': seq_timestamps,
                    'rule_flags': rule_flags_tensor
                }

    def __len__(self):
        return self._length


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

    MAX_RETX = 4
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


