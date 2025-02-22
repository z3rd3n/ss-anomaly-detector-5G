# dataset.py
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
                 skip_anomalies: bool = False,  # if True, skip sequences containing anomalies
                 normalization_stats: dict = None):
        """
        Loads sequences from a parquet file.
        Each sample includes a 'labels' tensor derived from the 'insight' column.
        
        The 'insight' column is assumed to be a comma-separated string 
        (e.g. "max_retx_achieved, unnecessary_retx") for anomalous rows and empty for normal rows.
        
        New deterministic mapping (no composite labeling):
          - Empty or missing insight: 0
          - unnecessary_retx: 1 
          - missing_retx: 2 
          - new_data_no_retx: 3 
          - max_retx_achieved: 4
          
        In cases where multiple anomalies occur in a single timestamp, if one of them
        is max_retx_achieved and there is at least one other anomaly, the max_retx_achieved 
        is ignored and the remaining anomaly with the smallest mapping value is chosen.
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

        # Define the fixed, deterministic mapping.
        self.anomaly_mapping = {
            "unnecessary_retx": 1,
            "missing_retx": 2,
            "new_data_no_retx": 3,
            "max_retx_achieved": 4,
        }

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

        # Report the class distribution from the selected training files.
        self.report_class_counts()

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

    def insight_to_label(self, insight: str) -> int:
        """
        Converts an insight string into a deterministic integer label.
        
        - Empty or missing insight yields label 0.
        - Otherwise, splits the string on commas and strips whitespace.
          If multiple anomalies exist and one of them is "max_retx_achieved",
          that anomaly is ignored (provided at least one other exists).
          From the remaining anomalies (or the single anomaly if only one exists),
          the one with the smallest corresponding mapping value is chosen.
        """
        if pd.isna(insight) or insight.strip() == "":
            return 0

        anomalies = [a.strip() for a in insight.split(",") if a.strip()]
        if not anomalies:
            return 0

        # If multiple anomalies exist and one is max_retx_achieved, remove it.
        if len(anomalies) > 1 and "max_retx_achieved" in anomalies:
            anomalies = [a for a in anomalies if a != "max_retx_achieved"]
        if len(anomalies) > 1:
            logging.info(f"Multiple anomalies found: {anomalies}")

        # Ensure there is only one anomaly.
        candidate_labels = [self.anomaly_mapping[a] for a in anomalies if a in self.anomaly_mapping]
        if len(candidate_labels) != 1:
            raise ValueError(f"Multiple or no valid anomalies found: {anomalies}")
        return candidate_labels[0]
    
    def report_class_counts(self):
        """
        Goes through all selected training files, counts the number of rows for each class,
        and logs the distribution. Also checks that the total number of labels matches the number of rows.
        """
        counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        total_labels = 0
        total_rows = 0
        for fid in self.selected_file_ids:
            try:
                df = pd.read_parquet(self.parquet_path, columns=['insight'], filters=[('file_id', '==', fid)])
            except Exception as e:
                logging.error(f"Error reading file {fid} for class counts: {e}")
                continue
            for insight in df['insight']:
                label = self.insight_to_label(insight)
                counts[label] += 1
                total_labels += 1
            total_rows += len(df)
        
        if total_labels != total_rows:
            logging.error(f"Total number of labels ({total_labels}) does not match total number of rows ({total_rows}).")
        else:
            logging.info(f"Total number of labels matches total number of rows: {total_rows}")

        logging.info("Class distribution in training files:")
        reverse_mapping = {v: k for k, v in self.anomaly_mapping.items()}
        reverse_mapping[0] = "normal"
        self.counts = counts
        for label, cnt in counts.items():
            label_name = reverse_mapping.get(label, "unknown")
            logging.info(f"  {label} ({label_name}): {cnt}")

    def __iter__(self):
        file_ids = self.selected_file_ids.copy()
        if self.shuffle_files:
            random.shuffle(file_ids)

        # Iterate over each selected file.
        for fid in file_ids:
            df = pd.read_parquet(self.parquet_path, filters=[('file_id', '==', fid)])
            df = df.reset_index(drop=True)
            num_rows = len(df)
            if num_rows < self.seq_len:
                continue

            # Get timestamps.
            timestamps_all = df['timestamp_str'].tolist() if 'timestamp_str' in df.columns else ["" for _ in range(num_rows)]
            
            # Get feature values.
            try:
                features_all = df[self.feature_columns].to_numpy()
            except Exception as e:
                logging.error(f"Error extracting features from file_id {fid}: {e}")
                if self.skip_anomalies:
                    continue
                else:
                    raise e

            # Read the 'insight' column (defaulting to empty strings if missing).
            if 'insight' in df.columns:
                insights_all = df['insight'].tolist()
            else:
                insights_all = ["" for _ in range(num_rows)]
            
            # Generate sequences.
            for start in range(0, num_rows - self.seq_len + 1, self.stride):
                end = start + self.seq_len
                seq_features = features_all[start:end]
                seq_timestamps = timestamps_all[start:end]
                seq_insights = insights_all[start:end]
                try:
                    seq_tensor = torch.tensor(seq_features, dtype=torch.float32)
                except Exception as e:
                    logging.error(f"Error converting features to tensor for file_id {fid} rows {start}:{end}: {e}")
                    if self.skip_anomalies:
                        continue
                    else:
                        raise e

                # Convert insight strings into deterministic labels.
                labels = torch.tensor(
                    [self.insight_to_label(insight) for insight in seq_insights],
                    dtype=torch.long
                )
                
                # Optionally, skip sequences that contain any anomalies.
                if self.skip_anomalies and (labels != 0).any():
                    continue

                norm_features = self._normalize(seq_tensor)

                yield {
                    'features': norm_features,
                    'timestamps': seq_timestamps,
                    'labels': labels
                }

    def __len__(self):
        return self._length
