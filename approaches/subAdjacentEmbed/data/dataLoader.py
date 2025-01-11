import torch
from torch.utils.data import IterableDataset
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq


class CategoricalParquetSequenceDataset(IterableDataset):
    def __init__(
        self,
        parquet_path: str,
        feature_config: Dict[str, Dict],
        seq_len: int,
        stride: Optional[int] = None,
        shuffle_files: bool = True,
        split: str = 'train',
        validation_ratio: float = 0.2,
        seed: int = 42
    ):
        """
        Args:
            parquet_path: path to the Parquet file.
            feature_config: the dictionary loaded from 'feature_mappings.json' 
                            or similar, i.e. {feat_name: {"value_to_index": {...}, ...}}.
            seq_len: how many rows go into each chunked sequence.
            stride: how far to move between one sequence and the next.
            shuffle_files: shuffle the file_ids or not.
            split: 'train' or 'val'.
            validation_ratio: fraction of total lines used for validation.
            seed: random seed for reproducibility.
        """
        super().__init__()
        self.parquet_path = Path(parquet_path)
        self.feature_config = feature_config
        self.feature_names = list(feature_config.keys())
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.shuffle_files = shuffle_files
        self.split = split
        self.validation_ratio = validation_ratio
        self.seed = seed

        # Attempt reading the Parquet metadata to figure out file_id splits
        try:
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

            if self.split == 'train':
                self.file_ids = train_file_ids
            else:
                self.file_ids = val_file_ids

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

    def __len__(self):
        # Approximate total number of sequences
        return self._length

    @staticmethod
    def create_train_val_splits(
        parquet_path: str,
        feature_config: Dict[str, Dict],
        seq_len: int,
        stride: Optional[int] = None,
        shuffle_files: bool = True,
        validation_ratio: float = 0.2,
        seed: int = 42
    ):
        """
        Helper to create train and validation datasets easily.
        """
        train_dataset = CategoricalParquetSequenceDataset(
            parquet_path=parquet_path,
            feature_config=feature_config,
            seq_len=seq_len,
            stride=stride,
            shuffle_files=shuffle_files,
            split='train',
            validation_ratio=validation_ratio,
            seed=seed
        )
        val_dataset = CategoricalParquetSequenceDataset(
            parquet_path=parquet_path,
            feature_config=feature_config,
            seq_len=seq_len,
            stride=stride,
            shuffle_files=shuffle_files,
            split='val',
            validation_ratio=validation_ratio,
            seed=seed
        )
        return train_dataset, val_dataset

    def _encode_feature(self, feat_name: str, series: pd.Series) -> torch.LongTensor:
        """
        Given a feature name and the corresponding Series from the chunk,
        map each value to an integer index using `value_to_index`.
        """
        # If the feature is not in feature_config for any reason, fallback
        if feat_name not in self.feature_config:
            logging.warning(f"{feat_name} not in feature_config; fallback to raw numeric.")
            return torch.tensor(series.values, dtype=torch.long)

        value_to_idx = self.feature_config[feat_name]['value_to_index']
        # For each row in 'series', map the value to the index
        # If there's any unknown, map to a special index if needed (or set -1).
        mapped_values = series.apply(
            lambda val: value_to_idx.get(str(val), 0)  # or some default index (e.g. 0)
        ).values

        return torch.tensor(mapped_values, dtype=torch.long)

    def _get_sequences_from_chunk(self, df: pd.DataFrame):
        """
        From a chunk of rows, create sliding windows of length `seq_len` with stride `self.stride`.
        Return:
           sequences: a list of dictionaries (one dict per step in the window)
           timestamps: a list of lists (one list of timestamps per window)
        """
        try:
            # Convert entire chunk's columns to Tensors for each feature
            # Each feature => LongTensor of shape [chunk_size]
            feat_arrays = {}
            for feat in self.feature_names:
                feat_arrays[feat] = self._encode_feature(feat, df[feat])

            timestamps = df['timestamp_str'].tolist()
            n_sequences = (len(df) - self.seq_len) // self.stride

            windowed_feature_dicts = []
            windowed_timestamps = []

            for i in range(n_sequences):
                start_idx = i * self.stride
                end_idx = start_idx + self.seq_len
                if end_idx > len(df):
                    break

                # Build one dictionary for the entire sequence
                seq_feat_dict = {}
                for feat in self.feature_names:
                    seq_feat_dict[feat] = feat_arrays[feat][start_idx:end_idx]  # shape [seq_len]

                windowed_feature_dicts.append(seq_feat_dict)
                windowed_timestamps.append(timestamps[start_idx:end_idx])

            if len(windowed_feature_dicts) == 0:
                return None, None
            return windowed_feature_dicts, windowed_timestamps

        except Exception as e:
            logging.error(f"Error extracting sequences from chunk: {e}")
            return None, None

    def _process_file_id(self, file_id: int):
        """
        Read the parquet for one file_id, build sequences for that file.
        Returns:
          a list of dictionaries of shape [#windows], each of the form { 'SFN':..., 'Slot':..., ... }
          a list of list of timestamps
        """
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
        """
        Iterate over the file_ids, producing each window as:
        {
           'features': { 'SFN': Tensor([seq_len]), 'Slot': Tensor([seq_len]), ..., },
           'timestamps': [ts1, ts2, ..., ts(seq_len)]
        }
        """
        worker_info = torch.utils.data.get_worker_info()
        file_ids = self.file_ids.copy()

        if self.shuffle_files:
            np.random.shuffle(file_ids)

        # If multi-worker, split up file_ids
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            file_ids = np.array_split(file_ids, num_workers)[worker_id]

        for fid in file_ids:
            sequences_list, timestamps_list = self._process_file_id(fid)
            if sequences_list is None or timestamps_list is None:
                continue
            # each element in sequences_list is a dict of shape {feat_name: 1D Tensor(seq_len)}, 
            # and timestamps_list is a list of lists
            for features_dict, ts in zip(sequences_list, timestamps_list):
                yield {
                    'features': features_dict,  # dict
                    'timestamps': ts            # list of str
                }


def custom_collate_fn_categorical(batch):
    # batch[i] = {
    #   'features': { 'SFN': LongTensor(seq_len), 'Slot': LongTensor(seq_len), ... },
    #   'timestamps': [list of seq_len strings]
    # }

    # Collect feature names from the first sample
    feature_names = list(batch[0]['features'].keys())
    collated_features = {f: [] for f in feature_names}
    collated_timestamps = []

    for item in batch:
        for f in feature_names:
            collated_features[f].append(item['features'][f])  # Already LongTensor
        collated_timestamps.append(item['timestamps'])        # list of str

    # Now stack each feature list
    for f in feature_names:
        collated_features[f] = torch.stack(collated_features[f], dim=0)
        # shape [batch_size, seq_len]

    return {
        'features': collated_features,
        'timestamps': collated_timestamps,
    }
