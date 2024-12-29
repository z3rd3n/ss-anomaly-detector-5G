import torch
from torch.utils.data import IterableDataset
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq

class ParquetSequenceDataset(IterableDataset):
    def __init__(
        self,
        parquet_path: str,
        feature_columns: List[str],
        seq_len: int,
        stride: Optional[int] = None,
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
            # Read all rows but only the file_id column
            parquet_file = pq.ParquetFile(self.parquet_path)
            file_info = parquet_file.read(columns=['file_id']).to_pandas()

            # Count how many rows in each file_id
            file_counts_df = file_info.groupby('file_id').size().reset_index(name='counts')

            total_lines = file_counts_df['counts'].sum()
            
            # How many lines we want in validation
            val_target = total_lines * self.validation_ratio

            # Shuffle file_id–size pairs
            rng = np.random.RandomState(self.seed)
            files_with_sizes = list(zip(file_counts_df['file_id'], file_counts_df['counts']))
            rng.shuffle(files_with_sizes)

            # Accumulate counts for the validation set
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
                f"Using {self.num_files} files for the {split} split "
                f"(split by line counts)."
            )

            # --- Compute approximate length so __len__ works for TQDM. ---
            # 1) Build a map: file_id -> count of lines in that file
            file_counts_map = dict(zip(file_counts_df['file_id'], file_counts_df['counts']))

            # 2) Sum up how many sequences you *might* get from each file
            #    This is just an approximation because you do actual chunk logic in _get_sequences_from_chunk()
            #    but for TQDM it should be acceptable.
            total_sequences = 0
            for fid in self.file_ids:
                n_lines = file_counts_map.get(fid, 0)
                # Each file yields (n_lines - seq_len) // stride (if > 0)
                possible_sequences = (n_lines - self.seq_len) // self.stride
                if possible_sequences > 0:
                    total_sequences += possible_sequences
            self._length = max(total_sequences, 0)

        except Exception as e:
            logging.error(f"Error reading Parquet file {self.parquet_path}: {e}")
            self.file_ids = []
            self.num_files = 0
            self._length = 0

    @staticmethod
    def create_train_val_splits(
        parquet_path: str,
        feature_columns: List[str],
        seq_len: int,
        stride: Optional[int] = None,
        shuffle_files: bool = True,
        validation_ratio: float = 0.2,
        seed: int = 42
    ):
        train_dataset = ParquetSequenceDataset(
            parquet_path=parquet_path,
            feature_columns=feature_columns,
            seq_len=seq_len,
            stride=stride,
            shuffle_files=shuffle_files,
            split='train',
            validation_ratio=validation_ratio,
            seed=seed
        )
        
        val_dataset = ParquetSequenceDataset(
            parquet_path=parquet_path,
            feature_columns=feature_columns,
            seq_len=seq_len,
            stride=stride,
            shuffle_files=shuffle_files,
            split='val',
            validation_ratio=validation_ratio,
            seed=seed
        )
        
        return train_dataset, val_dataset

    def _get_sequences_from_chunk(self, chunk: pd.DataFrame):
        try:
            features = torch.tensor(
                chunk[self.feature_columns].values,
                dtype=torch.float32
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

    def _process_file_id(self, file_id: int) -> Tuple[Optional[torch.Tensor], Optional[List[List[str]]]]:
        """Process a single file_id."""
        try:
            df = pd.read_parquet(
                self.parquet_path,
                filters=[('file_id', '=', file_id)]
            )
            return self._get_sequences_from_chunk(df)
        except Exception as e:
            logging.error(f"Error processing file_id {file_id}: {e}")
            return None, None

    def __iter__(self):
        """Iterator over the dataset."""
        worker_info = torch.utils.data.get_worker_info()
        file_ids = self.file_ids.copy()
        
        if self.shuffle_files:
            np.random.shuffle(file_ids)
        
        # Handle multiple workers
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
        """
        Let TQDM know how many *approximate* sequences we can iterate over.
        This is necessary for `tqdm(..., total=len(dataloader))`.
        """
        return self._length


def custom_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function to handle both tensor and string data."""
    try:
        features = torch.stack([item['features'] for item in batch])
        timestamps = [item['timestamps'] for item in batch]
        
        return {
            'features': features,
            'timestamps': timestamps,
        }
    except Exception as e:
        logging.error(f"Error in collate_fn: {e}")
        return {}
