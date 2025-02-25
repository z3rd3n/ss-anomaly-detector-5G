import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import random

EPS = 1e-8  # small constant

def custom_collate_fn(batch):
    """
    Merges a list of samples into a batch.
    """
    if not batch:
        return {}
    
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
        for i, item in enumerate(batch):
            logging.error(f"Item {i} keys: {item.keys()}")
            for k, v in item.items():
                if isinstance(v, torch.Tensor):
                    logging.error(f"Item {i} {k} shape: {v.shape}")
        return {}

class ParquetSequenceDataset(Dataset):
    def __init__(self,
                 parquet_path: str,
                 feature_columns: list,
                 seq_len: int,
                 stride: int = None,
                 ratio: float = 1.0,
                 seed: int = 42,
                 skip_anomalies: bool = False,
                 normalization_stats: dict = None,
                 pre_normalize: bool = True,
                 add_sequence_boundaries: bool = True):
        """
        Loads sequences from a parquet file.
        
        Args:
            parquet_path: Path to the parquet file
            feature_columns: List of column names to use as features
            seq_len: Length of each sequence
            stride: Stride between sequences (default: equal to seq_len)
            ratio: Portion of data to use (1.0 = all data)
            seed: Random seed for reproducibility
            skip_anomalies: If True, skip sequences containing anomalies
            normalization_stats: Dictionary with 'means' and 'variances' for normalization
            pre_normalize: If True, normalize features at load time
            add_sequence_boundaries: If True, add boundaries between sequences to handle temporal edges
        """
        super().__init__()
        self.parquet_path = parquet_path
        self.feature_columns = feature_columns
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.skip_anomalies = skip_anomalies
        self.normalization_stats = normalization_stats
        self.pre_normalize = pre_normalize
        self.add_sequence_boundaries = add_sequence_boundaries

        # Define mapping for anomaly labels
        self.anomaly_mapping = {
            "unnecessary_retx": 1,
            "missing_retx": 2,
            "new_data_no_retx": 3,
            "max_retx_achieved": 4,
            "none_of_them": 5  # Extra class for anomalies that don't fit other categories
        }

        # Load data and prepare sequences
        self._load_data(ratio, seed)
        self._prepare_sequences()
        self._report_class_counts()

    def _load_data(self, ratio, seed):
        """Load data from the parquet file"""
        logging.info(f"Loading data from {self.parquet_path}")
        
        # Sample files if ratio < 1.0
        if ratio < 1.0:
            df_ids = pd.read_parquet(self.parquet_path, columns=['file_id'])
            unique_file_ids = df_ids['file_id'].unique()
            
            np.random.seed(seed)
            num_files = max(1, int(len(unique_file_ids) * ratio))
            sampled_files = np.random.choice(unique_file_ids, num_files, replace=False)
            
            # Load only sampled files
            self.df = pd.read_parquet(
                self.parquet_path,
                filters=[('file_id', 'in', sampled_files.tolist())]
            )
        else:
            # Load all data
            self.df = pd.read_parquet(self.parquet_path)
        
        # Ensure required columns exist
        if 'insight' not in self.df.columns:
            self.df['insight'] = ""
        
        if 'timestamp_str' not in self.df.columns:
            logging.warning("No timestamp_str column found in parquet file, creating dummy timestamps")
            self.df['timestamp_str'] = [f"dummy_{i}" for i in range(len(self.df))]
        
        # Extract and potentially normalize features
        features = self.df[self.feature_columns].values
        
        # Normalize features if requested
        if self.normalization_stats and self.pre_normalize:
            means = np.array(self.normalization_stats.get('means'))
            variances = np.array(self.normalization_stats.get('variances'))
            std = np.sqrt(variances + EPS)
            features = (features - means) / std
        
        # Convert to torch tensor
        self.features = torch.tensor(features, dtype=torch.float32)
        
        # Process labels
        self.labels = torch.tensor([
            self.insight_to_label(insight) for insight in self.df['insight']
        ], dtype=torch.long)
        
        # Store timestamps
        self.timestamps = self.df['timestamp_str'].tolist()
        
        logging.info(f"Loaded {len(self.df)} rows from the dataset")

    def _prepare_sequences(self):
        """Pre-compute all valid sequence indices"""
        self.sequences = []
        
        # Group by file_id to ensure sequences come from the same file
        grouped_indices = {}
        if 'file_id' in self.df.columns:
            grouped_indices = self.df.groupby('file_id').groups
        else:
            # If no file_id, treat all data as one file
            grouped_indices = {'dummy_file': pd.Index(range(len(self.df)))}
        
        for file_id, group_indices in grouped_indices.items():
            indices = list(group_indices)
            num_rows = len(indices)
            
            if num_rows < self.seq_len:
                continue
            
            # Generate valid sequence indices
            for start_idx in range(0, num_rows - self.seq_len + 1, self.stride):
                seq_indices = indices[start_idx:start_idx + self.seq_len]
                
                # Skip sequences with anomalies if requested
                if self.skip_anomalies:
                    seq_labels = self.labels[seq_indices]
                    if (seq_labels != 0).any():
                        continue
                        
                self.sequences.append(seq_indices)
        
        if self.add_sequence_boundaries:
            # Add special handling for sequence boundaries
            self._handle_sequence_boundaries(grouped_indices)
        
        logging.info(f"Generated {len(self.sequences)} sequences")

    def _handle_sequence_boundaries(self, grouped_indices):
        """
        Add sequences that cross file boundaries to handle temporal dependencies.
        This creates overlapping sequences at the start of each file to ensure
        that anomalies at file boundaries are not missed.
        """
        if len(grouped_indices) <= 1:
            return  # No need for boundaries if only one file
        
        # For each file, try to include some context from previous sequences
        context_len = min(self.seq_len // 2, self.stride)
        
        boundary_sequences = []
        for file_id, group_indices in grouped_indices.items():
            indices = list(group_indices)
            
            # Skip very short files
            if len(indices) < self.seq_len - context_len:
                continue
            
            # Generate sequences that bridge across files
            for overlap in range(1, context_len + 1):
                start_idx = overlap
                if start_idx >= len(indices):
                    continue
                
                # Create sequence with overlap
                pre_indices = [-1] * overlap  # Use -1 as a sentinel value
                seq_indices = pre_indices + indices[:self.seq_len - overlap]
                
                # Add to boundary sequences
                boundary_sequences.append(seq_indices)
        
        # Add boundary sequences to main sequences list
        if boundary_sequences:
            logging.info(f"Added {len(boundary_sequences)} boundary sequences to handle temporal edges")
            self.sequences.extend(boundary_sequences)

    def insight_to_label(self, insight):
        """Convert insight string to integer label"""
        if pd.isna(insight) or insight.strip() == "":
            return 0

        # Parse anomalies from the insight string
        anomalies = [a.strip() for a in insight.split(",") if a.strip()]
        if not anomalies:
            return 0

        # Special case: remove max_retx_achieved if other anomalies exist
        if len(anomalies) > 1 and "max_retx_achieved" in anomalies:
            anomalies = [a for a in anomalies if a != "max_retx_achieved"]
        
        # Get valid anomaly labels
        valid_labels = [self.anomaly_mapping.get(a) for a in anomalies 
                        if a in self.anomaly_mapping]
        
        # Return 0 if no valid anomalies
        if not valid_labels:
            return 0
            
        # Return the smallest valid label (prioritize certain anomalies)
        return min(valid_labels)
    
    def _report_class_counts(self):
        """Report class distribution in the dataset"""
        unique_labels, counts = torch.unique(self.labels, return_counts=True)
        
        logging.info("Class distribution in the dataset:")
        reverse_mapping = {v: k for k, v in self.anomaly_mapping.items()}
        reverse_mapping[0] = "normal"
        
        # Store counts for future reference
        self.counts = {label.item(): count.item() 
                      for label, count in zip(unique_labels, counts)}
        
        # Log the counts
        for label, count in self.counts.items():
            label_name = reverse_mapping.get(label, "unknown")
            logging.info(f"  {label} ({label_name}): {count}")

    def _normalize(self, features):
        """Normalize features on-the-fly if not pre-normalized"""
        if not self.normalization_stats or self.pre_normalize:
            return features

        means = self.normalization_stats.get('means')
        variances = self.normalization_stats.get('variances')
        
        if not torch.is_tensor(means):
            means = torch.tensor(means, dtype=torch.float32, device=features.device)
        if not torch.is_tensor(variances):
            variances = torch.tensor(variances, dtype=torch.float32, device=features.device)

        std = torch.sqrt(variances + EPS)
        return (features.float() - means) / std

    def __getitem__(self, index):
        """Get a sequence by index"""
        seq_indices = self.sequences[index]
        
        # Check for boundary sequences with sentinel values
        has_sentinels = any(idx < 0 for idx in seq_indices)
        
        if has_sentinels:
            # Create new tensors with padding for sentinel values
            valid_mask = [idx >= 0 for idx in seq_indices]
            valid_indices = [idx for idx in seq_indices if idx >= 0]
            
            # Get valid features and pad with zeros
            valid_features = self.features[valid_indices]
            seq_features = torch.zeros(self.seq_len, len(self.feature_columns), dtype=valid_features.dtype)
            seq_features[sum(not x for x in valid_mask):] = valid_features
            
            # Get valid labels and pad with -1 (ignored in loss)
            valid_labels = self.labels[valid_indices]
            seq_labels = torch.ones(self.seq_len, dtype=valid_labels.dtype) * -1
            seq_labels[sum(not x for x in valid_mask):] = valid_labels
            
            # Get valid timestamps and pad with empty strings
            valid_timestamps = [self.timestamps[i] for i in valid_indices]
            seq_timestamps = [""] * sum(not x for x in valid_mask) + valid_timestamps
        else:
            # Normal sequence without padding
            seq_features = self.features[seq_indices]
            seq_labels = self.labels[seq_indices]
            seq_timestamps = [self.timestamps[i] for i in seq_indices]
        
        # Normalize if needed and not already done
        if self.normalization_stats and not self.pre_normalize:
            seq_features = self._normalize(seq_features)
        
        return {
            'features': seq_features,
            'timestamps': seq_timestamps,
            'labels': seq_labels
        }

    def __len__(self):
        """Return the number of sequences"""
        return len(self.sequences)


class OverSampledSequenceDataset(ParquetSequenceDataset):
    """
    Dataset that oversamples minority classes (anomalies) to handle class imbalance.
    """
    def __init__(self, 
                 parquet_path: str,
                 feature_columns: list,
                 seq_len: int,
                 stride: int = None,
                 ratio: float = 1.0,
                 seed: int = 42,
                 normalization_stats: dict = None,
                 oversample_factors: dict = None):
        """
        Args:
            oversample_factors: Dictionary mapping class indices to oversampling factors
                                e.g., {1: 5.0, 2: 10.0} would oversample class 1 by 5x and class 2 by 10x
        """
        # First load the data normally
        super().__init__(
            parquet_path=parquet_path,
            feature_columns=feature_columns,
            seq_len=seq_len,
            stride=stride,
            ratio=ratio,
            seed=seed,
            skip_anomalies=False,
            normalization_stats=normalization_stats
        )
        
        # Default oversample factors if not provided
        if oversample_factors is None:
            # Default to oversample anomaly classes proportional to class imbalance
            normal_count = self.counts.get(0, 1)
            self.oversample_factors = {
                cls: min(20.0, normal_count / count / 2) 
                for cls, count in self.counts.items() 
                if cls != 0 and count > 0
            }
        else:
            self.oversample_factors = oversample_factors
        
        # Apply oversampling
        self._apply_oversampling(seed)
    
    def _apply_oversampling(self, seed):
        """Apply oversampling to minority classes"""
        # Group sequences by their majority class
        seq_by_class = defaultdict(list)
        
        for idx, seq_indices in enumerate(self.sequences):
            # Get labels for this sequence
            seq_labels = self.labels[seq_indices]
            
            # Count class occurrences
            label_counts = torch.bincount(
                seq_labels, 
                minlength=max(self.anomaly_mapping.values()) + 2
            )
            
            # Determine majority class (skip normal class if any anomalies exist)
            anomaly_counts = label_counts[1:]  # Skip normal class
            if anomaly_counts.sum() > 0:
                # If any anomalies, use the most frequent anomaly class
                majority_class = anomaly_counts.argmax().item() + 1
            else:
                # Otherwise, use normal class
                majority_class = 0
            
            # Add to appropriate list
            seq_by_class[majority_class].append(idx)
        
        # Log original class distribution
        logging.info("Sequence distribution before oversampling:")
        for cls, sequences in seq_by_class.items():
            logging.info(f"  Class {cls}: {len(sequences)} sequences")
        
        # Create oversampled sequence list
        oversampled_sequences = list(self.sequences)  # Start with original sequences
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Oversample minority classes
        for cls, factor in self.oversample_factors.items():
            if cls not in seq_by_class or factor <= 1.0:
                continue
            
            # Get sequences for this class
            class_sequences = seq_by_class[cls]
            if not class_sequences:
                continue
            
            # Determine number of additional sequences to add
            n_orig = len(class_sequences)
            n_target = int(n_orig * factor) - n_orig
            
            # Sample with replacement
            if n_target > 0:
                additional_indices = random.choices(class_sequences, k=n_target)
                for idx in additional_indices:
                    oversampled_sequences.append(self.sequences[idx])
        
        # Replace original sequences with oversampled ones
        self.sequences = oversampled_sequences
        
        # Log oversampled distribution
        logging.info(f"Oversampled dataset from {len(self.sequences) - len(oversampled_sequences) + len(self.sequences)} to {len(self.sequences)} sequences")
        
        # Recompute class distribution for logging
        oversampled_by_class = defaultdict(int)
        for seq_indices in self.sequences:
            seq_labels = self.labels[seq_indices]
            label_counts = torch.bincount(
                seq_labels, 
                minlength=max(self.anomaly_mapping.values()) + 2
            )
            if label_counts[1:].sum() > 0:
                majority_class = label_counts[1:].argmax().item() + 1
            else:
                majority_class = 0
            oversampled_by_class[majority_class] += 1
        
        logging.info("Sequence distribution after oversampling:")
        for cls, count in sorted(oversampled_by_class.items()):
            cls_name = "normal" if cls == 0 else list(self.anomaly_mapping.keys())[list(self.anomaly_mapping.values()).index(cls)]
            logging.info(f"  Class {cls} ({cls_name}): {count} sequences")