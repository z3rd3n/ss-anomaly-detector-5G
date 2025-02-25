import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import defaultdict, Counter
import random
from typing import Dict, List, Tuple, Optional, Set

# Constants
EPS = 1e-8  # small constant

def binary_collate_fn(batch):
    """
    Merges a list of samples into a batch for binary classification.
    """
    try:
        # Extract numerical and categorical features
        numerical_features = torch.stack([item['numerical_features'] for item in batch])
        categorical_features = {
            key: torch.stack([item['categorical_features'][key] for item in batch])
            for key in batch[0]['categorical_features'].keys()
        }
        
        # Extract other information
        binary_labels = torch.stack([item['binary_labels'] for item in batch])
        original_labels = torch.stack([item['original_labels'] for item in batch]) if 'original_labels' in batch[0] else None
        harq_ids = torch.stack([item['harq_ids'] for item in batch]) if 'harq_ids' in batch[0] else None
        timestamps = [item['timestamps'] for item in batch]
        
        out = {
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
            'binary_labels': binary_labels,
            'timestamps': timestamps
        }
        
        if original_labels is not None:
            out['original_labels'] = original_labels
        if harq_ids is not None:
            out['harq_ids'] = harq_ids
            
        return out
    except Exception as e:
        logging.error(f"Error in binary_collate_fn: {e}")
        for i, item in enumerate(batch):
            logging.error(f"Item {i} keys: {item.keys()}")
        return {}


class BinaryAnomalyDataset(Dataset):
    """
    Optimized dataset for binary anomaly detection with hybrid feature modeling
    """
    def __init__(self,
                 parquet_path: str,
                 feature_columns: list,
                 seq_len: int,
                 stride: int = None,
                 ratio: float = 1.0,
                 seed: int = 42,
                 normalization_stats: dict = None,
                 pre_normalize: bool = True,
                 use_state_cache: bool = True,
                 overlap_ratio: float = 0.5,
                 keep_original_labels: bool = True,
                 max_samples: int = None,
                 transform=None):
        """
        Dataset for binary anomaly detection with hybrid feature modeling.
        
        Args:
            parquet_path: Path to the parquet file
            feature_columns: List of column names to use as features
            seq_len: Length of each sequence
            stride: Stride between sequences (default: seq_len * (1-overlap_ratio))
            ratio: Portion of data to use (1.0 = all data)
            seed: Random seed for reproducibility
            normalization_stats: Dictionary with 'means' and 'stds' for numerical features
            pre_normalize: If True, normalize numerical features at load time
            use_state_cache: If True, track state between sequences for each HARQ ID
            overlap_ratio: Amount of overlap between sequences (0.0 to 1.0)
            keep_original_labels: Keep original multi-class labels for reporting
            max_samples: Maximum number of samples to load (for testing)
            transform: Optional transform to apply to sequences
        """
        super().__init__()
        self.parquet_path = parquet_path
        self.feature_columns = feature_columns
        self.seq_len = seq_len
        self.overlap_ratio = overlap_ratio
        self.stride = stride if stride is not None else max(1, int(seq_len * (1 - overlap_ratio)))
        self.normalization_stats = normalization_stats
        self.pre_normalize = pre_normalize
        self.use_state_cache = use_state_cache
        self.keep_original_labels = keep_original_labels
        self.max_samples = max_samples
        self.transform = transform
        
        # Define categorical and numerical features
        self.categorical_features = ['HARQ', 'CRC', 'NDI']
        self.numerical_features = ['SFN', 'Slot', 'MCS', 'ReTx']
        
        # Define mapping for anomaly labels (keep for reporting)
        self.anomaly_mapping = {
            "unnecessary_retx": 1,
            "missing_retx": 2,
            "new_data_no_retx": 3,
            "max_retx_achieved": 4,
        }
        
        # Inverse mapping for reporting
        self.inverse_anomaly_mapping = {v: k for k, v in self.anomaly_mapping.items()}
        self.inverse_anomaly_mapping[0] = "normal"

        # Load data and prepare sequences
        self._load_data(ratio, seed)
        self._prepare_sequences()
        self._report_class_counts()
        
        # Create state cache for HARQ IDs
        self.harq_state_cache = {} if use_state_cache else None
        
        # Map sequences to their HARQ IDs for efficient retrieval
        self._map_sequences_to_harq()

    def _load_data(self, ratio, seed):
        """Load data from the parquet file with hybrid feature processing and optimized loading"""
        logging.info(f"Loading data from {self.parquet_path}")
        
        try:
            # Only load required columns to save memory
            columns_to_load = self.feature_columns + ['file_id', 'insight', 'timestamp_str']
            
            # Optimize loading by sampling at the file level if ratio < 1.0
            if ratio < 1.0:
                # First load only file_id to get unique file IDs
                df_ids = pd.read_parquet(self.parquet_path, columns=['file_id'])
                unique_file_ids = df_ids['file_id'].unique()
                
                np.random.seed(seed)
                num_files = max(1, int(len(unique_file_ids) * ratio))
                sampled_files = np.random.choice(unique_file_ids, num_files, replace=False)
                
                # Then load only sampled files and required columns
                self.df = pd.read_parquet(
                    self.parquet_path,
                    columns=columns_to_load,
                    filters=[('file_id', 'in', sampled_files.tolist())]
                )
            else:
                # Load only required columns
                self.df = pd.read_parquet(self.parquet_path, columns=columns_to_load)
            
            # Apply max_samples limit if specified
            if self.max_samples is not None and len(self.df) > self.max_samples:
                self.df = self.df.sample(self.max_samples, random_state=seed)
                
            # Ensure required columns exist
            if 'insight' not in self.df.columns:
                self.df['insight'] = ""
            
            if 'timestamp_str' not in self.df.columns:
                self.df['timestamp_str'] = ["" for _ in range(len(self.df))]
                
            # Extract HARQ ID for state tracking with safety checks
            harq_values = []
            if 'HARQ' in self.df.columns:
                harq_values = self.df['HARQ'].values
            elif 'HARQ_ID' in self.df.columns:
                harq_values = self.df['HARQ_ID'].values
            else:
                # If HARQ ID not available, use row index
                harq_values = np.arange(len(self.df))
            
            # Ensure HARQ values are valid integers within a reasonable range
            harq_values = np.clip(harq_values, 0, 15)  # Assuming max 16 HARQ IDs (0-15)
            self.df['harq_id'] = harq_values
            
            # Process original multi-class labels
            original_labels = []
            for insight in self.df['insight']:
                label = self.insight_to_label(insight)
                original_labels.append(label)
            
            self.original_labels = torch.tensor(original_labels, dtype=torch.long)
            
            # Convert to binary labels (0: normal, 1: anomaly)
            self.binary_labels = (self.original_labels != 0).long()
            
            # Process numerical features
            numerical_data = self.df[self.numerical_features].values
            numerical_data = np.nan_to_num(numerical_data, nan=0.0)
            
            # Normalize numerical features if requested
            if self.normalization_stats and self.pre_normalize:
                means = self.normalization_stats.get('means')
                stds = self.normalization_stats.get('stds')
                
                # Validate stats dimensions
                if means is not None and stds is not None:
                    # Make sure stats match feature dimensions
                    if len(means) != numerical_data.shape[1]:
                        logging.warning(f"Feature stats mismatch: got {len(means)} means but have {numerical_data.shape[1]} numerical features")
                        means = means[:numerical_data.shape[1]] if len(means) > numerical_data.shape[1] else means + [0.0] * (numerical_data.shape[1] - len(means))
                    
                    if len(stds) != numerical_data.shape[1]:
                        logging.warning(f"Feature stats mismatch: got {len(stds)} stds but have {numerical_data.shape[1]} numerical features")
                        stds = stds[:numerical_data.shape[1]] if len(stds) > numerical_data.shape[1] else stds + [1.0] * (numerical_data.shape[1] - len(stds))
                    
                    # Apply normalization
                    numerical_data = (numerical_data - np.array(means)) / (np.array(stds) + EPS)
                else:
                    logging.warning("Missing normalization stats, skipping normalization")
            
            self.numerical_features_tensor = torch.tensor(numerical_data, dtype=torch.float32)
            
            # Process categorical features (no normalization)
            self.categorical_features_tensors = {}
            for feature in self.categorical_features:
                if feature in self.df.columns:
                    feature_values = self.df[feature].values
                    feature_values = np.clip(feature_values, 0, self._get_max_value(feature))
                    self.categorical_features_tensors[feature] = torch.tensor(feature_values, dtype=torch.long)
                else:
                    logging.warning(f"Categorical feature {feature} not found in dataframe")
                    # Create a dummy tensor with zeros
                    self.categorical_features_tensors[feature] = torch.zeros(len(self.df), dtype=torch.long)
                    
            # Store timestamps and HARQ IDs
            self.timestamps = self.df['timestamp_str'].tolist()
            self.harq_ids = torch.tensor(self.df['harq_id'].values, dtype=torch.long)
            
            logging.info(f"Loaded {len(self.df)} rows from the dataset")
        
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            # Create empty datasets to prevent further errors
            self.df = pd.DataFrame(columns=self.feature_columns + ['file_id', 'insight', 'timestamp_str', 'harq_id'])
            self.numerical_features_tensor = torch.zeros((0, len(self.numerical_features)), dtype=torch.float32)
            self.categorical_features_tensors = {feature: torch.zeros(0, dtype=torch.long) for feature in self.categorical_features}
            self.original_labels = torch.zeros(0, dtype=torch.long)
            self.binary_labels = torch.zeros(0, dtype=torch.long)
            self.timestamps = []
            self.harq_ids = torch.zeros(0, dtype=torch.long)
            logging.warning("Created empty dataset due to loading error")
        
    def _get_max_value(self, feature):
        """Get maximum valid value for categorical feature"""
        if feature == 'HARQ':
            return 15  # 16 possible HARQ IDs (0-15)
        elif feature == 'CRC':
            return 1   # Binary (0-1)
        elif feature == 'NDI':
            return 1   # Binary (0-1)
        else:
            return 100  # Default for safety

    def _prepare_sequences(self):
        """Pre-compute all valid sequence indices with overlap between sequences"""
        self.sequences = []
        
        try:
            # Group by file_id to ensure sequences come from the same file
            for _, group_indices in self.df.groupby('file_id').groups.items():
                indices = list(group_indices)
                num_rows = len(indices)
                
                if num_rows < self.seq_len:
                    continue
                
                # Generate valid sequence indices with proper overlap
                for start_idx in range(0, num_rows - self.seq_len + 1, self.stride):
                    seq_indices = indices[start_idx:start_idx + self.seq_len]
                    
                    # Store sequence information
                    self.sequences.append({
                        'indices': seq_indices,
                        'start_idx': start_idx,
                        'file_id': self.df.iloc[seq_indices[0]]['file_id'] if len(seq_indices) > 0 else None
                    })
                    
            logging.info(f"Generated {len(self.sequences)} sequences")
        
        except Exception as e:
            logging.error(f"Error preparing sequences: {e}")
            self.sequences = []
            logging.warning("Created empty sequences list due to error")

    def _map_sequences_to_harq(self):
        """Create mappings for efficient sequence retrieval by HARQ ID"""
        self.harq_to_sequences = defaultdict(list)
        self.seq_to_harq_ids = {}
        
        try:
            for seq_idx, seq_info in enumerate(self.sequences):
                indices = seq_info['indices']
                if len(indices) == 0 or seq_idx >= len(self.harq_ids):
                    continue
                    
                harq_ids = self.harq_ids[indices].unique().tolist()
                
                self.seq_to_harq_ids[seq_idx] = harq_ids
                for harq_id in harq_ids:
                    self.harq_to_sequences[harq_id].append(seq_idx)
                    
            logging.info(f"Mapped {len(self.harq_to_sequences)} unique HARQ IDs to sequences")
        
        except Exception as e:
            logging.error(f"Error mapping sequences to HARQ IDs: {e}")
            self.harq_to_sequences = defaultdict(list)
            self.seq_to_harq_ids = {}
            logging.warning("Created empty HARQ mappings due to error")

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
            
        # Return the smallest valid label
        return min(valid_labels)
    
    def _report_class_counts(self):
        """Report class distribution in the dataset"""
        try:
            # Binary label distribution
            binary_unique, binary_counts = torch.unique(self.binary_labels, return_counts=True)
            
            logging.info("Binary class distribution in the dataset:")
            self.binary_counts = {label.item(): count.item() 
                                for label, count in zip(binary_unique, binary_counts)}
            
            for label, count in self.binary_counts.items():
                label_name = "normal" if label == 0 else "anomaly"
                logging.info(f"  {label} ({label_name}): {count}")
            
            # Original multi-class distribution (for reporting)
            original_unique, original_counts = torch.unique(self.original_labels, return_counts=True)
            
            logging.info("Original class distribution in the dataset:")
            self.original_counts = {label.item(): count.item() 
                                   for label, count in zip(original_unique, original_counts)}
            
            for label, count in self.original_counts.items():
                label_name = self.inverse_anomaly_mapping.get(label, "unknown")
                logging.info(f"  {label} ({label_name}): {count}")
                
            # Store sequence-level class counts
            self.seq_normal_count = 0
            self.seq_anomaly_count = 0
            
            for seq_info in self.sequences:
                seq_indices = seq_info['indices']
                if len(seq_indices) == 0:
                    continue
                    
                seq_binary_labels = self.binary_labels[seq_indices]
                
                # A sequence is considered anomalous if it contains any anomaly
                if torch.any(seq_binary_labels == 1):
                    self.seq_anomaly_count += 1
                else:
                    self.seq_normal_count += 1
                    
            logging.info(f"Sequence-level binary distribution: Normal: {self.seq_normal_count}, Anomaly: {self.seq_anomaly_count}")
        
        except Exception as e:
            logging.error(f"Error reporting class counts: {e}")
            self.binary_counts = {0: 0, 1: 0}
            self.original_counts = {0: 0}
            self.seq_normal_count = 0
            self.seq_anomaly_count = 0
            logging.warning("Created default class counts due to error")

    def _normalize_numerical(self, features):
        """Normalize numerical features on-the-fly if not pre-normalized"""
        if not self.normalization_stats or self.pre_normalize:
            return features

        try:
            means = self.normalization_stats.get('means')
            stds = self.normalization_stats.get('stds')
            
            if means is None or stds is None:
                return features
                
            # Make sure stats match feature dimensions
            if len(means) != features.shape[-1]:
                means = means[:features.shape[-1]] if len(means) > features.shape[-1] else means + [0.0] * (features.shape[-1] - len(means))
            
            if len(stds) != features.shape[-1]:
                stds = stds[:features.shape[-1]] if len(stds) > features.shape[-1] else stds + [1.0] * (features.shape[-1] - len(stds))
            
            if not torch.is_tensor(means):
                means = torch.tensor(means, dtype=torch.float32, device=features.device)
            if not torch.is_tensor(stds):
                stds = torch.tensor(stds, dtype=torch.float32, device=features.device)

            return (features.float() - means) / (stds + EPS)
            
        except Exception as e:
            logging.error(f"Error in normalization: {e}")
            return features

    def get_harq_state(self, harq_id):
        """Get cached state for a given HARQ ID"""
        if not self.use_state_cache or harq_id not in self.harq_state_cache:
            return None
        return self.harq_state_cache[harq_id]
    
    def update_harq_state(self, harq_id, state):
        """Update cached state for a given HARQ ID"""
        if self.use_state_cache:
            self.harq_state_cache[harq_id] = state

    def __getitem__(self, index):
        """Get a sequence by index with hybrid feature processing"""
        try:
            if index >= len(self.sequences):
                raise IndexError(f"Index {index} out of bounds for dataset with {len(self.sequences)} sequences")
                
            seq_info = self.sequences[index]
            seq_indices = seq_info['indices']
            
            if len(seq_indices) == 0:
                raise ValueError(f"Empty sequence at index {index}")
            
            # Get numerical features
            seq_numerical = self.numerical_features_tensor[seq_indices]
            
            # Normalize if needed and not already done
            if self.normalization_stats and not self.pre_normalize:
                seq_numerical = self._normalize_numerical(seq_numerical)
            
            # Get categorical features
            seq_categorical = {
                feature: self.categorical_features_tensors[feature][seq_indices]
                for feature in self.categorical_features
            }
            
            # Get binary labels
            seq_binary_labels = self.binary_labels[seq_indices]
            
            # Get original labels for reporting if needed
            seq_original_labels = self.original_labels[seq_indices] if self.keep_original_labels else None
            
            # Get timestamps and HARQ IDs
            seq_timestamps = [self.timestamps[i] for i in seq_indices]
            seq_harq_ids = self.harq_ids[seq_indices]
            
            # Apply transform if provided
            if self.transform is not None:
                seq_numerical = self.transform(seq_numerical)
            
            result = {
                'numerical_features': seq_numerical,
                'categorical_features': seq_categorical,
                'binary_labels': seq_binary_labels,
                'timestamps': seq_timestamps,
                'harq_ids': seq_harq_ids,
                'index': index
            }
            
            if seq_original_labels is not None:
                result['original_labels'] = seq_original_labels
                
            return result
            
        except Exception as e:
            logging.error(f"Error getting item at index {index}: {e}")
            # Return a default item to prevent crashes
            default_numerical = torch.zeros((self.seq_len, len(self.numerical_features)), dtype=torch.float32)
            default_categorical = {
                feature: torch.zeros(self.seq_len, dtype=torch.long)
                for feature in self.categorical_features
            }
            default_binary = torch.zeros(self.seq_len, dtype=torch.long)
            default_original = torch.zeros(self.seq_len, dtype=torch.long) if self.keep_original_labels else None
            default_timestamps = ["" for _ in range(self.seq_len)]
            default_harq_ids = torch.zeros(self.seq_len, dtype=torch.long)
            
            result = {
                'numerical_features': default_numerical,
                'categorical_features': default_categorical,
                'binary_labels': default_binary,
                'timestamps': default_timestamps,
                'harq_ids': default_harq_ids,
                'index': index
            }
            
            if default_original is not None:
                result['original_labels'] = default_original
                
            return result

    def __len__(self):
        """Return the number of sequences"""
        return max(len(self.sequences), 1)  # Ensure at least 1 to prevent runtime errors


def create_balanced_sampler(dataset):
    """
    Create a weighted sampler for balanced training
    """
    # Calculate weights for each sequence based on whether it contains anomalies
    weights = []
    
    if dataset.seq_normal_count == 0 and dataset.seq_anomaly_count == 0:
        # Empty dataset or error case - use uniform weights
        return None
    
    for seq_idx, seq_info in enumerate(dataset.sequences):
        indices = seq_info['indices']
        if len(indices) == 0:
            weights.append(1.0)
            continue
            
        seq_binary_labels = dataset.binary_labels[indices]
        
        # Check if sequence contains any anomaly
        if torch.any(seq_binary_labels == 1):
            # Anomaly sequence
            weight = 1.0 / max(1, dataset.seq_anomaly_count)
        else:
            # Normal sequence
            weight = 1.0 / max(1, dataset.seq_normal_count)
            
        weights.append(weight)
    
    # Create weighted sampler
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )


def create_binary_dataloader(dataset, batch_size, shuffle=True, balance=True, num_workers=4):
    """
    Create dataloader with optional balanced sampling
    """
    # Safety checks
    if len(dataset) == 0:
        logging.warning("Empty dataset provided to dataloader, using default batch size 1")
        batch_size = 1
        
    if balance and shuffle and dataset.seq_normal_count > 0 and dataset.seq_anomaly_count > 0:
        sampler = create_balanced_sampler(dataset)
        shuffle = False  # Can't use both shuffle and sampler
    else:
        sampler = None
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=binary_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )