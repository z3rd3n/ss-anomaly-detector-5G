import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from collections import defaultdict, Counter
import random
from typing import Dict, List, Tuple, Optional, Set

# Constants
EPS = 1e-8  # small constant

def custom_collate_fn(batch):
    """
    Merges a list of samples into a batch.
    """
    try:
        features = torch.stack([item['features'] for item in batch])
        timestamps = [item['timestamps'] for item in batch]
        labels = torch.stack([item['labels'] for item in batch]) if 'labels' in batch[0] else None
        harq_ids = torch.stack([item['harq_ids'] for item in batch]) if 'harq_ids' in batch[0] else None
        
        out = {'features': features, 'timestamps': timestamps}
        if labels is not None:
            out['labels'] = labels
        if harq_ids is not None:
            out['harq_ids'] = harq_ids
            
        return out
    except Exception as e:
        logging.error(f"Error in collate_fn: {e}")
        for i, item in enumerate(batch):
            logging.error(f"Item {i} keys: {item.keys()}")
        return {}


class SequenceStateCacheDataset(Dataset):
    """
    Dataset that caches state information between sequences for each HARQ ID
    """
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
                 use_state_cache: bool = True,
                 overlap_ratio: float = 0.5,
                 transform=None):
        """
        Loads sequences from a parquet file, preserving state across sequences.
        
        Args:
            parquet_path: Path to the parquet file
            feature_columns: List of column names to use as features
            seq_len: Length of each sequence
            stride: Stride between sequences (default: seq_len * (1-overlap_ratio))
            ratio: Portion of data to use (1.0 = all data)
            seed: Random seed for reproducibility
            skip_anomalies: If True, skip sequences containing anomalies
            normalization_stats: Dictionary with 'means' and 'variances' for normalization
            pre_normalize: If True, normalize features at load time
            use_state_cache: If True, track state between sequences for each HARQ ID
            overlap_ratio: Amount of overlap between sequences (0.0 to 1.0)
            transform: Optional transform to apply to sequences
        """
        super().__init__()
        self.parquet_path = parquet_path
        self.feature_columns = feature_columns
        self.seq_len = seq_len
        self.overlap_ratio = overlap_ratio
        self.stride = stride if stride is not None else max(1, int(seq_len * (1 - overlap_ratio)))
        self.skip_anomalies = skip_anomalies
        self.normalization_stats = normalization_stats
        self.pre_normalize = pre_normalize
        self.use_state_cache = use_state_cache
        self.transform = transform

        # Define mapping for anomaly labels
        self.anomaly_mapping = {
            "unnecessary_retx": 1,
            "missing_retx": 2,
            "new_data_no_retx": 3,
            "max_retx_achieved": 4,
        }

        # Load data and prepare sequences
        self._load_data(ratio, seed)
        self._prepare_sequences()
        self._report_class_counts()
        
        # Create state cache for HARQ IDs
        self.harq_state_cache = {} if use_state_cache else None
        
        # Map sequences to their HARQ IDs for efficient retrieval
        self._map_sequences_to_harq()

    def _load_data(self, ratio, seed):
        """Load data from the parquet file with robust HARQ handling"""
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
        
        # Extract and potentially normalize features
        features = self.df[self.feature_columns].values
        
        # Handle NaN values in features
        features = np.nan_to_num(features, nan=0.0)
        
        # Normalize features if requested
        if self.normalization_stats and self.pre_normalize:
            means = self.normalization_stats.get('means')
            variances = self.normalization_stats.get('variances')
            std = np.sqrt(np.array(variances) + EPS)
            features = (features - np.array(means)) / std
        
        # Convert to torch tensor
        self.features = torch.tensor(features, dtype=torch.float32)
        
        # Process labels with validation
        labels = []
        for insight in self.df['insight']:
            label = self.insight_to_label(insight)
            # Ensure label is in valid range [0, 4] for this problem
            if label < 0 or label > 4:
                label = 0  # Default to normal if invalid
            labels.append(label)
        
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        # Store timestamps and HARQ IDs
        self.timestamps = self.df['timestamp_str'].tolist()
        self.harq_ids = torch.tensor(self.df['harq_id'].values, dtype=torch.long)
        
        logging.info(f"Loaded {len(self.df)} rows from the dataset")

    def _prepare_sequences(self):
        """Pre-compute all valid sequence indices with overlap between sequences"""
        self.sequences = []
        
        # Group by file_id to ensure sequences come from the same file
        for _, group_indices in self.df.groupby('file_id').groups.items():
            indices = list(group_indices)
            num_rows = len(indices)
            
            if num_rows < self.seq_len:
                continue
            
            # Generate valid sequence indices with proper overlap
            for start_idx in range(0, num_rows - self.seq_len + 1, self.stride):
                seq_indices = indices[start_idx:start_idx + self.seq_len]
                
                # Skip sequences with anomalies if requested
                if self.skip_anomalies:
                    seq_labels = self.labels[seq_indices]
                    if (seq_labels != 0).any():
                        continue
                
                # Store sequence information
                self.sequences.append({
                    'indices': seq_indices,
                    'start_idx': start_idx,
                    'file_id': self.df.iloc[seq_indices[0]]['file_id']
                })
                
        logging.info(f"Generated {len(self.sequences)} sequences")

    def _map_sequences_to_harq(self):
        """Create mappings for efficient sequence retrieval by HARQ ID"""
        self.harq_to_sequences = defaultdict(list)
        self.seq_to_harq_ids = {}
        
        for seq_idx, seq_info in enumerate(self.sequences):
            indices = seq_info['indices']
            harq_ids = self.harq_ids[indices].unique().tolist()
            
            self.seq_to_harq_ids[seq_idx] = harq_ids
            for harq_id in harq_ids:
                self.harq_to_sequences[harq_id].append(seq_idx)
                
        logging.info(f"Mapped {len(self.harq_to_sequences)} unique HARQ IDs to sequences")

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
            
        # Store sequence-level class counts
        self.seq_class_counts = self._compute_sequence_level_class_counts()

    def _compute_sequence_level_class_counts(self):
        """Compute the distribution of classes at the sequence level"""
        seq_counts = defaultdict(int)
        
        for seq_info in self.sequences:
            seq_indices = seq_info['indices']
            seq_labels = self.labels[seq_indices]
            
            # Determine the "class" of the sequence
            # If any anomalies, use the most frequent anomaly
            # Otherwise, it's a normal sequence
            anomaly_labels = seq_labels[seq_labels != 0]
            if len(anomaly_labels) > 0:
                # Get the most frequent anomaly
                anomaly_counts = Counter(anomaly_labels.tolist())
                most_common_anomaly = anomaly_counts.most_common(1)[0][0]
                seq_counts[most_common_anomaly] += 1
            else:
                seq_counts[0] += 1
                
        logging.info("Sequence-level class distribution:")
        for label, count in seq_counts.items():
            label_name = self._get_label_name(label)
            logging.info(f"  {label} ({label_name}): {count}")
            
        return seq_counts
    
    def _get_label_name(self, label):
        """Get label name from label index"""
        reverse_mapping = {v: k for k, v in self.anomaly_mapping.items()}
        reverse_mapping[0] = "normal"
        return reverse_mapping.get(label, "unknown")

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
        """Get a sequence by index"""
        seq_info = self.sequences[index]
        seq_indices = seq_info['indices']
        
        # Get features, labels, and timestamps for this sequence
        seq_features = self.features[seq_indices]
        seq_labels = self.labels[seq_indices]
        seq_timestamps = [self.timestamps[i] for i in seq_indices]
        seq_harq_ids = self.harq_ids[seq_indices]
        
        # Normalize if needed and not already done
        if self.normalization_stats and not self.pre_normalize:
            seq_features = self._normalize(seq_features)
            
        # Apply transform if provided
        if self.transform is not None:
            seq_features = self.transform(seq_features)
        
        return {
            'features': seq_features,
            'timestamps': seq_timestamps,
            'labels': seq_labels,
            'harq_ids': seq_harq_ids,
            'index': index
        }

    def __len__(self):
        """Return the number of sequences"""
        return len(self.sequences)


class BalancedSequenceSampler(Sampler):
    """
    Sampler that balances classes at the sequence level
    """
    def __init__(self, dataset, oversample_ratio=1.0, undersample_ratio=0.5, seed=42):
        self.dataset = dataset
        self.oversample_ratio = oversample_ratio
        self.undersample_ratio = undersample_ratio
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Get sequence-level class counts
        self.seq_class_counts = dataset.seq_class_counts
        
        # Create indices by class
        self.indices_by_class = self._create_indices_by_class()
        
        # Determine sampling strategy
        self._compute_sampling_strategy()
    
    def _create_indices_by_class(self):
        """Group sequence indices by their majority class"""
        indices_by_class = defaultdict(list)
        
        for i, seq_info in enumerate(self.dataset.sequences):
            indices = seq_info['indices']
            labels = self.dataset.labels[indices]
            
            # Determine sequence class by majority anomaly
            anomaly_labels = labels[labels != 0]
            if len(anomaly_labels) > 0:
                anomaly_counts = Counter(anomaly_labels.tolist())
                most_common = anomaly_counts.most_common(1)[0][0]
                indices_by_class[most_common].append(i)
            else:
                indices_by_class[0].append(i)
                
        return indices_by_class
    
    def _compute_sampling_strategy(self):
        """Compute the number of samples per class based on balancing strategy"""
        # Get the count of the majority class (usually normal)
        max_count = max(self.seq_class_counts.values())
        normal_count = self.seq_class_counts.get(0, 0)
        
        # Compute target count for each class
        self.samples_per_class = {}
        
        # Undersample normal class
        self.samples_per_class[0] = max(1, int(normal_count * self.undersample_ratio))
        
        # Oversample anomaly classes
        for class_id, count in self.seq_class_counts.items():
            if class_id != 0:  # Not normal class
                self.samples_per_class[class_id] = max(count, int(max_count * self.oversample_ratio))
                
        # Log sampling strategy
        logging.info("Balanced sampling strategy:")
        for class_id, sample_count in self.samples_per_class.items():
            original_count = self.seq_class_counts.get(class_id, 0)
            class_name = self.dataset._get_label_name(class_id)
            logging.info(f"  Class {class_id} ({class_name}): {original_count} -> {sample_count}")
            
    def __iter__(self):
        """Return an iterator over indices, with balanced class representation"""
        indices = []
        
        # For each class, sample with replacement up to the target count
        for class_id, target_count in self.samples_per_class.items():
            class_indices = self.indices_by_class.get(class_id, [])
            
            if not class_indices:
                continue
                
            # Sample with replacement if we need more than available
            if target_count <= len(class_indices):
                sampled_indices = self.rng.sample(class_indices, target_count)
            else:
                # Oversample with replacement
                sampled_indices = class_indices.copy()
                additional_needed = target_count - len(class_indices)
                sampled_indices.extend(self.rng.choices(class_indices, k=additional_needed))
                
            indices.extend(sampled_indices)
            
        # Shuffle all indices
        self.rng.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        """Return the total number of samples across all classes"""
        return sum(self.samples_per_class.values())


class NeighborPreservingSampler(Sampler):
    """
    Sampler that preserves neighborhood relationships between sequences
    """
    def __init__(self, dataset, window_size=3, seed=42):
        self.dataset = dataset
        self.window_size = window_size
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Build sequence neighborhood based on HARQ IDs
        self.sequence_neighborhoods = self._build_sequence_neighborhoods()
        
        # Create sampling windows
        self.sampling_windows = self._create_sampling_windows()
    
    def _build_sequence_neighborhoods(self):
        """Build neighborhood relationship between sequences based on shared HARQ IDs"""
        neighborhoods = defaultdict(set)
        
        # For each HARQ ID, find sequences that contain it
        for harq_id, seq_indices in self.dataset.harq_to_sequences.items():
            # Connect all sequences with this HARQ ID
            for seq_idx in seq_indices:
                neighborhoods[seq_idx].update(seq_indices)
                
        # Remove self from neighborhood
        for seq_idx in neighborhoods:
            neighborhoods[seq_idx].discard(seq_idx)
            
        return neighborhoods
    
    def _create_sampling_windows(self):
        """Create sampling windows based on sequence neighborhoods"""
        windows = []
        visited = set()
        
        # Start from each unvisited sequence
        for seq_idx in range(len(self.dataset)):
            if seq_idx in visited:
                continue
                
            # Create a new window starting from this sequence
            window = [seq_idx]
            visited.add(seq_idx)
            
            # Add neighbors until window is filled or no more neighbors
            neighbors = list(self.sequence_neighborhoods.get(seq_idx, set()) - visited)
            self.rng.shuffle(neighbors)
            
            for neighbor in neighbors[:self.window_size - 1]:
                window.append(neighbor)
                visited.add(neighbor)
                
            windows.append(window)
            
        return windows
    
    def __iter__(self):
        """Return an iterator over indices, preserving neighborhood relationships"""
        # Shuffle the order of windows
        shuffled_windows = self.sampling_windows.copy()
        self.rng.shuffle(shuffled_windows)
        
        # Flatten windows into a single list of indices
        indices = []
        for window in shuffled_windows:
            indices.extend(window)
            
        return iter(indices)
    
    def __len__(self):
        """Return the total number of samples"""
        return len(self.dataset)


class TemporalSequenceSegmenter:
    """
    Divides a continuous stream of data into overlapping sequences
    while preserving HARQ state information
    """
    def __init__(self, seq_len, stride, overlap_ratio=0.5):
        self.seq_len = seq_len
        self.stride = stride if stride is not None else max(1, int(seq_len * (1 - overlap_ratio)))
        self.overlap_ratio = overlap_ratio
        self.harq_states = {}
        
    def segment(self, data_df, feature_columns, timestamp_column='timestamp_str', harq_column='HARQ'):
        """
        Segment a dataframe into sequences with preserved HARQ states
        
        Args:
            data_df: DataFrame with the data
            feature_columns: Columns to use as features
            timestamp_column: Column with timestamp information
            harq_column: Column with HARQ ID information
            
        Returns:
            List of sequences with state information
        """
        sequences = []
        
        # Extract features, timestamps, and HARQ IDs
        features = torch.tensor(data_df[feature_columns].values, dtype=torch.float32)
        timestamps = data_df[timestamp_column].tolist()
        harq_ids = data_df[harq_column].values
        
        # Group by HARQ ID for state tracking
        harq_groups = {}
        for i, harq_id in enumerate(harq_ids):
            if harq_id not in harq_groups:
                harq_groups[harq_id] = []
            harq_groups[harq_id].append(i)
            
        # Generate sequences with stride
        num_rows = len(data_df)
        for start_idx in range(0, num_rows - self.seq_len + 1, self.stride):
            end_idx = start_idx + self.seq_len
            
            # Get indices for this sequence
            indices = list(range(start_idx, end_idx))
            
            # Extract sequence data
            seq_features = features[indices]
            seq_timestamps = [timestamps[i] for i in indices]
            seq_harq_ids = harq_ids[indices]
            
            # Track HARQ IDs in this sequence
            seq_harq_set = set(seq_harq_ids)
            
            # Get initial states for HARQ IDs in this sequence
            initial_states = {}
            for harq_id in seq_harq_set:
                if harq_id in self.harq_states:
                    initial_states[harq_id] = self.harq_states[harq_id]
            
            # Add sequence to results
            sequences.append({
                'features': seq_features,
                'timestamps': seq_timestamps,
                'harq_ids': seq_harq_ids,
                'initial_states': initial_states,
                'indices': indices
            })
            
        return sequences
    
    def update_state(self, harq_id, state):
        """
        Update the state for a given HARQ ID
        """
        self.harq_states[harq_id] = state