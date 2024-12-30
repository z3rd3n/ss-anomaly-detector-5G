'''
Look at my code base, and understand its structure, then tell me how can I do Hyperparameter search on the parameters on the configClass.py, it should select some small subset of the dataset, and should look at the validation loss because I don't have any labels, implement the best hyperparameter search possible.
'''



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
            assert not df.isna().any().any(), "Found NaNs!"
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
# # subAdjacent/model/dataEmbedding.py
import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        layers = []
        in_channels = c_in
        while in_channels * 4 < d_model:
            out_channels = in_channels * 4
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=padding,
                    padding_mode='circular',
                    bias=False
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        # Final layer to reach d_model
        layers.append(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=d_model,
                kernel_size=3,
                padding=padding,
                padding_mode='circular',
                bias=False
            )
        )
        layers.append(nn.BatchNorm1d(d_model))
        layers.append(nn.ReLU())
        self.tokenConv = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
# # subAdjacent/model/attnetionsLayer.py
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAnomalyAttention(nn.Module):    
    def __init__(
        self,
        dropout: float = 0.0,
        output_attention: bool = False,
    ):
        super().__init__()
        self.output_attention = output_attention
        self.dropout = nn.Dropout(dropout)
        
        self.softmax = nn.Softmax(dim=-1)
        self.delta1 = nn.Parameter(torch.tensor(1.0))

    def _apply_mapping(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        queries[queries < 0] = -10
        keys[keys < 0] = -10
        delta = nn.Softplus()(self.delta1)
        queries = self.softmax(queries / delta)
        keys = self.softmax(keys / delta)

        return queries, keys

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of the linear attention mechanism."""
        
        queries, keys = self._apply_mapping(queries, keys)
        
        sum_q = queries.isnan().any() or queries.isinf().any()
        sum_k = keys.isnan().any() or keys.isinf().any()
        if sum_q or sum_k:
            print("NaN/Inf found in queries/keys!")
            print("queries min/max:", queries.min(), queries.max())
            print("keys min/max:", keys.min(), keys.max())
        
        kv = torch.einsum("b e h l, b l h f -> b h e f", keys.transpose(1, 3), values)

        z = 1 / (torch.einsum("b l h e, b h e -> b l h", queries, keys.sum(dim=1)) + 1e-6)
        output = torch.einsum("b l h e, b h e e, b l h -> b l h e", queries, kv, z)
        
        if self.output_attention:
            return output.contiguous(), (queries, keys)
        return output.contiguous(), None


class AttentionLayer(nn.Module): 
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None
    ):
        super().__init__()
        
        self.d_keys = d_keys or (d_model // n_heads)
        self.d_values = d_values or (d_model // n_heads)
        self.n_heads = n_heads
        
        # Layer components
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        
        # Projection layers
        self.query_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, self.d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(self.d_values * n_heads, d_model) # d_values * n_heads = d_model

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size, seq_len, _ = queries.shape
        _, source_len, _ = keys.shape
        
        # Project inputs to multi-head representations
        queries = self.query_projection(queries).view(batch_size, seq_len, self.n_heads, -1)
        keys = self.key_projection(keys).view(batch_size, source_len, self.n_heads, -1)
        values = self.value_projection(values).view(batch_size, source_len, self.n_heads, -1)
        
        # Apply attention mechanism
        output, (query_weights, key_weights) = self.inner_attention(
            queries,
            keys,
            values
        )
        
        # Reshape and project output
        output = output.view(batch_size, seq_len, -1)
        output = self.out_projection(output)
        
        return output, query_weights, key_weights
# subAdjacent/model/anomalyTransformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from subAdjacent.model.attentionsLayer import LinearAnomalyAttention, AttentionLayer
from subAdjacent.model.dataEmbedding import DataEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, d_model, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = 4 * d_model
        self.attention_layer = attention_layer
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x, queries, keys = self.attention_layer(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), queries, keys


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        # x [B, L, D]
        queries_list = []
        keys_list = []
        for attn_layer in self.attn_layers:
            x, queries, keys = attn_layer(x)
            queries_list.append(queries)
            keys_list.append(keys)

        if self.norm is not None:
            x = self.norm(x)

        return x, queries_list, keys_list


class AnomalyTransformer(nn.Module):
    def __init__(self, enc_in, c_out, d_model=512, n_heads=8, e_layers=3,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        attention_layers = [
            EncoderLayer(
            AttentionLayer(
                LinearAnomalyAttention(
                dropout=dropout, 
                output_attention=output_attention, 
                ),
                d_model, 
                n_heads
            ),
            d_model,
            dropout=dropout,
            activation=activation
            ) 
            for _ in range(e_layers)
        ]
        
        self.encoder = Encoder(
            attn_layers=attention_layers,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, queries_list, keys_list = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, queries_list, keys_list
        else:
            return enc_out  # [B, L, D]
        
    def compute_sub_adj_contrib(self, q, k, span, one_side):
        """
        Same as your SACon function, returning shape [B, L].
        Minimally renamed here to 'compute_sub_adj_contrib'.
        """
        L = q.shape[1]
        assert L >= span[1] >= span[0] >= 0

        # compute attention matrix
        attnMatrix = torch.einsum("b l h e, b s h e -> b h l s", q, k)
        den = attnMatrix.sum(dim=-1, keepdim=True)
        if (den <= 1e-12).any():
            print("attnMatrix min/max:", attnMatrix.min().item(), attnMatrix.max().item())
        den = den.clamp(min=1e-8)
        attnMatrix = attnMatrix / den


        lossMat = None
        for k in range(-span[1], span[1] + 1):  # range(-span[1], -span[0]+1)
            # only one-side is used
            if one_side:
                if k < span[0]:
                    continue
            else:
                if abs(k) < span[0]:
                    continue

            diag1 = torch.diagonal(attnMatrix, offset=k, dim1=-2, dim2=-1)
            if k > 0:
                p1d = (k, 0)
            else:
                p1d = (0, abs(k))
            diag1 = F.pad(diag1, p1d)

            if lossMat is None:
                lossMat = diag1
            else:
                lossMat += diag1

            if k > 0:
                offset_k = -(L-k)
            else:
                offset_k = L+k
            diag1 = torch.diagonal(attnMatrix, offset=offset_k, dim1=-2, dim2=-1)  # why use L-k ?  L-k performs better
            if offset_k > 0:
                p1d = (offset_k, 0)
            else:
                p1d = (0, abs(offset_k))
            diag1 = F.pad(diag1, p1d)

            lossMat += diag1

        # b,h,l
        lossMat = torch.mean(lossMat, dim=-2)

        return lossMat  # B,L
# subAdjacent/trainer.py
import logging
import torch
from subAdjacent.run_epoch import train_one_epoch, validate_one_epoch
from utils import *
from tqdm import tqdm
# import softm


def train_model(params, model, optimizer, scheduler, train_loader, val_loader):
    start_epoch = 0
    if params.checkpoint_path:
        start_epoch, prev_loss = load_checkpoint(
            model,
            optimizer,
            params.checkpoint_path,
            params.device
        )
        logging.info(f"Resuming training from epoch {start_epoch + 1} and from loss {prev_loss:.4f}")
        start_epoch += 1

    # 4) training
    best_val_loss = float('inf')
    not_improved_count = 0
    early_stop_patience = 5

    all_train_losses = []
    all_val_losses = []

    criterion_mse = torch.nn.MSELoss(reduction='none')

    for epoch in range(start_epoch, params.num_epochs):
        logging.info(f"===== Epoch {epoch+1}/{params.num_epochs} =====")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            params.device,
            criterion_mse,
            params.span,
            params.one_side,
            lambda_sacon=params.k_value  # or some param name
        )
        val_rec_loss, val_total_loss = validate_one_epoch(
            model,
            val_loader,
            params.device,
            criterion_mse,
            params.span,
            params.one_side,
            lambda_sacon=params.k_value
        )

        all_train_losses.append(train_loss)
        all_val_losses.append(val_total_loss)

        logging.info(
            f"Epoch [{epoch+1}/{params.num_epochs}] | "
            f"TrainLoss: {train_loss:.4f} | ValRec: {val_rec_loss:.4f} | ValTotal: {val_total_loss:.4f}"
        )

        #scheduler.step(val_total_loss)

        # Early stopping
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            not_improved_count = 0
            save_checkpoint(model, optimizer, epoch, val_total_loss, params)
        else:
            not_improved_count += 1
            logging.info(f"No improvement. Early stopping count: {not_improved_count}/{early_stop_patience}")
            if not_improved_count >= early_stop_patience:
                logging.info("Early stopping triggered.")
                break

    plot_training_curves(all_train_losses, all_val_losses, params.output_dir)
    logging.info("Training finished.")


def detect_anomalies(params, model, optimizer, val_loader):
    load_last_checkpoint(params, model, optimizer)
    
    model.eval()
    all_scores = []
    all_features = []
    all_timestamps = []
    criterion_mse = torch.nn.MSELoss(reduction='none')
    softmax = torch.nn.Softmax(dim=-1)

    with torch.no_grad():
        train_energy = []
        for batch in tqdm(val_loader, desc="Detecting anomalies"):
            features = batch['features'].to(params.device)
            timestamps = batch['timestamps']
            
            # Get model outputs
            enc_out, queries_list, keys_list = model(features)
            
            # Per-window reconstruction loss 
            rec_loss = criterion_mse(enc_out, features).mean(dim=-1)
            loss_attn = 0.0
            # Calculate SACon from all layers
            for q, k in zip(queries_list, keys_list):
                loss_attn += model.compute_sub_adj_contrib(q, k, params.span, params.one_side)
            loss_attn /= len(queries_list)
            
            train_score = softmax(-loss_attn) * rec_loss
            train_energy.append(train_score.cpu().numpy())
            all_features.append(features.cpu().numpy())
            all_timestamps.extend([t for sublist in timestamps for t in sublist])

    train_attn_array = np.concatenate(train_energy, axis=0).reshape(-1)
    all_features = np.concatenate(all_features, axis=0).reshape(-1, len(params.feature_columns))
   
    # Calculate threshold using EVT
    threshold = calculate_threshold_evt(train_attn_array)
    anomalies_mask = train_attn_array > threshold
    
    # Save results
    unscale_and_save_anomalies(
        timestamps=all_timestamps,
        features=all_features,
        anomaly_scores=train_attn_array,
        threshold=threshold,
        output_csv=os.path.join(params.output_dir, "detected_anomalies.csv")
    )

    # Visualizations
    plot_anomalies(train_attn_array, anomalies_mask, threshold, params.output_dir)
    plot_attention_matrices(model, val_loader, params.device, params.output_dir)
    
    logging.info("Anomaly detection complete.")
# subAdjacent/run_epoch.py
import torch
from tqdm import tqdm
import logging

def train_one_epoch(model, dataloader, optimizer, device, 
                    criterion_mse, span, one_side, lambda_sacon=0.1):
    """
    Changes:
      - total_loss = rec_loss - lambda_sacon * sacon_loss
      - We skip "2.0 * rec_loss - k * loss_attn" approach.
    """
    model.train()
    total_loss = 0.0
    batch_count = 0

    train_pbar = tqdm(dataloader, desc="Training", total=len(dataloader))
    for batch_idx, batch in enumerate(train_pbar):
        features = batch['features'].to(device)  # [B, seq_len, D]
        with torch.autograd.set_detect_anomaly(True):
            enc_out, queries_list, keys_list = model(features)

            # Reconstruction loss
            rec_loss = criterion_mse(enc_out, features) # shape [B, seq_len, D]

            sacon_all_layers = 0.0
            for (q, k_) in zip(queries_list, keys_list): # e_layers times
                sacon_all_layers += model.compute_sub_adj_contrib(q, k_, span, one_side)
            sacon_all_layers /= len(queries_list)  # shape [B, L]

            sacon_mean = sacon_all_layers.mean() 

            rec_loss_scalar = rec_loss.mean()

            # Final total loss
            loss = 2*rec_loss_scalar - lambda_sacon * sacon_mean

            optimizer.zero_grad()

            loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        # Logging
        if (batch_idx + 1) % 5000 == 0:
            logging.info(
                f"Batch {batch_idx+1} => "
                f"RecLoss: {rec_loss_scalar.item():.4f}, "
                f"SACon: {sacon_mean.item():.4f}, "
                f"TotalLoss: {loss.item():.4f}"
            )

        train_pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'rec': f"{rec_loss_scalar.item():.4f}",
            'SACon': f"{sacon_mean.item():.4f}"
        })

    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    return avg_loss


def validate_one_epoch(model, dataloader, device, criterion_mse, span, one_side, lambda_sacon=10.0):
    """
    Validation with the same objective. We return just the rec_loss or total_loss
    for logging.
    """
    model.eval()
    total_rec_loss = 0.0
    total_loss = 0.0
    batch_count = 0

    val_pbar = tqdm(dataloader, desc="Validation", total=len(dataloader), leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_pbar):
            features = batch['features'].to(device)
            enc_out, queries_list, keys_list = model(features)

            # reconstruction loss
            rec_loss = criterion_mse(enc_out, features).mean()

            # SACon
            sacon_all_layers = 0.0
            for (q, k_) in zip(queries_list, keys_list):
                sacon_all_layers += model.compute_sub_adj_contrib(q, k_, span, one_side)
            sacon_all_layers /= len(queries_list)
            sacon_mean = sacon_all_layers.mean()

            # total loss
            loss_val = 2*rec_loss - lambda_sacon * sacon_mean

            total_rec_loss += rec_loss.item()
            total_loss += loss_val.item()
            batch_count += 1

            val_pbar.set_postfix({
                'val_rec': f"{rec_loss.item():.4f}",
                'val_loss': f"{loss_val.item():.4f}"
            })

    avg_rec_loss = total_rec_loss / batch_count if batch_count > 0 else 0.0
    avg_total_loss = total_loss / batch_count if batch_count > 0 else 0.0
    return avg_rec_loss, avg_total_loss
# subAdjacent/configClass.py
import os

class Config:
    def __init__(self):
        import torch
        import numpy as np
        self.seed=42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.train = False
        self.debug = True
        self.seq_len = 32
        self.stride = 32
        self.batch_size = 64
        self.num_epochs = 50
        self.validation_ratio=0.2
        self.learning_rate = 1e-4
        self.pretrain= None

        self.feature_columns = [
            'SFN', 'Slot', 'CC', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI',
        ]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = None  # Path to load checkpoint from

        # Directories
        self.parquet_path = "data/scaled_pdsch.parquet"
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        self.output_dir = results_dir

        # Model architecture params
        self.model_dim = 128       # d_model
        self.n_heads = 8          # number of attention heads
        self.e_layers = 3         # number of encoder layers
        self.activation = 'gelu'  # activation function
        self.k_value = 0.1        # trade-off parameter for loss
        self.dropout = 0.1
        self.span = [4,8]
        self.one_side = False

        # Training specific params
        self.shuffle_files = True
        self.output_attention = True

        # System params
        self.num_workers = 0
        self.pin_memory = True

    def build_model(self):
        from subAdjacent.model.anomalyTransformer import AnomalyTransformer
        model = AnomalyTransformer(
            enc_in=len(self.feature_columns),
            c_out=len(self.feature_columns),
            d_model=self.model_dim,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout,
            activation=self.activation,
            output_attention=self.output_attention,
        ).to(self.device)
        return model
#!/usr/bin/env python3
# main.py
from utils import start_logging, bring_approach
from data.dataLoader import ParquetSequenceDataset, custom_collate_fn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import logging

def main(params, train_func, detect_func):    
    start_logging(params)

    logging.info("Building model...")
    model = params.build_model()

    logging.info("Creating train and validation datasets...")
    train_dataset, val_dataset = ParquetSequenceDataset.create_train_val_splits(
        parquet_path=params.parquet_path,
        feature_columns=params.feature_columns,
        seq_len=params.seq_len,
        validation_ratio=params.validation_ratio,
        seed=params.seed
    )

    logging.info("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=params.pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=params.pin_memory,
        drop_last=True
    )

    logging.info("Initializing optimizer and scheduler...")
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=True
    )

    if params.train:
        logging.info("Starting training...")
        train_func(
            params,
            model,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
        )
    else:
        logging.info("Starting detection...")
        detect_func(
            params, 
            model, 
            optimizer,
            val_loader, 
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main script with specified approach.")
    parser.add_argument('--approach', type=str, default='subAdjacent', help='Specify the approach to use.')
    args = parser.parse_args()
    main(**bring_approach(args))
# utils.py
import logging
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from scipy import stats
import pandas as pd


def start_logging(params):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(params.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'logTraining_{current_time}.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def save_checkpoint(model, optimizer, epoch, loss, params):
    checkpoint_path = os.path.join(params.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    logging.info(f"Checkpoint saved: {checkpoint_path}")

def load_last_checkpoint(params, model, optimizer):
    result_files = [f for f in os.listdir(params.output_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    if not result_files:
        logging.info(f"No checkpoints found in {params.output_dir}, aborting detection.")
        return

    def get_epoch(fname):
        return int(fname.split('_')[-1].replace('.pt',''))
    result_files_sorted = sorted(result_files, key=lambda x: get_epoch(x))
    last_ckpt = os.path.join(params.output_dir, result_files_sorted[-1])
    logging.info(f"Loading last checkpoint: {last_ckpt}")

    epoch, loss = load_checkpoint(model, optimizer, last_ckpt)
    logging.info(f"Loaded last checkpoint from epoch {epoch+1} with loss {loss:.4f}")


def load_checkpoint(model, optimizer, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    logging.info(f"Loaded checkpoint from epoch {epoch+1} with loss {loss:.4f}")
    return epoch, loss


def plot_training_curves(train_losses, val_losses, output_dir):
    """
    Saves a PNG plot of train vs validation loss over epochs.
    """
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Val Loss')
    ax.set_title('Training & Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.tight_layout()
    save_path = os.path.join(output_dir, 'train_val_loss_curve.png')
    plt.savefig(save_path)
    plt.close(fig)
    logging.info(f"Saved train/val loss curve to {save_path}")



def plot_attention_matrices(
    model, 
    dataloader, 
    device, 
    output_dir, 
    max_plots=5,   # how many batches you want to visualize
    max_heads=8,    # how many heads per batch you want to plot
    sample_idx=16    # which sample in the batch to visualize
):
    """
    Plots multi-head attention from the last encoder layer.
    
    Args:
        model: your model with a forward() returning queries_list, keys_list.
        dataloader: the DataLoader providing (features, etc).
        device: the device ('cpu' or 'cuda') to use.
        output_dir: where to save the generated plots.
        max_plots: number of batches to plot from the dataloader.
        max_heads: how many heads to visualize per batch.
        sample_idx: which sample in the batch we want to plot.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_plots:
            break  # stop if we already plotted enough

        features = batch['features'].to(device)  # shape [B, L, D]
        with torch.no_grad():
            # The model returns something like enc_out, [queries_per_layer], [keys_per_layer]
            _, queries_list, keys_list = model(features)

        queries = torch.stack(queries_list, dim=0).mean(dim=0) # shape [B, L, H, d_k]
        keys = torch.stack(keys_list, dim=0).mean(dim=0) # shape [B, L, H, d_k]
        
        # We'll visualize a single sample in the batch: sample_idx
        # queries_0 shape => [L, H, d_k]
        queries_0 = queries[sample_idx]  # shape [L, H, d_k]
        keys_0 = keys[sample_idx]        # shape [L, H, d_k]

        # Permute so each head is first: [H, L, d_k]
        queries_0 = queries_0.permute(1, 0, 2)  # => [H, L, d_k]
        keys_0 = keys_0.permute(1, 0, 2)        # => [H, L, d_k]

        # Now compute attention for each head: 
        # attention = Q * K^T => shape [H, L, L]
        # Usually we do scale = 1 / sqrt(d_k).
        d_k = queries_0.size(-1)
        scale = 1.0 / (d_k**0.5)
        attn_matrices = torch.bmm(queries_0, keys_0.transpose(1, 2))  # => [H, L, L]
        attn_matrices = attn_matrices * scale
        attn_matrices = torch.softmax(attn_matrices, dim=-1)          # => [H, L, L]

        # Plot up to max_heads heads
        num_heads = min(attn_matrices.size(0), max_heads)
        num_cols = 4
        num_rows = (num_heads + num_cols - 1) // num_cols  # ceiling division
        fig, axes = plt.subplots(
            nrows=num_rows,
            ncols=num_cols,
            figsize=(4*num_cols, 4*num_rows),  # wide enough for each head
            squeeze=False
        )

        for h in range(num_heads):
            attn_head_h = attn_matrices[h]  # shape [L, L]
            row = h // num_cols
            col = h % num_cols
            ax = axes[row, col]
            im = ax.imshow(
            attn_head_h.cpu().numpy(),
            cmap='hot',
            interpolation='nearest',
            aspect='auto'
            )
            ax.set_title(f"Batch {batch_idx}, Sample {sample_idx}, Head {h}")
            ax.set_xlabel("Key positions")
            ax.set_ylabel("Query positions")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide any unused subplots
        for h in range(num_heads, num_rows * num_cols):
            fig.delaxes(axes.flatten()[h])

        fig.tight_layout()
        save_path = os.path.join(output_dir, f"attention_batch{batch_idx}_sample{sample_idx}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logging.info(f"Saved attention matrix to {save_path}")


def plot_anomalies(all_scores, anomalies_mask, threshold, output_dir):
    """
    Plots the anomaly score over sample index, highlighting anomalies above threshold.
    """
    fig, ax = plt.subplots(figsize=(6,4))
    idx = np.arange(len(all_scores))
    ax.plot(idx, all_scores, label='Anomaly Score')
    ax.axhline(threshold, color='r', linestyle='--', label=f'Threshold={threshold:.2f}')
    # highlight anomalies
    ax.scatter(idx[anomalies_mask], all_scores[anomalies_mask], color='red', s=10, label='Detected Anomalies')
    ax.set_title('Anomaly Scores (Validation Set)')
    ax.set_xlabel('Index')
    ax.set_ylabel('Score')
    ax.legend()
    fig.tight_layout()
    save_path = os.path.join(output_dir, 'anomaly_scores.png')
    plt.savefig(save_path)
    plt.close(fig)
    logging.info(f"Saved anomaly score plot to {save_path}")

def unscale_features(features):
    scaler_json_path = "data/scaling_params.json"
    # 1) Load scaling parameters
    if not os.path.exists(scaler_json_path):
        logging.warning("Scaling files not found; will save scaled features as-is.")
        unscaled = features
    else:
        with open(scaler_json_path, 'r') as f:
            scaling_params = json.load(f)
            
        # Create numpy arrays from the parameters    
        mean = np.array(scaling_params['mean_'])
        scale = np.array(scaling_params['scale_'])
        
        # Perform inverse transform manually: X_orig = X_scaled * scale + mean
        unscaled = features * scale + mean

    # 2) Round to integers
    unscaled = np.rint(unscaled).astype(int)
    return unscaled

def unscale_and_save_anomalies(
    timestamps,
    features,
    anomaly_scores,
    threshold,
    output_csv
):
  
    unscaled = unscale_features(features)

    anomaly_indices = np.where(anomaly_scores >= threshold)[0]
    logging.info(f"Total anomaly count: {len(anomaly_indices)} from {len(anomaly_scores)} ratio {len(anomaly_indices)/len(anomaly_scores)} (threshold={threshold})")

    if len(anomaly_indices) == 0:
        logging.info(f"No anomalies found above threshold = {threshold}. No CSV created.")
        return
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    if len(anomaly_indices) > 0:
        anomalies = pd.DataFrame({
            'timestamp_str': [timestamps[i] for i in anomaly_indices],
            'SFN': unscaled[anomaly_indices, 0].astype(int),
            'Slot': unscaled[anomaly_indices, 1].astype(int),
            'CC': unscaled[anomaly_indices, 2].astype(int),
            'HARQ': unscaled[anomaly_indices, 3].astype(int),
            'MCS': unscaled[anomaly_indices, 4].astype(int),
            'CRC': unscaled[anomaly_indices, 5].astype(int),
            'ReTx': unscaled[anomaly_indices, 6].astype(int),
            'NDI': unscaled[anomaly_indices, 7].astype(int),
            'threshold': threshold,
            'anomaly_score': anomaly_scores[anomaly_indices],  # Add anomaly scores,
            'distance_from_threshold': (anomaly_scores[anomaly_indices] - threshold)
        })
        
        # Sort by anomaly score in descending order
        anomalies = anomalies.sort_values('anomaly_score', ascending=False)
        anomalies.to_csv(output_csv, index=False)

    logging.info(f"Anomalies saved to CSV => {output_csv}")

    # Create histogram of anomaly scores
    plt.figure(figsize=(10, 6))
    plt.hist(anomaly_scores, bins=50, edgecolor='black')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})')
    plt.title('Distribution of Anomaly Scores')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('subAdjacent/results/anomaly_scores_histogram.png')
    plt.close()


def calculate_threshold_evt(scores, q=0.99):
    # Fit generalized Pareto distribution
    tail_scores = scores[scores > np.percentile(scores, 95)]
    shape, loc, scale = stats.genpareto.fit(tail_scores)
    
    # Calculate threshold using inverse CDF
    threshold = stats.genpareto.ppf(q, shape, loc, scale)
    return threshold

def bring_approach(args):
    if args.approach == 'subAdjacent':
        from subAdjacent.trainer import train_model, detect_anomalies
        from subAdjacent.configClass import Config
        train_func = train_model
        detect_func = detect_anomalies
        config = Config()
    return {'params': config, 'train_func': train_func, 'detect_func': detect_func}
