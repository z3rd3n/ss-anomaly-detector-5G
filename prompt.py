My implementation is based on SuSub-Adjacent Transformer: Improving Time Series Anomaly Detection with Reconstruction Error from Sub-Adjacent Neighborhoods
and their code in the github is given ALSO BELOW

however, I'm always getting 
Found zero or near-zero row in attnMatrix! 
attnMatrix min/max: 0/1 errors. 

But it seems, they are not getting any of these attention matrix errors, looking at the both code, do I implement something wrong in the logic, surely I've changed some of those for my implementation but I try to follow the same logic where we change the attention matrix so that it focuses more on the sub-adjacent parts. Also do my original implementation wrong? How do I ensure better attention behaviors?

I also link the original paper, so look at it mathematically, carefully analyze both code and paper, and make sure that logic is correct. For example check the dimensions if they match in order, or for example my attn_matrix B,H,L,L etc., maybe check the einsums if the ordering is wrong. Since you are expert in machine learning, you will see if there is an error easily. Also their papers implementation is based on AnomalyTransformer so also you have information on that.





MY CODE:
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
            print("Found zero or near-zero row in attnMatrix! Debug info:")
            print("attnMatrix min/max:", attnMatrix.min().item(), attnMatrix.max().item())
            # Possibly bail out or clamp:
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
    
    all_scores = []
    all_rec = []
    all_sacon = []
    all_features = []
    all_timestamps = []

    criterion_mse = torch.nn.MSELoss(reduction='none')

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Detecting anomalies")):
            if params.debug and i >= 50:
                break
            features = batch['features'].to(params.device)
            timestamps_list = batch['timestamps']

            enc_out, queries_list, keys_list = model(features)
            # rec_loss shape = [B, seq_len, D], we reduce feature-dim => [B, seq_len]
            rec_loss = criterion_mse(enc_out, features).mean(dim=-1) #it means over features

            # SACon (averaged across layers)
            sacon_all_layers = torch.zeros_like(rec_loss)
            for (q, k_) in zip(queries_list, keys_list):
                sacon_all_layers += model.compute_sub_adj_contrib(q, k_, params.span, params.one_side)
            sacon_all_layers /= len(queries_list)

            # flatten
            rec_loss_flat = rec_loss.view(-1).cpu().numpy() # shape [B,L] => [B*L]
            sacon_flat = sacon_all_layers.view(-1).cpu().numpy() # shape [B,L] => [B*L]

            # store
            all_rec.append(rec_loss_flat)
            all_sacon.append(sacon_flat)
            all_features.append(features.view(-1, features.shape[-1]).cpu().numpy())
            all_timestamps.extend([t for sublist in timestamps_list for t in sublist])

    # concat
    all_rec = np.concatenate(all_rec, axis=0)
    all_sacon = np.concatenate(all_sacon, axis=0)
    all_features = np.concatenate(all_features, axis=0)

    # anomaly_score = rec_loss * softmax(-sacon)
    # We'll do a single global softmax across all points for simplicity:
    negative_sacon = -all_sacon
    # be mindful of big shape => do it with stable code
    sacon_weights = np.exp(negative_sacon - negative_sacon.max())
    sacon_weights /= (sacon_weights.sum() + 1e-6)

    all_scores = all_rec * sacon_weights  # multiply elementwise

    # threshold
    threshold = np.percentile(all_scores, 99)  # or whichever
    anomalies_mask = (all_scores > threshold)

    # Save to CSV (unscale -> int)
    unscale_and_save_anomalies(
        timestamps=all_timestamps,
        features=all_features,
        anomaly_scores=all_scores,
        threshold=threshold,
        output_csv=os.path.join(params.output_dir, "detected_anomalies.csv")
    )

    # Plot
    plot_anomalies(all_scores, anomalies_mask, threshold, params.output_dir)
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
            loss = rec_loss_scalar - lambda_sacon * sacon_mean

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
            loss_val = rec_loss - lambda_sacon * sacon_mean

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
        self.debug = False
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
        self.num_workers = 4
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
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import csv
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

        loss_mat = model.compute_sub_adj_contrib(queries, keys, model.span, model.one_side)
        
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
    logging.info(f"Total anomaly count: {len(anomaly_indices)} (threshold={threshold})")

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

def bring_approach(args):
    if args.approach == 'subAdjacent':
        from subAdjacent.trainer import train_model, detect_anomalies
        from subAdjacent.configClass import Config
        train_func = train_model
        detect_func = detect_anomalies
        config = Config()
    return {'params': config, 'train_func': train_func, 'detect_func': detect_func}


SOURCE CODE:

# AnomalyTransformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import LinearAnomalyAttention, AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
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

    def forward(self, x, attn_mask=None):
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
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True, linear_attn=True, mapping_fun='ours'):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        dim_per_head = d_model//n_heads
        self.linear_attn = linear_attn
        # Encoder
        if self.linear_attn:
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            LinearAnomalyAttention(win_size, False, attention_dropout=dropout,
                                                   output_attention=output_attention, dim_per_head=dim_per_head,
                                                   mapping_fun=mapping_fun),
                            d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for _ in range(e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            )
        else:
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            # LinearAnomalyAttention(win_size, False, attention_dropout=dropout,
                            #                        output_attention=output_attention, dim_per_head=dim_per_head),
                            AnomalyAttention(win_size, False, attention_dropout=dropout,
                                             output_attention=output_attention),
                            d_model, n_heads),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for _ in range(e_layers)
                ],
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
        
# attn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        tmp = torch.arange(win_size)
        self.distances = (tmp.unsqueeze(1)-tmp.unsqueeze(0)).abs().cuda()

    def forward(self, queries, keys, values):  # sigma, attn_mask
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        assert L == S, "L!=S"
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # if self.mask_flag:
        #     if attn_mask is None:
        #         attn_mask = TriangularCausalMask(B, L, device=queries.device)
        #     scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        # sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        # window_size = attn.shape[-1]
        # sigma = torch.sigmoid(sigma * 5) + 1e-5
        # sigma = torch.pow(3, sigma) - 1
        # sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        # prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        # prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("b h l s, b s h d -> b l h d", series, values)

        if self.output_attention:
            return V.contiguous(), series, None
        else:
            return V.contiguous(), None


class LinearAnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=False, scale=None, attention_dropout=0.0, output_attention=False,
                 dim_per_head=64, mapping_fun='ours'):
        super(LinearAnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = win_size
        self.softmax = nn.Softmax(dim=-1)
        self.mapping_fun = mapping_fun

        self.delta1 = nn.Parameter(torch.tensor(1.0))
        # self.delta2 = nn.Parameter(torch.tensor(1.0))
        # self.scale = nn.Parameter(torch.zeros(size=(1, 1, 1, dim_per_head)))

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        assert L == S, "L!=S"

        if self.mapping_fun == 'ours':
            queries[queries < 0] = -100
            keys[keys < 0] = -100
            queries = self.softmax(queries / nn.Softplus()(self.delta1))
            keys = self.softmax(keys / nn.Softplus()(self.delta1))
        elif self.mapping_fun == 'softmax_q_k':
            # softmax2
            queries = self.softmax(queries)
            softmax2 = nn.Softmax(dim=1)
            keys = softmax2(keys)
        elif self.mapping_fun == 'x_3':
            # x**3 and relu
            queries = nn.ReLU()(queries)
            keys = nn.ReLU()(keys)
            # x**3
            q_norm = queries.norm(dim=-1, keepdim=True)
            k_norm = keys.norm(dim=-1, keepdim=True)
            queries = queries**3
            keys = keys**3
            queries = queries / (queries.norm(dim=-1, keepdim=True)+1e-6) * q_norm.clone()
            keys = keys / (keys.norm(dim=-1, keepdim=True) + 1e-6) * k_norm.clone()
        elif self.mapping_fun == 'relu':
            queries = nn.ReLU()(queries)
            keys = nn.ReLU()(keys)
        elif self.mapping_fun == 'elu_plus_1':
            # elu+1
            queries = F.elu(queries) + 1
            keys = F.elu(keys) + 1

        kv = torch.einsum("b e h l, b l h f -> b h e f", keys.transpose(1, 3), values)

        z = 1 / (torch.einsum("b l h e, b h e -> b l h", queries, keys.sum(dim=1)) + 1e-6)
        V = torch.einsum("b l h e, b h e e, b l h -> b l h e", queries, kv, z)

        if self.output_attention:
            return V.contiguous(), queries, keys
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, queries, keys = self.inner_attention(
            queries,
            keys,
            values
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), queries, keys
    

# eval.py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import *
import matplotlib.pyplot as plt
from tadpak import pak
import time
import os
from scipy.stats import norm
import torch.nn.functional as F


def get_fp_tp_rate(predict, actual):
    tn, fp, fn, tp = confusion_matrix(actual, predict, labels=[0, 1]).ravel()

    # recall
    true_pos_rate = tp / (tp + fn)
    #
    false_pos_rate = fp / (fp + tn)

    return false_pos_rate, true_pos_rate


def pak_protocol(scores, labels, threshold, max_k=100):
    f1s = []
    ks = []
    fprs = []
    tprs = []
    preds = []

    for k in range(0, max_k + 1, 10):  # modify to range(0, max_k + 1, 1) for more precise result
        ks.append(k / 100)
        adjusted_preds = pak.pak(scores, labels, threshold, k=k)
        f1 = f1_score(labels, adjusted_preds)
        fpr, tpr = get_fp_tp_rate(adjusted_preds, labels)
        fprs.append(fpr)
        tprs.append(tpr)
        # print(f1)
        # print(k)
        f1s.append(f1)
        preds.append(adjusted_preds)

    area_under_f1 = auc(ks, f1s)
    max_f1_k = max(f1s)
    k_max = f1s.index(max_f1_k)
    preds_for_max = preds[f1s.index(max_f1_k)]
    # import matplotlib.pyplot as plt
    # plt.cla()
    # plt.plot(ks, f1s)
    # plt.savefig('DiffusionAE/plots/PAK_PROTOCOL')
    # print(f'AREA UNDER CURVE {area}')
    return area_under_f1, max_f1_k, k_max, preds_for_max, fprs, tprs


def evaluate(score, label, validation_thresh=None):
    if len(score) != len(label):
        score = score[:len(label)]
    false_pos_rates = []
    true_pos_rates = []
    f1s = []
    max_f1s_k = []
    preds = []
    # thresholds = np.arange(0, score.max(), min(0.001, score.max()/50))#0.001
    thresholds = np.arange(0, score.max() + 1, score.max() / 50)  # 0.001

    max_ks = []
    pairs = []

    for thresh in thresholds:
        f1, max_f1_k, k_max, best_preds, fprs, tprs = pak_protocol(score, label, thresh)
        max_f1s_k.append(max_f1_k)
        max_ks.append(k_max)
        preds.append(best_preds)
        false_pos_rates.append(fprs)
        true_pos_rates.append(tprs)
        f1s.append(f1)
        pairs.extend([(thresh, i) for i in range(101)])

    if validation_thresh:
        print(f'validation_thresh is provided: {validation_thresh}')
        f1, max_f1_k, max_k, best_preds, _, _ = pak_protocol(score, label, validation_thresh)
    else:
        print('validation_thresh is not provided')
        f1 = max(f1s)
        max_possible_f1 = max(max_f1s_k)
        max_idx = max_f1s_k.index(max_possible_f1)
        max_k = max_ks[max_idx]
        thresh_max_f1 = thresholds[max_idx]
        best_preds = preds[max_idx]
        best_thresh = thresholds[f1s.index(f1)]

    roc_max = auc(np.transpose(np.array(false_pos_rates))[max_k], np.transpose(np.array(true_pos_rates))[max_k])
    # np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/fprs_diff_score_pa.npy', np.transpose(false_pos_rates)[0])
    # np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/tprs_diff_score_pa.npy', np.transpose(true_pos_rates)[0])

    false_pos_rates = np.array(false_pos_rates).flatten()
    true_pos_rates = np.array(true_pos_rates).flatten()

    sorted_indexes = np.argsort(false_pos_rates)
    false_pos_rates = false_pos_rates[sorted_indexes]
    true_pos_rates = true_pos_rates[sorted_indexes]
    pairs = np.array(pairs)[sorted_indexes]
    roc_score = auc(false_pos_rates, true_pos_rates)

    # np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/tprs_diff_score.npy', true_pos_rates)
    # np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/fprs_diff_score.npy', false_pos_rates)
    # np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/pairs_diff_score.npy', pairs)
    # preds = predictions[f1s.index(f1)]
    if validation_thresh:
        return {
            'f1-AUC': f1,  # f1_k(area under f1) for validation threshold
            'ROC/AUC': roc_score,  # for all ks and all thresholds obtained on test scores
            'f1_max': max_f1_k,  # best f1 across k values
            'preds': best_preds,  # corresponding to best k
            'k': max_k,  # the k value correlated with the best f1 across k=1,100
            'thresh_max': validation_thresh,
            'roc_max': roc_score,
        }
    else:
        return {
            'f1-AUC': f1,
            'ROC/AUC': roc_score,
            'threshold': best_thresh,
            'f1_max': max_possible_f1,
            'roc_max': roc_max,
            'thresh_max': thresh_max_f1,
            'preds': best_preds,
            'k': max_k,
        }, false_pos_rates, true_pos_rates


def get_pred_from_loss(test_energy, test_rec_loss, thresh, thresh_rec_loss):
    pred = (test_energy > thresh).astype(int)
    pred2 = (test_rec_loss > thresh_rec_loss).astype(int)

    fixMode = False
    for i in range(len(pred)):
        if pred[i] == 1 and not fixMode:
            fixMode = True
            for j in range(i - 1, 0, -1):
                if pred2[j]:
                    pred[j] = 1
                else:
                    break
            for j in range(i + 1, len(pred)):
                if pred2[j]:
                    pred[j] = 1
                else:
                    break
        elif pred[i] == 0:
            fixMode = False

    return pred


def write_into_xls(att_matrix, excel_name):
    folder_name = os.path.dirname(excel_name)
    if folder_name:
        os.makedirs(folder_name, exist_ok=True)
    dataframe = pd.DataFrame(att_matrix)
    # print(dataframe)
    # print(excel_name)
    dataframe.to_excel(excel_name, index=False)


def myplot(test_labels, test_data, cri_loss, rec_loss, att_loss, dataset_name='dataset', anomaly_score=None):
    ind = np.nonzero(test_labels)[0]
    win_size = 1000

    nums = np.arange(test_data.shape[-1])
    dim = np.random.choice(nums, 2, replace=True)

    # print(ind)
    start = np.maximum(np.random.choice(ind) - 100, 0)
    x = np.arange(start, np.minimum(start + win_size, len(test_labels)))
    y1 = test_labels[x]
    y2 = test_data[x, dim[0]]
    y3 = test_data[x, dim[1]]

    print(f'testdata max: {np.max(test_data)}; min: {np.min(test_data)}')

    y4 = cri_loss[x]
    y5 = rec_loss[x]
    y6 = att_loss[x]

    # 创建一个2*2的子图布局
    fig, axs = plt.subplots(2, 3)

    # 在第一个子图上画第一条曲线
    axs[0, 0].plot(x, y1)
    axs[0, 0].set_title(dataset_name + ' test_labels')

    axs[0, 1].plot(x, y2)
    axs[0, 1].set_title(f'{dataset_name} data dim:{dim[0]}')

    axs[0, 2].plot(x, y3)
    axs[0, 2].set_title(f'{dataset_name} data dim:{dim[1]}')

    # 在第二个子图上画第二条曲线
    axs[1, 0].plot(x, y4)
    axs[1, 0].set_title('cri_loss')

    # 在第三个子图上画第三条曲线
    axs[1, 1].plot(x, y5)
    axs[1, 1].set_title('rec_loss')

    # 在第四个子图上画第四条曲线
    axs[1, 2].plot(x, y6)
    axs[1, 2].set_title('att_loss')

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()

    # 保存图像为png格式，文件名为当前时间戳
    timestamp = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    os.makedirs('output', exist_ok=True)
    fig.savefig(os.path.join('output', f'{dataset_name}-{timestamp}.png'))

    # save to excel
    if len(test_data) < 1000:
        excel_name = os.path.join('output', f'{dataset_name}-{timestamp}-rec-attn-score-energy.xlsx')
        if anomaly_score is not None:
            att_matrix = np.concatenate((test_data, rec_loss.reshape(-1, 1), att_loss.reshape(-1, 1),
                                         anomaly_score.reshape(-1, 1), cri_loss.reshape(-1, 1),
                                         test_labels.reshape(-1, 1)), axis=1)
        else:
            att_matrix = np.concatenate((test_data, rec_loss.reshape(-1, 1), att_loss.reshape(-1, 1),
                                         cri_loss.reshape(-1, 1),  test_labels.reshape(-1, 1),), axis=1)
        write_into_xls(att_matrix, excel_name)


def compute_longest_anomaly(test_labels):
    list_ = []
    countFlag = False
    count = 0
    for label in test_labels:
        if label:
            if countFlag:
                count = count + 1
            else:
                countFlag = True
                count = 1
        else:
            if countFlag:
                countFlag = False
                list_.append(count)
                count = 0
    if count:
        list_.append(count)

    return np.max(np.array(list_)), np.min(np.array(list_))


def plot_mat(att_matrix, str0='tmp'):
    if not isinstance(att_matrix, np.ndarray):
        att_matrix = np.array(att_matrix)
    fig, axs = plt.subplots(1, 1)
    plt.imshow(att_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    timestamp = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    plt.savefig(os.path.join('output', f'attn_mat_{str0}-{timestamp}.png'))
    plt.show()
    # save to excel
    excel_name = os.path.join('output', f'attn_mat_{str0}-{timestamp}.xlsx')
    write_into_xls(att_matrix, excel_name)


def myLoss(queries, keys, span=None, one_side=True):
    # queries,keys : B,L,H,D  --> output: B,L
    # how much the point can help others; anomaly helps little while the normals help more
    L = queries.shape[1]
    if span is None:
        span = [20, 30]

    assert L >= span[1] >= span[0] >= 0

    z = 1 / (torch.einsum("b l h e, b h e -> b l h", queries, keys.sum(dim=1)) + 1e-6)
    # lossMat0 = (queries * keys).sum(dim=-1) * z
    lossMat = None
    for k in range(-span[1], span[1] + 1):  # range(-span[1], -span[0]+1)
        # only one-side is used
        if one_side:
            if k < span[0]:
                continue
        else:
            if abs(k) < span[0]:
                continue

        shifted_queries = torch.roll(queries, shifts=k, dims=1)
        shifted_z = torch.roll(z, shifts=k, dims=1)

        if lossMat is None:
            # b,l,h
            lossMat = (shifted_queries * keys).sum(dim=-1) * shifted_z
        else:
            lossMat += (shifted_queries * keys).sum(dim=-1) * shifted_z

    # b,l,h
    lossMat = torch.mean(lossMat, dim=-1)

    return lossMat  # B,L


def myLossNew(queries, keys, span=None, one_side=True):
    # queries,keys : B,L,H,D  --> output: B,L
    # explicitly compute attention matrix
    # how much the point can help others; anomaly helps little while the normals help more
    L = queries.shape[1]
    if span is None:
        span = [20, 30]

    assert L >= span[1] >= span[0] >= 0

    # compute attention matrix
    attnMatrix = torch.einsum("b l h e, b s h e -> b h l s", queries, keys)
    attnMatrix = attnMatrix / attnMatrix.sum(dim=-1, keepdim=True)

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


def myLoss2(attnMatrix, keys=None, span=None, one_side=True):
    # traditional self attention
    # b h l l

    B, H, L, _ = attnMatrix.shape
    lossMat = None
    for k in range(20, 30):
        diag1 = torch.diagonal(attnMatrix, offset=k, dim1=-2, dim2=-1)
        p1d = (k, 0)
        diag1 = F.pad(diag1, p1d)

        if lossMat is None:
            lossMat = diag1
        else:
            lossMat += diag1

        # -(L-k)
        diag1 = torch.diagonal(attnMatrix, offset=-(L-k), dim1=-2, dim2=-1)  # why use L-k ?  L-k performs better
        p1d = (0, L-k)  # p1d = (0, k)
        diag1 = F.pad(diag1, p1d)
        lossMat += diag1

        # # -k
        # diag1 = torch.diagonal(attnMatrix, offset=-k, dim1=-2, dim2=-1)
        # p1d = (0, k)
        # diag1 = F.pad(diag1, p1d)
        # lossMat += diag1
        #
        # # (L-k)
        # diag1 = torch.diagonal(attnMatrix, offset=L-k, dim1=-2, dim2=-1)
        # p1d = (L-k, 0)
        # diag1 = F.pad(diag1, p1d)
        # lossMat += diag1

        lossMat += diag1

    lossMat = torch.mean(lossMat, dim=1)

    return lossMat


def myLoss0(queries, keys):
    # all points are used, except itself
    attn_mat = queries.permute([0, 2, 1, 3]) @ keys.permute([0, 2, 3, 1])
    z = 1 / (torch.einsum("b l h e, b h e -> b h l", queries, keys.sum(dim=1)) + 1e-6)
    attn_mat = attn_mat * z.unsqueeze(-1)
    loss_mat = attn_mat.sum(dim=-2) - torch.diagonal(attn_mat, offset=0, dim1=-2, dim2=-1)
    return loss_mat.mean(dim=1)


def softmax(x, temperature=1, window=None):
    # softmax for numpy
    # print(x)
    x = x * temperature
    shape = x.shape[0]
    if window is not None:
        window = max(min(int(window), shape),1)
        rem = shape % window
        if rem != 0:
            x = np.concatenate([x, x[:int(window - rem)]], axis=0)

        x = x.reshape(-1, window)

    x = x.clip(-100, 100)
    output = (np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)).reshape(-1)
    # print(output)
    return output[:shape]


def sliding_window_mean_std(vector, window_size=100):
    vector = previous_vector(vector, shift=5)
    mean = np.convolve(vector, np.ones(window_size), 'same') / window_size
    if len(mean) > len(vector):
        mean = mean[:len(vector)]
    std = np.sqrt(np.convolve(np.square(vector - mean), np.ones(window_size), 'same') / (window_size - 1))
    return mean, std


def previous_vector(vector, shift=10):
    return np.roll(vector, shift=-abs(shift))


def get_probs_from_cri(results, window_size=10):
    std_gauss = norm(loc=0, scale=1)
    mean, _ = sliding_window_mean_std(results, window_size=window_size)
    # _, std = sliding_window_mean_std(results, window_size=1000)
    std = results.std()
    # print(std)
    probs = -std_gauss.logsf((results - mean) / std)
    return probs


def use_smooth(vector, kernel_length=100):
    if kernel_length <= 0:
        return vector
    std_gauss = norm(loc=0, scale=1)
    x = np.linspace(-3, 3, int(kernel_length))
    kernel = std_gauss.pdf(x)
    vector = np.convolve(vector, kernel, 'same')
    return vector

# solver.py
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging
import time
from utils.utils import *
from utils.eval import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment
import matplotlib.pyplot as plt
from thop import profile


def my_kl_loss(p, q):
    # p,q : B,H,L,L
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)  # B,L


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, model, path):
        loss = val_loss
        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint(val_loss, model, path)
        elif loss > self.best_loss + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} because score {loss} '
                  f'> best_score {self.best_loss}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss


def point_adjust(pred, gt):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    pred = np.array(pred)
    return pred


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               step=self.train_data_step,
                                               ratio=self.train_data_ratio, mode='train',
                                               dataset=self.dataset)
        self.train_nolap_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                                     step=self.train_data_step,
                                                     ratio=self.train_data_ratio, mode='train-nolap',
                                                     dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              step=self.train_data_step,
                                              ratio=self.train_data_ratio, mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              step=self.train_data_step,
                                              ratio=self.train_data_ratio, mode='thres',
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

        if self.no_linear_attn:
            self.loss_fun = myLoss2
        else:
            self.loss_fun = myLossNew

        str1 = ''
        if self.no_point_adjustment:
            str1 = str1 + '_no_pa'
        if self.no_linear_attn:
            str1 = str1 + '_no_li'
        self.checkpoint_file = os.path.join(self.model_save_path, str(self.dataset) + '_checkpoint' + str1 + '.pth')

        log_file = os.path.join('logs', f'log-{self.dataset}.log')
        logging.basicConfig(filename=log_file, filemode='a', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def build_model(self):
        linear_attn = not self.no_linear_attn
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3,
                                        linear_attn=linear_attn, mapping_fun=self.mapping_function)
        self.model2 = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3,
                                         linear_attn=linear_attn, mapping_fun=self.mapping_function)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()
            self.model2.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input_ = input_data.float().to(self.device)
            output, queries_list, keys_list = self.model(input_)
            len_list = len(queries_list)
            loss_attn = 0.0

            for u in range(len_list):
                loss_attn += self.loss_fun(queries_list[u], keys_list[u], self.span, self.oneside).mean()

            loss_attn = loss_attn / len_list

            rec_loss = self.criterion(output, input_)

            thisLoss1 = rec_loss
            thisLoss2 = rec_loss - self.k * loss_attn

            loss_1.append(thisLoss1.item())
            loss_2.append(thisLoss2.item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        info = "======================TRAIN MODE======================"
        print(info)
        logging.info(info)

        self.build_model()

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        # early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        params, flops = 0, 0
        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []
            loss2_list = []
            epoch_start_time = time.time()
            self.model.train()

            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input_ = input_data.float().to(self.device)

                if epoch == 0 and i == 0:
                    input_profile = input_[[0]]
                    flops, params = profile(self.model2, inputs=(input_profile,))
                    flops = flops/1e9
                    params = params/1e6
                    epoch_start_time = time.time()

                output, queries_list, keys_list = self.model(input_)
                len_list = len(queries_list)

                # calculate Association discrepancy
                loss_attn = 0.0
                if not self.no_point_adjustment:
                    for u in range(len_list):
                        # b,l,h,d
                        loss_attn += self.loss_fun(queries_list[u], keys_list[u], self.span, self.oneside).mean()
                else:
                    loss_attn = 0

                loss_attn = loss_attn / len_list

                rec_loss = self.criterion(output, input_)

                loss1 = rec_loss
                loss2 = rec_loss - self.k * loss_attn  # loss_attn is used to distinguish normals and anomalies
                # test Equivalence
                loss3 = 2*rec_loss - self.k * loss_attn

                loss1_list.append(loss1.item())
                loss2_list.append(loss2.item())

                if (i + 1) % 50 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)

                    info = (
                        f'\t loss1: {loss1:.4f}, loss2: {loss2:.4f}; rec_loss: {rec_loss:.4f}, loss_attn: {loss_attn:.4f}'
                        f' speed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    print(info)
                    logging.info(info)

                    iter_count = 0
                    time_now = time.time()

                if not self.no_point_adjustment:
                    # using point adjustment
                    loss3.backward()
                    # loss1.backward()  # just for attention matrix plot
                else:
                    # no_point_adjustment
                    loss3.backward()

                self.optimizer.step()

            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0 * 1024.0)
            epoch_time = time.time() - epoch_start_time
            info = (f"Epoch: {epoch + 1} cost time: {epoch_time}, memory: {memory_used:.2f}GB, "
                    f"flops: {flops}GFLOPS, params: {params}M")
            print(info)
            logging.info(info)

            torch.save(self.model.state_dict(), self.checkpoint_file)
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

        return epoch_time, flops, params, memory_used

    def test(self):
        self.model.load_state_dict(torch.load(self.checkpoint_file))
        self.model.eval()
        temperature = self.temperature
        softmax_span = self.softmax_span

        info = "======================TEST MODE======================"
        print(info)
        logging.info(info)

        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        attens_energy = []
        loss_list = []
        for i, (input_data, labels) in enumerate(self.train_nolap_loader):
            input = input_data.float().to(self.device)
            output, queries_list, keys_list = self.model(input)
            len_list = len(queries_list)
            # loss [B, L]  [256, 100]
            loss = torch.mean(criterion(input, output), dim=-1)
            loss_attn = 0.0
            for u in range(len_list):
                if u == 0:
                    # b,l
                    loss_attn = self.loss_fun(queries_list[u], keys_list[u], self.span, self.oneside)  # * temperature
                else:
                    loss_attn += self.loss_fun(queries_list[u], keys_list[u], self.span, self.oneside)  # * temperature

            # metric = torch.softmax(-loss_attn / len_list, dim=-1)
            loss_attn = loss_attn / len_list

            loss_attn = loss_attn.detach().cpu().numpy()
            attens_energy.append(loss_attn)
            loss_list.append(loss.detach().cpu().numpy())

        train_attn_array = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_loss_array = np.concatenate(loss_list, axis=0).reshape(-1)

        # aggregation
        if not self.no_point_adjustment:
            train_energy = softmax(-train_attn_array, temperature=temperature, window=softmax_span) * train_loss_array
        else:
            # no point adjustment
            train_energy = train_loss_array  # / np.maximum(train_attn_array, 1e-6)

        # (2) find the threshold
        attens_energy = []
        loss_list = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, queries_list, keys_list = self.model(input)
            len_list = len(queries_list)

            loss = torch.mean(criterion(input, output), dim=-1)

            loss_attn = 0.0
            for u in range(len_list):
                if u == 0:
                    # b,l
                    loss_attn = self.loss_fun(queries_list[u], keys_list[u], self.span, self.oneside)
                else:
                    loss_attn += self.loss_fun(queries_list[u], keys_list[u], self.span, self.oneside)

            # Metric
            loss_attn = loss_attn / len_list

            loss_attn = loss_attn.detach().cpu().numpy()
            attens_energy.append(loss_attn)
            loss_list.append(loss.detach().cpu().numpy())

        test_attn_array = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_loss_array = np.concatenate(loss_list, axis=0).reshape(-1)

        # aggregation
        if not self.no_point_adjustment:
            test_energy = softmax(-test_attn_array, temperature=temperature, window=softmax_span) * test_loss_array
        else:
            # no point adjustment
            # test_energy = test_loss_array / np.maximum(test_attn_array, 1e-6)
            # test_energy = test_loss_array * np.exp(-test_attn_array * self.attn_temp)
            test_energy = test_loss_array
            # test_energy = test_loss_array * -np.log(test_attn_array)
            # test_energy = np.maximum(softmax(-test_attn_array, temperature=temperature, window=softmax_span),
            #                          1 / np.maximum(test_attn_array, 1e-6)) * test_loss_array

        # probs
        if not self.no_gauss_dynamic and not self.no_point_adjustment:
            train_energy = get_probs_from_cri(train_energy)
            test_energy = get_probs_from_cri(test_energy)

        # smooth it out
        if self.no_point_adjustment:
            # no point adjustment; smooth it out
            train_energy = use_smooth(train_energy, kernel_length=self.kernel_length)
            test_energy = use_smooth(test_energy, kernel_length=self.kernel_length)

        # thres
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)

        info = f"Threshold : {thresh}"
        print(info)
        logging.info(info)

        combined_loss = np.concatenate([train_loss_array, test_loss_array], axis=0)  #
        thresh_rec_loss = np.percentile(combined_loss, 100 - self.anormly_ratio)

        info = f"Threshold for reconstruction loss: {thresh_rec_loss}"
        print(info)
        logging.info(info)

        # (3) evaluation on the test set
        test_labels = []
        test_data = []
        attens_energy = []

        loss_list = []

        eval_time = 0
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input_ = input_data.float().to(self.device)

            start_time = time.time()
            output, queries_list, keys_list = self.model(input_)
            eval_time += time.time() - start_time

            len_list = len(queries_list)

            loss = torch.mean(criterion(input_, output), dim=-1)

            loss_attn = []
            for u in range(len_list):
                if u == 0:
                    # b,l
                    loss_attn = self.loss_fun(queries_list[u], keys_list[u], self.span, self.oneside)
                else:
                    loss_attn += self.loss_fun(queries_list[u], keys_list[u], self.span, self.oneside)

            # Metric
            loss_attn = loss_attn / len_list

            loss_attn = loss_attn.detach().cpu().numpy()
            attens_energy.append(loss_attn)

            loss_list.append(loss.detach().cpu().numpy())

            # test_label
            test_labels.append(labels)
            # test data
            test_data.append(input_data)

        if self.record_state:
            # just for recording state
            return eval_time

        test_attn_array = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_att_loss2 = test_attn_array
        test_rec_loss = np.concatenate(loss_list, axis=0).reshape(-1)

        # time
        info = f"{self.dataset} Evaluation: cost time: {eval_time}"
        print(info)
        logging.info(info)

        # aggregation
        anomaly_score = None
        if not self.no_point_adjustment:
            test_att_loss2 = softmax(-test_attn_array, temperature=temperature, window=softmax_span)
            test_energy = test_att_loss2 * test_rec_loss
            anomaly_score = test_energy
        else:
            # no point adjustment
            # test_energy = test_loss_array / np.maximum(test_attn_array, 1e-6)
            # test_energy = test_loss_array * np.exp(-test_attn_array * self.attn_temp)
            test_energy = test_loss_array
            # test_energy = test_loss_array * -np.log(test_attn_array)
            # test_energy = np.maximum(softmax(-test_attn_array, temperature=temperature, window=softmax_span),
            #                          1 / np.maximum(test_attn_array, 1e-6)) * test_loss_array

        # use prob, not for non-point-adjustment
        if not self.no_gauss_dynamic and not self.no_point_adjustment:
            test_energy = get_probs_from_cri(test_energy)

        # smooth it out
        if self.no_point_adjustment:
            # no point adjustment; smooth it out
            test_energy = use_smooth(test_energy, kernel_length=self.kernel_length)

        # test labels
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        lmax, lmin = compute_longest_anomaly(test_labels)
        print(f'The anomaly span is [{lmin}-{lmax}]')
        print(f'anomaly ratio in test set: {sum(test_labels)} / {len(test_labels)} = '
              f'{sum(test_labels) / len(test_labels):.2%}')

        # b,l,d
        test_data = np.concatenate(test_data, axis=0).reshape(-1, test_data[0].shape[-1])



        # plot something
        if self.monte_carlo <= 1 and self.mode in ['test', 'monte_carlo']:

            myplot(test_labels, test_data, test_energy, test_rec_loss, test_attn_array, str(self.dataset), anomaly_score)
            print('here')
            # use test_attn_array instead of test_att_loss2

            # plot attn_matrix
            att_matrix = queries_list[-1].detach().cpu().numpy()
            if att_matrix.shape[-1] == att_matrix.shape[-2]:
                batch_idx = random.randint(0, att_matrix.shape[0] - 1)
                head_idx = random.randint(0, att_matrix.shape[1] - 1)
                att_matrix = att_matrix[batch_idx, head_idx, :, :]
                strflag = 'Vanilla_'
            else:
                # queries, keys: B, L, H, D
                batch_idx = random.randint(0, att_matrix.shape[0] - 1)
                head_idx = random.randint(0, att_matrix.shape[2] - 1)
                Q_mat = att_matrix[batch_idx, :, head_idx, :]
                K_mat = (keys_list[-1].detach().cpu().numpy())[batch_idx, :, head_idx, :]
                att_matrix = Q_mat @ K_mat.T
                att_matrix = att_matrix / att_matrix.sum(axis=1, keepdims=True)
                strflag = 'Linear_'
            plot_mat(att_matrix, strflag + str(self.dataset) + f"_batch_{batch_idx}_head_{head_idx}")

        gt = test_labels.astype(int)

        # get final pred
        pred = (test_energy > thresh).astype(int)
        # pred = np.random.rand(gt.shape[0]) > 0.8  # F:79
        if not self.no_point_adjustment:
            pred = point_adjust(pred, gt)

        info = f"pred: {pred.shape}"
        print(info)
        logging.info(info)
        info = f"gt: {gt.shape}"
        print(info)
        logging.info(info)

        ############    evaluate     ############
        print('------------Begin evaluating------------')
        # eval_dict = evaluate(pred, gt, validation_thresh=None)  # , false_pos_rates, true_pos_rates
        # if isinstance(eval_dict, tuple):
        #     eval_dict = eval_dict[0]
        # print('------------ eval Metrics -------------')
        # for k, v in eval_dict.items():
        #     info = f'{k}:\t{v}'
        #     print(info)
        #     logging.info(info)

        # detection adjustment: please see this issue for more information
        # https://github.com/thuml/Anomaly-Transformer/issues/14
        # pred = point_adjust(pred, gt)
        # gt = np.array(gt)
        # print("pred: ", pred.shape)
        # print("gt:   ", gt.shape)
        #
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')

        info = f"Accuracy : {accuracy:.2%}, Precision : {precision:.2%}, Recall : {recall:.2%}, F-score : {f_score:.2%}"
        print(info)
        logging.info(info)

        print('-------------- End ----------------')
        logging.info('-------------- End ----------------')

        # write into txt
        timestamp = time.strftime("%Y%m%d %H:%M:%S", time.localtime())
        os.makedirs('output', exist_ok=True)
        with open(os.path.join('output', str(self.dataset) + '-result.txt'), 'a') as f:
            f.write(timestamp + '\n')
            f.write(f"\tAccuracy : {accuracy:.2%}, Precision : {precision:.2%}, Recall : {recall:.2%}, "
                    f"F-score : {f_score:.2%}" + '\n')
            # for k, v in eval_dict.items():
            #     f.write(f'\t{k}:\t{v}' + '\n')
            f.write('\n')

        return accuracy, precision, recall, f_score
    

    %%%% ijcai24.tex

\typeout{IJCAI--24 Instructions for Authors}

% These are the instructions for authors for IJCAI-24.

\documentclass{article}
\pdfpagewidth=8.5in
\pdfpageheight=11in

% The file ijcai24.sty is a copy from ijcai22.sty
% The file ijcai22.sty is NOT the same as previous years'
\usepackage{ijcai24}

% Use the postscript times font!
\usepackage{times}
\usepackage{soul}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\usepackage[utf8]{inputenc}
\usepackage[small]{caption}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amsthm}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage[switch]{lineno}

\usepackage{amssymb}
\usepackage{newtxmath}
\usepackage{makecell}

\usepackage{multirow}
\usepackage[normalem]{ulem}
\useunder{\uline}{\ul}{}
\usepackage[table,xcdraw]{xcolor}
\usepackage{marvosym}
\usepackage{rotating}
\usepackage{hyperref}

% Comment out this line in the camera-ready submission
% \linenumbers

\urlstyle{same}

% the following package is optional:
%\usepackage{latexsym}

% See https://www.overleaf.com/learn/latex/theorems_and_proofs
% for a nice explanation of how to define new theorems, but keep
% in mind that the amsthm package is already included in this
% template and that you must *not* alter the styling.
\newtheorem{example}{Example}
\newtheorem{theorem}{Theorem}

% Following comment is from ijcai97-submit.tex:
% The preparation of these files was supported by Schlumberger Palo Alto
% Research, AT\&T Bell Laboratories, and Morgan Kaufmann Publishers.
% Shirley Jowell, of Morgan Kaufmann Publishers, and Peter F.
% Patel-Schneider, of AT\&T Bell Laboratories collaborated on their
% preparation.

% These instructions can be modified and used in other conferences as long
% as credit to the authors and supporting agencies is retained, this notice
% is not changed, and further modification or reuse is not restricted.
% Neither Shirley Jowell nor Peter F. Patel-Schneider can be listed as
% contacts for providing assistance without their prior permission.

% To use for other conferences, change references to files and the
% conference appropriate and use other authors, contacts, publishers, and
% organizations.
% Also change the deadline and address for returning papers and the length and
% page charge instructions.
% Put where the files are available in the appropriate places.


% PDF Info Is REQUIRED.

% Please leave this \pdfinfo block untouched both for the submission and
% Camera Ready Copy. Do not include Title and Author information in the pdfinfo section
\pdfinfo{
/TemplateVersion (IJCAI.2024.0)
}

\title{Sub-Adjacent Transformer: Improving Time Series Anomaly Detection \\ with Reconstruction Error from Sub-Adjacent Neighborhoods}


% Single author syntax
% \author{
%     Author Name
%     \affiliations
%     Affiliation
%     \emails
%     email@example.com
% }

% Multiple author syntax (remove the single-author syntax above and the \iffalse ... \fi here)
% \iffalse
\author{
Wenzhen Yue$^1$
\and
Xianghua Ying$^1$\footnote{Corresponding Author}\and
Ruohao Guo$^{1}$\and
DongDong Chen$^2$\and
Ji Shi$^1$\and
Bowei Xing$^1$\and
Yuqing Zhu$^3$\And
Taiyan Chen$^1$
\affiliations
$^1$National Key Laboratory of General Artificial Intelligence, School of Intelligence Science and Technology, Peking University \\
$^2$Microsoft Cloud + AI \\ 
$^3$Tsinghua University \\
\emails
yuewenzhen@stu.pku.edu.cn,
xhying@pku.edu.cn, 	dochen@microsoft.com
}
% \fi

\begin{document}

\maketitle

\begin{abstract}
    In this paper, we present the Sub-Adjacent Transformer with a novel attention mechanism for unsupervised time series anomaly detection. Unlike previous approaches that rely on all the points within some neighborhood for time point reconstruction, our method restricts the attention to regions not immediately adjacent to the target points, termed {\em sub-adjacent neighborhoods}. Our key observation is that owing to the rarity of anomalies, they typically exhibit more pronounced differences from their sub-adjacent neighborhoods than from their immediate vicinities. By focusing the attention on the sub-adjacent areas, we make the reconstruction of anomalies more challenging, thereby enhancing their detectability. Technically, our approach concentrates attention on the non-diagonal areas of the attention matrix by enlarging the corresponding elements in the training stage. To facilitate the implementation of the desired attention matrix pattern, we adopt linear attention because of its flexibility and adaptability. Moreover, a learnable mapping function is proposed to improve the performance of linear attention. Empirically, the Sub-Adjacent Transformer achieves state-of-the-art performance across six real-world anomaly detection benchmarks, covering diverse fields such as server monitoring, space exploration, and water treatment. 

\end{abstract}

% The code is available at \href{https://github.com/jackyue1994/Sub-Adjacent-Transformer}{https://github.com/jackyue1994/Sub-Adjacent-Transformer}.

\section{Introduction}

\begin{figure}[t]
   \centering
   \includegraphics[width=1\linewidth]{figures/Visio_fig1_temporal_.pdf}
   \caption{Illustration of our method in time domain. The point marked with a red circle, along with its neighbors, represents an anomaly on the sinusoidal signal. (a) Previous works typically utilize the attention across all the points within the window, while (b) our method encourages the use of attention of sub-adjacent neighborhoods (the highlighted area). Such an imposed constraint enlarges the reconstruction challenge for anomalies and thus improves the anomaly detection performance.}
   \label{fig1}
\end{figure}

In modern industrial systems such as data centers and smart factories, numerous sensors consistently produce significant volumes of measurements  ~\cite{zhou2019beatgan,thoc}. To effectively monitor the systems' real-time conditions and avoid potential losses, it is crucial to identify anomalies in the multivariate time series ~\cite{intro1,intro2}. This problem is known as time series anomaly detection ~\cite{acmreview:2021}. Comprehensive surveys can be found in the literature ~\cite{acmreview:2021,evaluation,ieeereview:2013}. ~\nocite{ijcai2019_ensemble}

Effective and robust time series anomaly detection remains a challenging and open problem ~\cite{intro3}. Real-world time series usually exhibit nonlinear dependencies and intricate interactions among time points. Besides, the vast scale of data makes the labeling of these anomalies time-consuming and costly in practice  ~\cite{nominality}. Therefore, time series anomaly detection is usually conducted in an unsupervised manner ~\cite{anomalytrans}, which is also the focus of this paper.

Before the era of deep learning, many classic anomaly detection methods were proposed. These include the density-estimation method proposed in ~\cite{breunig2000lof}, the clustering-based method presented in one-class SVM ~\cite{ocsvm2001}, and a series of SVDD works ~\cite{svdd2004,liu2013svdd}, along with graph-based methods ~\cite{graph2008}. Traditional methods, which use handcrafted features, struggle to accurately express the relationship among time points and often suffer from poor generalization. Deep learning-based  techniques have witnessed a diversity of methods during the recent decade. Broadly, they can be categorized into three main categories: Reconstruction-based methods reconstruct the time series and compare it with the original data ~\cite{lstm_vae_rec,rec_2021,rec_rnn_OmniAnomaly,diffusion}, whereas prediction-based methods predict future points for comparison with actual data ~\cite{gnn2021_pred,pred2,lstm-gmm-pred,nominality}. Dissimilarity-based methods emphasize the discrepancy representation between normal and abnormal points in various domains ~\cite{thoc,dissimilarity2021,anomalytrans}. It is noteworthy that these methods often combine multiple approaches to enhance performance. For example, TranAD ~\cite{tranad2022} integrates an integrated reconstruction error and discriminator loss; while ~\cite{nominality} uses prediction results as the nominal data and combines them with reconstruction results. 
~\nocite{fox1972outliers}

In this paper, we focus on the attention matrix within the Transformer module. Transformers ~\cite{transformer} have achieved great success in natural language processing ~\cite{gpt3_2020}, computer vision ~\cite{swintrans} and time series ~\cite{anomalytrans} in recent years. Previous work in \cite{anomalytrans} primarily concentrated on the dissimilarities in attention distributions. Our paper takes a different approach and introduces a simple yet effective attention learning paradigm. Our fundamental assumption is that anomalies are less related to their non-immediate neighborhoods when compared with normal points. Therefore, focusing exclusively on these non-immediate neighborhoods is likely to result in larger reconstruction errors for anomalies. Based on this assumption, our study introduces two key concepts: {\em sub-adjacent neighborhoods} and {\em sub-adjacent attention contribution}. As shown in Figure \ref{fig1}, sub-adjacent neighborhoods indicate the areas not immediately adjacent to the target point, and the sub-adjacent attention contribution is defined as the sum of particular non-diagonal elements in the corresponding column of the attention matrix. These concepts are integrated with reconstruction loss, forming the cornerstone of our anomaly detection strategy. Furthermore, we observe that the traditional {\tt Softmax} operation used in standard self-attention impedes the formation of the desired attention matrix, where the predefined non-diagonal stripes are dominant. In response, we adopt linear attention ~\cite{shen2021efficient,flattentrans} for its greater flexibility in attention matrix configurations. We also tailor the mapping function within this framework, using learnable parameters to enhance performance. The main contributions of this paper are summarized as follows.

\begin{itemize}
    \item We propose a novel attention learning regime based on the sub-adjacent neighborhoods and attention contribution. Specifically, a new attention matrix pattern is designed to enhance discrimination between anomalies and normal points. 
    \item Furthermore, we leverage the linear attention to achieve the desired attention pattern. To the best of our knowledge, this is the first introduction of linear attention with a learnable mapping function to time series anomaly detection.
    \item Extensive experiments show that the proposed Sub-Adjacent Transformer delivers state-of-the-art (SOTA) performance across six real-world benchmarks and one synthetic benchmark.      
\end{itemize}

\section{Related Works}

\paragraph{Time Series Anomaly detection.} In recent year, graph neural networks and self-attention have been explored in time series anomaly detection ~\cite{gnn2021_pred,gnn2020,Series2graph2022,anomalytrans,nominality,timesnet}. The most related work to ours is Anomaly Transformer ~\cite{anomalytrans}, which also exploits the attention information. The discrepancy between the learned Gaussian distribution and the actual distribution is used to distinguish abnormal points from normal ones. A somewhat complicated min-max training strategy is adopted to train the model. Generally, Anomaly Transformer is complex and somewhat indirect. Our proposed method is more straightforward and exhibits improved performance. ~\nocite{gnn2020,gnn2021_pred,Series2graph2022}

\paragraph{Linear Attention.} Compared with vanilla self-attention, linear attention enjoys more flexibility and lighter computation burden. Existing linear attention methods can be divided into three categories: pattern based methods, kernel based methods and mapping based methods. We mainly focus on mapping based methods in this paper. Efficient attention ~\cite{shen2021efficient} applies {\tt Softmax} function to $\mathbf{Q}$ (in a row-wise manner) and $K$ (in a column-wise manner) to ensure each row of $\mathbf{Q}\mathbf{K}^T$ sums up to 1. Hydra attention ~\cite{hydranet} studies the special case where attention heads are as many as feature dimension and $\mathbf{Q}$ and $\mathbf{K}$ are normalized so that $\left \| \mathbf{Q}_{i}  \right \|$  and $\left \| \mathbf{K}_{i}  \right \|$ share the same value. EfficientVit ~\cite{efficientvit} uses the simple ReLU function as the mapping function. FLatten Transformer ~\cite{flattentrans} proposes the power function as the mapping function to improve the focus capability. In this paper, we apply {\tt Softmax} to matrices $\mathbf{Q}$ and $\mathbf{K}$ both in a row-wise manner to approximate the focus property of the vanilla self-attention. Empirical experiments verify its performance superiority to other mapping methods in time series anomaly detection. 

\section{Methods}

\subsection{Problem Formulation}

Let $\mathbf{X} = \left \{ \mathbf{x}_1, \cdots, \mathbf{x}_T \right \}\in \mathbb{R}^{T\times D} $ denote a set of time series with $\mathbf{x}_t \in \mathbb{R}^D$, where $ T $ is the time steps and $ D $ is the number of channels. The label vector $\mathbf{y}=\left \{ y_1,\cdots, y_T\right \} $ indicates whether the corresponding time stamp is normal ($y_t=0$) or abnormal ($y_t=1$). Our task is to determine anomaly labels for all time points $ \hat{\mathbf{y} } = \left \{ \hat{y}_1, \cdots, \hat{y}_t\right \} $, where $ \hat{y}_t\in \left \{ 0,1 \right \} $, to match the ground truth $\mathbf{y}$ as much as possible. Following the practice of ~\cite{anomalytrans,evaluation,tranad2022,nominality}, we mainly focus on the F1 score with point adjustment. 

\subsection{Sub-Adjacent Neighborhoods}

Time points usually have stronger connections with their neighbors and fewer connections with distant points. This characteristic is more pronounced for anomalies ~\cite{anomalytrans}. As shown in Figure \ref{fig1}, if we rely solely on sub-adjacent neighborhoods to reconstruct time points, the reconstruction errors of anomalies will become more pronounced, thereby enhancing their distinguishability. This is the core idea of the Sub-Adjacent Transformer.

In this paper, we define the {\em sub-adjacent neighborhoods} as the region where the distance to the target point is between K1 and K2.  $K_1$ and $K_1$ are the pre-defined area bounds and satisfy $K_2\ge K_1 > 0$. The highlighted red area in Figure \ref{fig1} (b) show the sub-adjacent neighborhoods of the point marked with the red circle. 

We now apply our thought to the attention matrix. The sub-adjacent neighborhoods are represented by the highlighted stripes in the attention matrix, as depicted in Figure \ref{fig2}. To focus attention on these stripes, we introduce the concept of {\em attention contribution}. This concept involves viewing the columns of the attention matrix as each point's contribution to others within the same window. Let $\mathbf{A}_{ij}$ denote the element in the $i$-th row and $j$-th column of the attention matrix $\mathbf{A}$. The value of $\mathbf{A}_{ij}$ reflects the extent of point $i$'s contribution to point $j$; the larger $\mathbf{A}_{ij}$ is, the more significant the contribution. We define the sub-adjacent attention contribution of each point as the aggregate of values within the pre-defined sub-adjacent span in the corresponding column, which is 

\begin{equation}
\begin{aligned}
    & \mathrm{SACon}\left ( \mathbf{A}  \right ) =\left [ \mathrm{SACon}\left ( \mathbf{A}_{:,i} \right )  \right ]_{i=0,\cdots,\mathrm{win\_size-1}}  \\
    & \mathrm{SACon}\left ( \mathbf{A}_{:,i} \right ) = {\textstyle \sum_{\left | j-i \right |\ge K_1 }^{\left | j-i \right |\le K_2}}\mathbf{A} _{ji}, 0\le j<win\_size
\end{aligned}
\label{SACon}
\end{equation}


\begin{figure}[t]
   \centering
   \includegraphics[width=0.8\linewidth]{figures/SubAdjacent_mat_ill.pdf}
   \caption{Illustration of attention contribution and the desired attention matrix. For clearness, only the main stripes are depicted.}
   \label{fig2}
\end{figure}

\noindent where subscript $\left ( :,i \right ) $ denotes the $i$th column of the corresponding matrix, and $\mathrm{win\_size}$ is the window size. 

The sub-adjacent attention contribution plays a pivotal role in two aspects. Firstly, it steers the focus of attention towards the sub-adjacent neighborhoods by being integrated into the loss function (Eq. \ref{loss}). This is achieved by increasing $\mathrm{SACon}\left ( \mathbf{A} \right )$ for all points during the training stage. Secondly, it assists in anomaly detection. This is due to its incorporation into the anomaly score calculation, where anomalies typically show a lower attention contribution than normal points.

Moreover, the number of the highlighted cells in Figure \ref{fig2} is lower for the marginal points (where $i<K_2$ or $i>\mathrm{ win\_size}-K_2$), leading to imbalance among points. It is mainly because the $j$ in Eq. \ref{SACon} is bounded by $0\le j<win\_size$. To break through this limitation, one plausible way is to use a circular shift function to calculate $\mathbf{A}_{ji}$ in Eq. \ref{SACon}:

\begin{equation}
    \mathbf{A} _{ji}=\left \langle \left [ \mathrm{Roll}\left ( \mathbf{Q} , i-j \right )\right ]_{i,:},  \mathbf{K} _{i,:} \right \rangle  
    \label{cyclic}
\end{equation}


\noindent where  $\mathbf{Q,K}\in \mathbb{R} ^{\mathrm{win\_size}\times {\mathrm{d_{model}}}} $  are the query and key matrix, respectively, $\left \langle \cdot,\cdot  \right \rangle $ represents the inner product of two vectors, $\mathrm{Roll}\left ( \mathbf{Q}, i-j \right ) $  cyclically shift matrix $\mathbf{Q}$ by $i-j$ along the first dimension. The $j$ values in $\mathbf{A} _{ji}$ of Eq. \ref{cyclic} can satisfy the conditions $j<0$ or $j\ge win\_size$ and ensure that the number of $j$s for each $i$ is the same. Actually, the cyclic operation in Eq. \ref{cyclic} is equivalent to setting $\mathrm{SACon}\left ( \mathbf{A}_{:,i} \right ) $ in Eq. \ref{SACon} as:

\begin{equation}
    \mathrm{SACon}\left ( \mathbf{A}_{:,i} \right ) = {\textstyle \sum_{\left | j-i \right |\ge K_1 }^{\left | j-i \right |\le K_2}}\mathbf{A} _{\left [ j \right ]  i}, \left | j \right | <win\_size
    \label{eq1_2}
\end{equation}

\noindent where $\left [ j \right ]=j+n\cdot win\_size, n\in \left \{ 0,\pm 1 \right \} $ such that $0\le \left [ j \right ]<  win\_size$. In implementation, we use Eq. \ref{eq1_2} instead of Eq. \ref{cyclic} for efficiency. As shown in Figure \ref{fig3}(b), the use of Eq. \ref{eq1_2} results in two extra side stripes in the attention matrix.

\begin{figure}[t]
   \centering
   \includegraphics[width=1\linewidth]{figures/fig_3_new.pdf}
   \caption{Attention matrices obtained using (a) vanilla self-attention and (b) the linear attention with the proposed mapping function. The SMAP dataset \protect\cite{smap} and the proposed sub-adjacent neighborhoods are used.}
   \label{fig3}
\end{figure}

\begin{figure}[t]
   \centering
   \includegraphics[width=1\linewidth]{figures/linear_attention.pdf}
   \caption{Illustration of vanilla self-attention and linear attention. Without the direct application of {\tt Softmax}, the attention matrix $\Phi\left (  \mathbf{Q} \right ) \Phi\left (  \mathbf{K} \right )^T$ of linear attention usually exhibits more flexibility.}
   \label{fig4}
\end{figure}

\paragraph{Linear Attention.} Vanilla self-attention employs the {\tt Softmax} function on a row-wise basis within the attention matrix, leading to competition among values in the same row, as depicted in Figure \ref{fig3}(a). In contrast, linear attention does not face such constraints. As shown in Figure \ref{fig4}, linear attention can be expressed as $\Phi\left (  \mathbf{Q} \right ) \Phi\left (  \mathbf{K} \right )^T\mathbf{V}$, where $\Phi\left (  \cdot  \right ) $ is the mapping function. As shown in Figure \ref{fig3}, linear attention demonstrates better attention matrix shaping capabilities. Quantitative results can be found in Table \ref{mappingfunc}.

The mapping function directly impacts the focusing capability and overall performance of linear attention. Herein we introduce a novel learnable mapping function:

\begin{equation}
    \Phi \left ( \cdot \right ) = \mathrm{Softmax} _{\mathrm{row}}\left (  \cdot  / \tau \right )
    \label{mymapping}
\end{equation}

\noindent where {\tt Softmax} is applied row-wise to the input matrix, and $\tau$ is a learnable parameter to adjust dynamically the {\tt Softmax} temperature. Note that in Eq. \ref{mymapping}, the {\tt Softmax} function is applied to the matrices $\mathbf{Q}$ and $\mathbf{K}$, rather than directly to the attention matrix, as is the case with vanilla self-attention. This distinction is fundamental to the increased flexibility of the attention matrix. Moreover, we set all negative values in the matrices $\mathbf{Q}$ and $\mathbf{K}$ to a large negative number, such as $-100$, to ensure that these values are close to 0 after applying the mapping function. Various mapping functions have been explored in prior research, including the power function ~\cite{flattentrans}, column-wise {\tt Softmax} ~\cite{shen2021efficient}, ReLU ~\cite{efficientvit}, and elu function ~\cite{lineartrans2020_elu}. Our empirical experiments demonstrate the effectiveness of our mapping function in time series anomaly detection, as detailed in Table \ref{mappingfunc}. 

~\nocite{diffusion2020}

\subsection{Loss Function and Anomaly Score}

\paragraph{Loss Function.} Reconstruction loss is fundamental in unsupervised time series anomaly detection. As aforementioned, we also introduce the sub-adjacent attention contribution into the loss function, which guides the model to focus on the sub-adjacent neighborhoods. By integrating these two losses, the loss function for the input series $\mathbf{X} \in \mathbb{R}^{T\times D}$ is formulated as follows:   

\begin{equation}
    \begin{aligned}
        \mathcal{L}_{\mathrm{Total}}\left ( \mathbf{X},\hat{\mathbf{X}},\mathbf{A}\right ) & = \mathcal{L}_{\mathrm{rec} } + \uplambda \cdot \mathcal{L}_{\mathrm{attn} } \\
        & = \left \|  \mathbf{X}-\hat{\mathbf{X}}\right \|_{F }^{2}-\mathrm{\uplambda}  \cdot \left \| \mathrm{SACon}\left ( \mathbf{A} \right )   \right \|_{1}   
        \label{loss}
    \end{aligned}
\end{equation}

\noindent where $\mathcal{L}_{\mathrm{rec} }$ is the reconstruction loss and $\mathcal{L}_{\mathrm{attn} }$ is the attention loss, $\uplambda >0 $ is the weight to trade off the two terms. $\left \| \cdot  \right \|  _F$ and $\left \| \cdot  \right \|  _1$ in Eq. \ref{loss} are the Frobenius and k-norm, respectively. $\hat{\mathbf{X}} \in \mathbb{R}^{T\times D}$ denotes the reconstructed $\mathbf{X}$. 

\paragraph{Anomaly Score.} To identify anomalies, we combine the sub-adjacent attention contribution with the reconstruction errors. Consistent with with the approach in ~\cite{anomalytrans}, the {\tt Softmax} function is applied to $-\mathrm{SACon}\left ( \mathbf{A} \right )$ to highlight the anomalies with less attention contribution. Subsequently, we perform an element-wise multiplication of the attention results and reconstruction errors. Finally, the anomaly score for each point can be expressed as 

\begin{equation}
    \begin{aligned}
        \mathrm{AnomalyScore} \left ( \mathbf{X}  \right ) = & \mathrm{Softmax} \left ( -\mathrm{SACon} \left ( \mathbf{A} \right )  \right ) \\ 
        & \odot \left [ \left \|\mathbf{X}_{:,i} - \hat{\mathbf{X}}_{:,i}  \right \|_{F}^{2} \right ]_{i=1,\cdots T}    
        \label{score}
    \end{aligned}
\end{equation}

\noindent where $\odot$ is the element-wise multiplication. Typically, the anomalies exhibit less attention contribution and higher  reconstruction errors, leading to larger anomaly scores. 

\paragraph{Dynamic Gaussian Scoring.} Following the practice of ~\cite{evaluation}, we fit a dynamic Gaussian distribution to anomaly scores obtained by Eq. \ref{score} and design a score based on the fitted distribution. Let $\mu _t$ and $ \sigma  _t$ denote the dynamic mean and standard variance, respectively, which can be computed in the way of sliding windows. Then the final score with dynamic Gaussian fitting can be computed via 

\begin{equation}
    \begin{aligned}
        \mathrm{DyAnoSco} _t = -\mathrm{log}  \left ( 1-\mathrm{cdf} \left ( \frac{\mathrm{AnomalyScore} _t-\mu_t}{\sigma _{t}^{2} }  \right )  \right ) 
        \label{score2}
    \end{aligned}
\end{equation}

\noindent where $\mathrm{cdf}$ is the cumulative distribution function of the standard Gaussian distribution $N\left ( 0,1 \right ) $. 

\section{Experiments}

\subsection{Datasets}

We evaluate the Sub-Adjacent Transformer on the following datasets, whose statistics are summarized in Table \ref{dataset}.

\textbf{SWaT} (Secure Water Treatment)  ~\cite{swat} is collected continuously over 11 days from 51 sensors located at a water treatment plant. \textbf{WADI} (WAter DIstribution)  ~\cite{wadi} is acquired from 123 sensors of a reduced water distribution system for 16 days.  \textbf{PSM} (Pooled Server Metrics)  ~\cite{psm} is collected from multiple servers at eBay with 26 dimensions. \textbf{MSL} (Mars Science Laboratory rover) and \textbf{SMAP} (Soil Moisture Active Passive satellite) are datasets released by NASA with 55 and 25 dimensions respectively. \textbf{SMD} (Server Machine Dataset) ~\cite{rec_rnn_OmniAnomaly} is collected from a large compute cluster, consisting of 5 weeks of data from 28 server machines with 38 sensors. \textbf{NeurIPS-TS} (NeurIPS 2021 Time Series Benchmark) is a synthetic dataset proposed by ~\cite{nipsdataset} and includes 5 kinds of anomalies that cover point- and pattern-wise behaviors: global (point), contextual (point), shapelet (pattern), seasonal (pattern), and trend (pattern). 

\begin{table}[t]
    \centering
    \begin{tabular}{llllll}
    \toprule
    Dataset & Dims & Entities & \multicolumn{1}{c}{\begin{tabular}[c]{@{}c@{}}\#Train \\ (K)\end{tabular}} & \multicolumn{1}{c}{\begin{tabular}[c]{@{}c@{}}\#Test \\ (K)\end{tabular}} & \multicolumn{1}{c}{\begin{tabular}[c]{@{}c@{}}AR \\ (\%)\end{tabular}} \\ \midrule
SWaT    & 51   & 1        & 495                                                                        & 449                                                                       & 12.14                                                                  \\
WADI    & 123  & 1        & 1209                                                                       & 172                                                                       & 5.71                                                                   \\
PSM     & 25   & 1        & 132                                                                        & 87                                                                        & 27.76                                                                  \\
MSL     & 55   & 27       & 58                                                                         & 73                                                                        & 10.48                                                                  \\
SMAP    & 25   & 55       & 140                                                                        & 444                                                                       & 12.83                                                                  \\
SMD     & 38   & 28       & 708                                                                        & 708                                                                       & 4.16                                                                   \\
NeurIPS-TS & 1    & 1        & 20                                                                         & 20                                                                        & 22.44       \\ \bottomrule
    \end{tabular}
    \caption{Datasets used in this study. AR is short for the anomaly rate. \#Train and \#Test denote the number of the training and test time points, respectively.}
    \label{dataset}
\end{table}

\subsection{Implementation Details}

\begin{table*}[t]
\begin{center}
\renewcommand\arraystretch{1}
\begin{tabular}{c|cccccc|cccccc}
\toprule
Category                                                       & \multicolumn{6}{c|}{Single-Entity}                                                                                                      & \multicolumn{6}{c}{Multi-Entity$\dag$}                                                                                                        \\ \midrule
Datset                                                         & \multicolumn{2}{c}{SWaT}                           & \multicolumn{2}{c}{WADI}                           & \multicolumn{2}{c|}{PSM}      & \multicolumn{2}{c}{MSL}                            & \multicolumn{2}{c}{SMAP}                           & \multicolumn{2}{c}{SMD}       \\ \midrule
Metric                                                         & AUC           & \multicolumn{1}{c|}{F1}            & AUC           & \multicolumn{1}{c|}{F1}            & AUC           & F1            & AUC           & \multicolumn{1}{c|}{F1}            & AUC           & \multicolumn{1}{c|}{F1}            & AUC           & F1            \\ \midrule 
DAGMM~\shortcite{dagmm}                                                         & 91.6          & \multicolumn{1}{c|}{85.3}          & 29.2          & \multicolumn{1}{c|}{20.9}          & 85.2          & 76.1          & 80.6          & \multicolumn{1}{c|}{70.1}          & 81.5          & \multicolumn{1}{c|}{71.2}          & 82.3          & 72.3          \\
LSTM-VAE~\shortcite{lstm_vae_rec}                                                     & 88.3          & \multicolumn{1}{c|}{80.5}          & 49.8          & \multicolumn{1}{c|}{38.0}          & 88.6          & 80.9          & 91.6          & \multicolumn{1}{c|}{85.4}          & 84.8          & \multicolumn{1}{c|}{75.6}          & 88.6          & 80.8          \\
MSCRED ~\shortcite{MSCRED}                                                        & 88.5          & \multicolumn{1}{c|}{80.7}          & 49.1          & \multicolumn{1}{c|}{37.4}          & 74.3          & 62.6          & 96.6          & \multicolumn{1}{c|}{93.6}          & 92.4          & \multicolumn{1}{c|}{86.6}          & 90.8          & 84.1          \\
OmniAnomaly ~\shortcite{rec_rnn_OmniAnomaly}                                                   & 92.4          & \multicolumn{1}{c|}{86.6}          & 53.9          & \multicolumn{1}{c|}{41.7}          & 77.6          & 66.4          & 94.6          & \multicolumn{1}{c|}{90.1}          & 91.6          & \multicolumn{1}{c|}{85.4}          & 98.0          & 96.2          \\
MAD-GAN  ~\shortcite{mad-gan}                                                      & 89.0          & \multicolumn{1}{c|}{81.5}          & 67.9          & \multicolumn{1}{c|}{55.6}          & 77.1          & 65.8          & 95.5          & \multicolumn{1}{c|}{91.7}          & 92.3          & \multicolumn{1}{c|}{86.5}          & 95.4          & 91.5          \\
MTAD-GAT~\shortcite{gnn2020}                                                    & 92.0          & \multicolumn{1}{c|}{86.0}          & 72.2          & \multicolumn{1}{c|}{60.2}          & 86.6          & 78.0          & 95.0          & \multicolumn{1}{c|}{90.8}          & 94.6          & \multicolumn{1}{c|}{90.1}          & 95.0          & 90.8          \\
USAD~\shortcite{audibert2020usad}                                                           & 91.1          & \multicolumn{1}{c|}{84.6}          & 55.3          & \multicolumn{1}{c|}{43.0}          & 82.5          & 72.5          & 95.2          & \multicolumn{1}{c|}{91.1}          & 89.3          & \multicolumn{1}{c|}{81.9}          & 97.2          & 94.6          \\
THOC~\shortcite{thoc}                                                    & 93.3          & \multicolumn{1}{c|}{88.1}          & 63.1          & \multicolumn{1}{c|}{50.6}          & 94.2          & 89.5          & 96.7          & \multicolumn{1}{c|}{93.7}          & 97.5          & \multicolumn{1}{c|}{95.2}          & 66.5          & 54.1          \\
UAE~\shortcite{evaluation_ieee_2022}                                                            & 92.6          & \multicolumn{1}{c|}{86.9}          & 97.8          & \multicolumn{1}{c|}{95.7}          & 96.6          & 93.6          & 95.7          & \multicolumn{1}{c|}{92.0}          & 94.3          & \multicolumn{1}{c|}{89.6}          & {\ul 98.6}    & {\ul 97.2}    \\
GDN~\shortcite{gnn2021_pred}                                                            & 96.5          & \multicolumn{1}{c|}{93.5}          & 91.7          & \multicolumn{1}{c|}{85.5}          & 95.9          & 92.3          & 94.7          & \multicolumn{1}{c|}{90.3}          & 81.1          & \multicolumn{1}{c|}{70.8}          & 81.8          & 71.6          \\
GTA~\shortcite{gta}                                                            & 95.1          & \multicolumn{1}{c|}{91.0}          & 90.7          & \multicolumn{1}{c|}{84.0}          & 91.7          & 85.5          & 95.2          & \multicolumn{1}{c|}{91.1}          & 94.7          & \multicolumn{1}{c|}{90.4}          & 95.6          & 91.9          \\
TranAD~\shortcite{tranad2022}                                                           & 89.0          & \multicolumn{1}{c|}{81.5}          & 62.0          & \multicolumn{1}{c|}{49.5}          & 93.4          & 88.2          & 97.3          & \multicolumn{1}{c|}{94.9}          & 94.0          & \multicolumn{1}{c|}{89.2}          & 98.0          & 96.1          \\

\begin{tabular}[c]{@{}c@{}} Heuristics ~\shortcite{evaluation_ieee_2022}\end{tabular}    & {\ul 98.4}    & \multicolumn{1}{c|}{{\ul 96.9}}    & 98.2          & \multicolumn{1}{c|}{96.5}          & {\ul 99.2}    & {\ul 98.5}    & {\ul 98.2}    & \multicolumn{1}{c|}{{\ul 96.5}}    & 98.0          & \multicolumn{1}{c|}{96.1}          & 96.5          & 93.4          \\
\begin{tabular}[c]{@{}c@{}}Anomaly Transformer~\shortcite{anomalytrans}\end{tabular} & 96.9          & \multicolumn{1}{c|}{94.1}          & {\ul 98.3}    & \multicolumn{1}{c|}{{\ul 96.6}}    & 98.9          & 97.9          & 96.6          & \multicolumn{1}{c|}{93.6}          & 98.3          & \multicolumn{1}{c|}{96.7}          & 95.9          & 92.3          \\ 
NPSR~\shortcite{nominality}                                                           & 97.5          & \multicolumn{1}{c|}{95.3}          & 96.7          & \multicolumn{1}{c|}{93.8}          & 97.8          & 95.7          & 97.9          & \multicolumn{1}{c|}{96.0}          & {\ul 98.9}    & \multicolumn{1}{c|}{{\ul 97.8}}    & 91.4          & 85.0          \\
TimesNet~\shortcite{timesnet}                                                           & -          & \multicolumn{1}{c|}{92.1}          & -          & \multicolumn{1}{c|}{-}          & -          & 97.5          & -          & \multicolumn{1}{c|}{85.2}          & -    & \multicolumn{1}{c|}{71.5}    & -          & 85.8          \\
\midrule
Ours                                                           & \textbf{99.5} & \multicolumn{1}{c|}{\textbf{99.0}} & \textbf{99.7} & \multicolumn{1}{c|}{\textbf{99.3}} & \textbf{99.4} & \textbf{98.9} & \textbf{98.3} & \multicolumn{1}{c|}{\textbf{96.7}} & \textbf{99.1} & \multicolumn{1}{c|}{\textbf{98.2}} & \textbf{98.7} & \textbf{97.7} \\ \bottomrule
\end{tabular}
\caption{Quantitative results for various anomoly detection methods in the six real-world datasets. AUC means area under the ROC curve. The largest and second-largest values are highlighted with bold text and underlined text, respectively. The values in this table are as $\%$ for ease of display. \dag: For multi-entity datasets, we use a single model to train and test all entities together, posing additional challenges.}
\label{mainresults}
\end{center}
\end{table*}

\begin{table*}[t]
\centering
\begin{tabular}{@{}cccccccccc@{}}
\toprule
Methods & \begin{tabular}[c]{@{}c@{}}DAGMM\\ ~\shortcite{dagmm}\end{tabular} & \begin{tabular}[c]{@{}c@{}}LSTM-VAE\\ ~\shortcite{lstm_vae_rec}\end{tabular} & \begin{tabular}[c]{@{}c@{}}OmniAnomaly\\ ~\shortcite{rec_rnn_OmniAnomaly}\end{tabular} & \begin{tabular}[c]{@{}c@{}}THOC\\ ~\shortcite{thoc}\end{tabular} 
& \begin{tabular}[c]{@{}c@{}}Heuristics \\ ~\shortcite{evaluation_ieee_2022}\end{tabular} 
& \begin{tabular}[c]{@{}c@{}}Anomaly \\Transformer\\ ~\shortcite{anomalytrans}\end{tabular} 
& \begin{tabular}[c]{@{}c@{}}NPSR\\ ~\shortcite{nominality}\end{tabular} 
& \begin{tabular}[c]{@{}c@{}}TimesNet\\ ~\shortcite{timesnet}\end{tabular} 
& Ours          \\ \midrule
AUC     & 64.3                                              & 70.5                                                 & 72.2                                                    & 74.2                                                                                        & 85.5                                                         & {\ul 86.2 }                        & 76.7  & 79.6     & \textbf{92.4} \\
F1      & 51.8                                              & 58.4                                                 & 60.2                                                    & 62.5                                                                                   & 76.3                                                        & {\ul 77.5}                                           & 65.4   & 69.5             & \textbf{86.6} \\ \bottomrule
\end{tabular}
\caption{Anomaly detection performance in the synthetic dataset NeurIPS-TS. AUC means area under the ROC curve. The largest and second-largest values are highlighted with bold text and underlined text, respectively. The values in this table are presented in percentages.}
\label{mainresults2}
\end{table*}




Following the common practice, we adopt a non-overlapping sliding window mechanism to obtain a series of sub-series. The sliding window size is set as 100 without particular statements. $K_1$ and $K_2$ are set as 20 and 30, respectively. The choices of these parameters will be discussed later in ablation studies. The points are judged to be anomalies if their anomaly scores (Eq. \ref{score2}) are larger than a certain threshold $\delta $. In this study, following the practice of paper ~\cite{nominality,anomalytrans}, the thresholds are chosen to output the best F1 scores. The widely-used point adjustment strategy ~\cite{anomalytrans,tranad2022,evaluation_ieee_2022,timesnet} is adopted.  Note that point adjustment has its practicality: one detected anomaly point will guide system administrators to identify the entire anomalous segment. The Sub-Adjacent Transformer contains 3 layers. Specifically, we set the hidden dimension $d_{\mathrm{model}}$ as 512, and the head number as 8. The hyper-parameter $\uplambda $ as 10 in Eq. \ref{loss} to balance recognition loss and attention loss. The Adam optimizer ~\cite{adam_opt} is used with an initial learning rate of $10^{-4}$. Following ~\cite{anomalytrans}, the training is early stopped within 10 epochs with the batch size of 128. Experiments are conducted using PyTorch and one NVIDIA RTX A6000 GPU.

\subsection{Main Results}

\begin{figure*}[t]
   \centering
   \includegraphics[width=1\linewidth]{figures/nips_plot_new_in_paper.pdf}
   \caption{Visualization of detection results for different anomaly categories in NeurIPS-TS benchmark. The anomalous area are highlighted with red lines/areas. The first and second row represent point and pattern anomalies, respectively. From left to right, the columns indicate raw data, recognition error (Eq. \ref{loss}), attention contribution (Eq. \ref{SACon}), anomaly score (Eq. \ref{score}) and dynamic Gaussian score (Eq. \ref{score2}).}
   \label{fig_nips}
\end{figure*}

\begin{table}[t]
\centering
\renewcommand\arraystretch{1}
\begin{tabular}{@{}l|cccccc@{}}
\toprule
 $K_1:K_2$ & SWaT                                 & WADI                                 & PSM                                  & MSL                                  & SMAP                                 & SMD                                  \\ \midrule
 $0:0$  &   94.8        &     93.5      & 95.3                                 & 91.5                                 & 92.4                                 & 93.8                                 \\ \midrule
 $0:5$  &   96.7        &     95.9      & 97.6                                 & 93.6                                 & 93.7                                 & 94.1                                 \\ \midrule
                       $0:10$  & {\color[HTML]{FE0000} {\ul 98.5}}    & {\color[HTML]{FE0000} {\ul 98.2}}    & {\color[HTML]{FE0000} 98.7}          & 94.4                                 & 96.9                                 & 95.9                                 \\
                      $10:20$  & {\color[HTML]{FE0000} 97.6}          & {\color[HTML]{FE0000} 98.1}          & 98.1                                 & {\color[HTML]{FE0000} {\ul 96.5}}    & {\color[HTML]{FE0000} {\ul 97.8}}    & 94.4                                 \\
                      $20:30$  & {\color[HTML]{FE0000} \textbf{99.0}} & {\color[HTML]{FE0000} \textbf{99.3}} & {\color[HTML]{FE0000} \textbf{98.9}} & {\color[HTML]{FE0000} \textbf{96.7}} & {\color[HTML]{FE0000} \textbf{98.2}} & {\color[HTML]{FE0000} {\ul 97.7}}    \\
 $30:40$  & {\color[HTML]{FE0000} 97.6}          & 94.8                                 & 97.8                                 & 95.6                                 & 97.5                                 & 94.0                                 \\ \midrule
                      $10:30$  & {\color[HTML]{FE0000} 97.8}          & {\color[HTML]{FE0000} 97.8}          & {\color[HTML]{FE0000} {\ul 98.8}}    & 95.0                                 & 97.4                                 & 94.5                                 \\
 $20:40$  & {\color[HTML]{FE0000} 97.9}                                 & {\color[HTML]{FE0000} 98.0}                                 & 93.5                                 & 95.7                                 & 95.6                                 & 94.4                                 \\  \midrule
  $10:40$  & {\color[HTML]{FE0000} 98.0}          & {\color[HTML]{FE0000} 97.7}          & {\color[HTML]{FE0000} 98.7}          & 96.1                                 & 97.5                                 & {\color[HTML]{FE0000} \textbf{97.8}} \\ \bottomrule
\end{tabular}
\caption{Performance comparison with different $K_1$ and $K_2$ settings for real-world datasets. The largest value for each dataset is emphasized in bold, while the second largest value is underlined. The values are marked in red if the value is not less than SOTA.}
\label{K1andK2}
\end{table}


\begin{table*}[t]
\centering
% \renewcommand\arraystretch{1.0}
\begin{tabular}{@{}llcccccc@{}}
\toprule
\multicolumn{2}{c}{Dataset}               & SWaT        & WADI          & PSM           & MSL           & SMAP          & SMD           \\ \midrule
\multicolumn{2}{l}{Vanilla self-attention ~\cite{transformer}}   & {\color[HTML]{FE0000} 98.6}        & {\color[HTML]{FE0000} 98.2}    & {\color[HTML]{FE0000} {\ul 98.8}}    & 94.1          & 97.4          & {\ul 96.5}    \\ \midrule
\multirow{5}{*}{\rotatebox[origin=c]{90}{Lin. Attn.}} & 
\begin{tabular}[c]{@{}l@{}}{\tt Softmax\_column} ~\cite{shen2021efficient} \end{tabular}  & {\color[HTML]{FE0000} 98.0}        & {\color[HTML]{FE0000} 97.5}          & 96.9          & {\ul 95.2}    & {\color[HTML]{FE0000} {\ul 97.8}}    & 95.8          \\ \cmidrule{2-8}
                        & \begin{tabular}[c]{@{}l@{}}Power function ~\cite{flattentrans} \end{tabular}   & {\color[HTML]{FE0000} 98.4}       & {\color[HTML]{FE0000} {\ul 98.3}}         & 98.4          & 93.7          & 97.5          & 95.9          \\ \cmidrule{2-8}
                        & \begin{tabular}[c]{@{}l@{}}ReLU ~\cite{efficientvit} \end{tabular}        & {\color[HTML]{FE0000}{\ul 98.8}}  & {\color[HTML]{FE0000}98.1}          & 97.6          & 94.3          & 97.5          & 94.8          \\ \cmidrule{2-8}
                        & \begin{tabular}[c]{@{}l@{}}ELU+1  ~\cite{lineartrans2020_elu} \end{tabular}
                                & 95.4        & 95.6          & 94.1          & 94.7          & 96.3          & 94.5          \\ \cmidrule{2-8}
                        & \begin{tabular}[c]{@{}l@{}}Ours  (Eq. \ref{mymapping})\end{tabular}            & {\color[HTML]{FE0000}\textbf{99.0}} & {\color[HTML]{FE0000}\textbf{99.3}} & {\color[HTML]{FE0000}\textbf{98.9}} & {\color[HTML]{FE0000}\textbf{96.7}} & {\color[HTML]{FE0000}\textbf{98.2}} & {\color[HTML]{FE0000}\textbf{97.7}} \\ \bottomrule
\end{tabular}
\caption{F1 values of the proposed method with vanilla self-attention and linear attention with various mapping functions. Bold and underlined text denote the largest and second-largest values per dataset. We mark the value in red if it is not less than SOTA.}
\label{mappingfunc}
\end{table*}

\begin{table*}[t]
\centering
\renewcommand\arraystretch{1.1}
\begin{tabular}{@{}lccc|llllll@{}}
\toprule
                                    & Baseline & \multicolumn{1}{l}{\begin{tabular}[c]{@{}l@{}}Linear \\ Attention\end{tabular}} & \multicolumn{1}{l|}{\begin{tabular}[c]{@{}l@{}}Dynamic \\ Scoring\end{tabular}} & SWaT & WADI & PSM  & MSL  & SMAP & SMD  \\ \midrule
\multirow{3}{*}{Ours}               &    \checkmark   &                  &                 & 98.2 & 98.0 & 98.7 & 93.9 & 97.0 & 96.1 \\
                                    &     \checkmark     &     \checkmark             &                 & $98.8_{\left ( +0.6 \right ) }$ & $99.1_{\left ( +1.1 \right ) } $ & $98.8_{\left ( +0.1 \right ) }$ & $96.4_{\left ( +2.5 \right ) }$ & $97.9_{\left ( +0.9 \right ) }$ & $97.0_{\left ( +0.9 \right ) }$ \\
                                    &      \checkmark & \checkmark               &     \checkmark       & $\mathbf{99.0_{\left ( +0.8 \right ) }} $ & $\mathbf{99.3_{\left ( +1.3 \right ) }} $ & $\mathbf{98.9_{\left ( +0.2 \right ) }} $ & $\mathbf{96.7_{\left ( +2.8 \right ) }} $ & $\mathbf{98.2_{\left ( +1.2 \right ) }} $ & $\mathbf{97.7_{\left ( +1.6 \right ) }} $ \\ \midrule
\multirow{3}{*}{\begin{tabular}[c]{@{}l@{}}Anomaly\\ Transformer\\~\shortcite{anomalytrans}\end{tabular}} &    \checkmark   &                  &                 & 94.1 & 96.6 & 97.9 & 93.6 & 96.7 & 92.3 \\
                                    &      \checkmark    &  \checkmark              &                 & $98.1_{\left ( +4.0 \right ) }$  & $97.6_{\left ( +1.0 \right ) }$  & $\mathbf{98.3_{\left ( +0.4 \right ) }} $ & $93.6_{\left ( +0.0 \right ) }$  & $97.6_{\left ( +0.9 \right ) }$  & $93.4_{\left ( +1.1 \right ) }$  \\
                                    &      \checkmark    &    \checkmark              &     \checkmark            &  $\mathbf{98.3_{\left ( +4.2 \right ) }} $ & $\mathbf{97.9_{\left ( +1.3 \right ) }} $ & $98.2_{\left ( +0.3 \right ) }$          & $\mathbf{93.7_{\left ( +0.1 \right ) }} $ & $\mathbf{97.7_{\left ( +1.0 \right ) }} $ & $\mathbf{93.6_{\left ( +1.3 \right ) }} $  \\ \bottomrule
\end{tabular}
\caption{Ablation on modules for the proposed Sub-Adjacent Transformer and Anomaly Transformer. The largest values for each dataset are highlighted with bold text. The gain with respect to baseline is given in the subscript.}
\label{module}
\end{table*}



\paragraph{Real-World Datasets.} In Table ~\ref{mainresults}, we report AUC and F1 metrics for various methods on six real-world datasets with 15 competitive baselines. The Sub-Adjacent Transformer consistently achieves state-of-the-art performance across all benchmarks, with an average improvement of 1.7 and 0.4 percentage points in F1 score over the current SOTA for single-entity and multi-entity datasets, respectively. Note that for multi-entity datasets, we combine all entities and train them together using a single model, whereas some methods, such as simple heuristics ~\cite{evaluation_ieee_2022} and NPSR ~\cite{nominality}, train each entity separately and average the results. Even though entity-to-entity variations are large (especially MSL and SMD datasets), the Sub-Adjacent Transformer still achieves better anomaly detection results using one model than the multi-model counterpart, showing the advantage of the proposed attention mechanism. We have retained all dimensions of the original datasets, even though some of them are constant and do not play a role. Additionally, we also report F1 results without using point-adjustment in the supplementary material, where the Sub-Adjacent Transformer also exhibits competitive performance.



\paragraph{Synthetic Dataset.} NeurIPS-TS is a challenging synthetic dataset. We generate the NeurIPS-TS dataset using the source codes released by ~\cite{nipsdataset}, and it includes 5 anomalous types covering both the point-wise (global and contextual) and pattern-wise (shapelet, seasonal and trend) anomalies. As shown in Table ~\ref{mainresults2}, the Sub-Adjacent Transformer achieves state-of-the-art performance with an astonishing 9.1 percentage points improvement of F1 score relative to Anomaly Transformer ~\cite{anomalytrans}, demonstrating the effectiveness of the proposed model on various anomalies. Visualization results can be found in Figure \ref{fig_nips}. We can see that anomalies do have less sub-adjacent contribution (the third column), which is consistent with our assumption. A full version of Figure \ref{fig_nips} can be found in the supplementary material. 

\subsection{Ablation Studies} 

\paragraph{Vanilla and Linear Transformer.} We compare vanilla self-attention and linear attention with different mapping functions using our method. As shown in Table \ref{mappingfunc}, the proposed {\tt Softmax}-based mapping function with learnable parameters outperforms other mapping functions and vanilla self-attention, with a maximum F1 score increase of 1.5 percentage points.

\paragraph{Choices of $K_1$ and $K_2$.} Choices of $K_1$ and $K_2$ directly affect the performance of the Sub-Adjacent Transformer. We evaluate the effect in Table \ref{K1andK2} with the window size being 100. The configuration of $K_1=20$ and $K_2=30$ typically yields the best performance (with the exception of the SMD dataset), while the performance of other settings is also satisfactory, with nearly half of the values exceeding the current SOTA. When the sub-adjacent span is getting close to the diagonal (the second and third rows in Table \ref{K1andK2}), the performance worsens. We also explore the extreme case where $K_1=K_2=0$. In that case, the performance drops significantly, highlighting the importance of sub-adjacent attention. 



\begin{table}[t]
\centering
\begin{tabular}{@{}c|cccccc@{}}
\toprule
W\_S & SWaT          & WADI          & PSM           & MSL           & SMAP          & SMD           \\ \midrule
50        & 97.6          & 97.8          & 98.4          & 94.7          & 96.9          & 92.1          \\
100       & {\ul 99.0}    & \textbf{99.3} & \textbf{98.9} & \textbf{96.7} & \textbf{98.2} & \textbf{97.7} \\
150       & 98.8          & {\ul 98.8}    & {\ul 98.7}    & 94.9          & 97.3          & {\ul 96.4}    \\
200       & \textbf{99.1} & 98.7          & 98.5          & 94.9          & {\ul 97.5}    & 94.2          \\
250       & 97.2          & 98.1          & 98.3          & {\ul 95.3}    & 97.4          & 93.9          \\
300       & 98.9          & 98.2          & 98.1          & 94.3          & 96.9          & 94.3          \\ \bottomrule
\end{tabular}
\caption{F1 values for different window sizes.  The largest and second-largest values are made bold and underlined, respectively.}
\label{window_size}
\end{table}

\begin{table}[htbp]
\centering
\begin{tabular}{@{}c|cccccc@{}}
\toprule
$\uplambda$ & SWaT          & WADI          & PSM           & MSL           & SMAP          & SMD           \\ \midrule
0     & 95.8    & 94.2          & 95.6    & 92.7          & 95.3          & 94.8          \\
4     & {\ul 98.8}    & 97.8          & {\ul 98.8}    & 94.7          & 97.6          & 96.7          \\
6     & 98.6          & {\ul 98.3}    & 98.6          & {\ul 96.1}    & 97.4          & {\ul 97.3}    \\
8     & 98.3          & 97.8          & 98.7          & 95.8          & {\ul 98.0}    & 96.9          \\
10    & \textbf{99.0} & \textbf{99.3} & \textbf{98.9} & \textbf{96.7} & \textbf{98.2} & \textbf{97.7} \\
12    & 98.5          & {\ul 98.3}    & 98.6          & 95.8          & 97.6          & 97.1          \\
14    & 98.7          & 98.2          & 98.6          & 95.2          & 97.3          & 96.3          \\ \bottomrule
\end{tabular}
\caption{F1 values for different $\uplambda$ settings. Bold and underlined texts denote the largest and second largest values, respectively.}
\label{lamda_tab}
\end{table}

\paragraph{Module Ablation.} We evaluate the contribution of linear attention module and dynamic scoring module in Table \ref{module}. For the Sub-Adjacent Transformer, the baseline means the proposed sub-adjacent attention mechanism accompanied with vanilla self-attention and anomaly score (Eq. \ref{score}). Then linear attention and Gaussian dynamic scoring (Eq. \ref{score2}) are introduced in turn. As shown in Table \ref{module}, compared with the baseline, linear attention brings +1.0 improvement on average over six datasets and Gaussian dynamic scoring brings another +0.2 improvement, verifying the effectiveness of the proposed modules. \\
Furthermore, we also apply the two modules to Anomaly Transformer ~\cite{anomalytrans}. As we can see from Table \ref{module}, the proposed linear attention model provides +1.2 percentage points performance gain averagely, and Gaussian dynamic scoring provides another +0.1 gain. Moreover, as shown in Table \ref{module}, without the support of linear attention and dynamic Gaussian score, our method (baseline) still outperforms the Anomaly Transformer by an average of +1.8 percentage points, thereby validating the efficacy of our sub-adjacent attention design.

\paragraph{Choices of Window Size.} \label{winsize_para} Table \ref{window_size} gives F1 values with different window sizes for six datasets. One can see that our model exhibits good robustness to the window size. The optimal performance is typically achieved with a window size of 100, the only exception being the SWaT dataset. 


\paragraph{Choices of Parameter $\uplambda$.} Table \ref{lamda_tab} compares the performance for different values of $\uplambda$. The best results are achieved when $\uplambda$ is set to 10, while the second-best results occur at different values of $\uplambda$ for different datasets. It is noteworthy that without the inclusion of attention loss ($\uplambda=0$), performance drops drastically, affirming the effectiveness of our sub-adjacent attention design.



\section{Conclusion}
In this paper, we present the Sub-Adjacent Transformer, a novel paradigm for utilizing attention in time series anomaly detection. Our method distinctively combines sub-adjacent attention contribution, linear attention and reconstruction error to effectively detect anomalies, thereby enhancing the efficacy of anomaly detection. It offers a novel perspective on the utilization of attention in this domain. Without bells and whistles, our model demonstrates superior performance across common benchmarks. We hope that the Sub-Adjacent Transformer could act as a baseline framework for the future works in time series anomaly detection. 

\section*{Acknowledgments}
This work was supported by the National Science Foundation of China (NSFC) (No. 62371009 and No. 61971008).



%% The file named.bst is a bibliography style file for BibTeX 0.99c
\bibliographystyle{named}
\bibliography{ijcai24}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\appendix

\section{Overall Architecture}

Figure \ref{overall_arch} illustrates the overall architecture of the model used in this paper. Specifically, Figure \ref{overall_arch}(c) illustrates the customized multi-head linear attention. Our main contributions lie in the sub-adjacent attention design, the introduction of linear attention with a learnable mapping function, and the corresponding loss function and anomaly score design.  

\section{Codes and Dataset Preprocessing Scripts}

It is widely recognized that the method of data preprocessing significantly influences the performance of anomaly detection. In our implementation, we use the {\tt scikit-learn} module and the {\tt StandardScaler} class to normalize the raw input data to follow a Gaussian distribution before feeding them into the network. The codes and the preprocessing script files will be released later at GitHub.

\section{F1 Without Point Adjustment}

We report the F1 scores without point adjustment in Table \ref{F1_table}. It is noticeable that our approach maintains satisfactory performance even in the absence of point adjustment. Technically, dynamic Gaussian scoring and the {\tt Softmax} operation in anomaly score are turned off, because these operations could cause the masking effect of strong values over the surrounding weak values. The reconstruction error is used as anomaly score while the loss function in the training phase is unchanged. The proposed Sub-Adjacent Transformer achieves state-of-the-art (SOTA) results in 2 out of 6 datasets and demonstrates performance closely approaching SOTA in the remaining cases, underscoring its robustness and efficacy.

\section{Full Version of Visualization}



NeurIPS-TS is a challenging synthetic dataset. Figure \ref{nips_visual} provides a visualization of intermediate variables that we discuss in Section 3 (Method). We can see that the Sub-adjacent Transformer can handle these 5 anomalous types well. Notably, in the third row, the attention contribution of anomalies is lower than the normal points, which is consistent with our assumption. 




\section{Training Set Size}

Table \ref{trainset_ratio} reports the effects of the training set size. Training set ratio ranges from 10\% to 100\%. We can observe that the anomaly detection performance is positively correlated with the training set size and that the impact of training set size varies depending on the dataset. Particularly, for the PSM dataset, the difference between the F1 scores at 10\% and 100\% is only 1.8 percentage points. Generally speaking, our method is robust to the training set size.

\begin{table}[ht]
\centering
\renewcommand\arraystretch{1.0}
\begin{tabular}{@{}ccccccc@{}}
\toprule
T. R. & SWaT          & WADI          & PSM           & MSL           & SMAP          & SMD           \\ \midrule
10\%                                                      & 94.6          & 88.8          & 97.1          & 88.3          & 96.2          & 96.2          \\
20\%                                                      & 95.4          & 95.8          & 98.5          & 92.2          & 97.0          & 96.2          \\
40\%                                                      & 98.3          & 96.9          & 98.6          & 95.4          & 97.0          & 97.2          \\
60\%                                                      & 98.3          & 97.2          & 98.6          & 95.9          & 97.2          & 97.4          \\
80\%                                                      & {\ul 98.5}    & {\ul 98.3}    & {\ul 98.7}    & {\ul 96.2}    & {\ul 97.7}    & {\ul 97.5}    \\
100\%                                                     & \textbf{99.0} & \textbf{99.3} & \textbf{98.9} & \textbf{96.7} & \textbf{98.2} & \textbf{97.3} \\ \bottomrule
\end{tabular}
\caption{Change of F1 score with training set size. T.R. stands for the training set ratio that is actually utilized. As training data increases, the detection performance tends to increase. The effects of the training set ratio vary for different datasets due to the diverse scale of the original training sets.}
\label{trainset_ratio}
\end{table}


\section{Resource Footprint}



Table \ref{gpu_state} presents training and inference statistics for the Sub-Adjacent Transformer and its variants across six real-world datasets. The Sub-Adjacent Transformer demonstrates notable efficiency in both training and inference phases, and it maintains efficient GPU memory utilization. However, methods based on linear attention do not demonstrate significant advantages over the traditional self-attention in these respects. This phenomenon is primarily attributed to the utilization of a small window size and the implicit computation of the attention matrix. While the explicit computation of the attention matrix for small window sizes leads to enhanced efficiency, as indicated by the bracketed values in Table \ref{gpu_state}, this approach may become impractical for larger window sizes due to the resultant increase in computational complexity and memory requirements. In general, without specific statements, implicit computation is employed to align with the conventional practices of linear attention.





\begin{figure*}[t]
   \centering
   \includegraphics[width=1\linewidth]{figures/architecture.pdf}
   \caption{Network architecture for the Sub-Adjacent Transformer.}
   \label{overall_arch}
\end{figure*}

\begin{table*}[t]
\centering
\begin{tabular}{@{}lcccccc@{}}
\toprule
Methods             & SWaT          & WADI          & PSM           & MSL           & SMAP          & SMD           \\ \midrule
DAGMM               & 75.0          & 12.1          & 48.3          & 19.9          & 33.3          & 23.8          \\
LSTM-VAE            & 77.6          & 22.7          & 45.5          & 21.2          & 23.5          & 43.5          \\
MSCRED              & 75.7          & 4.6           & 55.6          & 25.0          & 17.0          & 38.2          \\
OmniAnomaly         & 78.2          & 22.3          & 45.2          & 20.7          & 22.7          & 47.4          \\
MAD-GAN             & 77.0          & 37.0          & 47.1          & 26.7          & 17.5          & 22.0          \\
MTAD-GAT            & 78.4          & 43.7          & 57.1          & 27.5          & 29.6          & 40.0          \\
USAD                & 79.2          & 23.3          & 47.9          & 21.1          & 22.8          & 42.6          \\
THOC                & 61.2          & 13.0          & -           & 19.0          & 24.0          & 16.8          \\
UAE                 & 45.3          & 35.4          & 42.7          & 45.1          & 39.0          & 43.5          \\
GDN                 & 81.0          & 57.0          & 55.2          & 21.7          & 25.2          & {\ul 52.9}    \\
GTA                 & 76.1          & 53.1          & 54.2          & 21.8          & 23.1          & 35.1          \\
Heuristics   & 78.9          & 35.3          & 50.9          & 23.9          & 22.9          & 49.4          \\
Anomaly Transformer & 22.0          & 10.8          & 43.4          & 19.1          & 22.7          & 8.0           \\
TranAD              & 66.9          & 41.5          & {\ul 64.9}    & 25.1          & 24.7          & 31.0          \\
NPSR                & {\ul 83.9}    & \textbf{64.2} & 64.8          & \textbf{55.1} & \textbf{51.1} & \textbf{53.5} \\ \midrule
Ours                & \textbf{84.2} & {\ul 63.5}    & \textbf{65.3} & {\ul 50.3}    & {\ul 45.2}    & 50.6          \\ \bottomrule
\end{tabular}
\caption{F1 score without point adjustment on real-world datasets. Bold text represents the best result, while underlined text represents the second best result. All channels of all the datasets are used. In our implementation, for multi-entity datasets (MSL, SMAP and SMD) we train one model for each entity and average the results. }
\label{F1_table}
\end{table*}

\begin{figure*}[t]
   \centering
   \includegraphics[width=1\linewidth]{figures/nips_plot_all.pdf}
   \caption{Visualization for different anomaly categories in the NeurIPS-TS dataset. Red lines or highlighted areas are used to mark the anomalous regions. From top to bottom, the rows are raw data, recognition loss, sub-adjacent attention contribution, anomaly score and Gaussian dynamic score, respectively. From left to right, the columns indicate global and contextual point anomalies, shapelet, seasonal and trend pattern anomalies, respectively.}
   \label{nips_visual}
\end{figure*}

\begin{table*}[t]
\centering
\renewcommand\arraystretch{1.0}
\begin{tabular}{@{}llllllll@{}}
\toprule
\multirow{2}{*}{Dataset} & \multirow{2}{*}{Metric} & \multirow{2}{*}{\begin{tabular}[c]{@{}l@{}}Vanilla \\ Attention\end{tabular}} & \multicolumn{5}{c}{Linear Attention}              \\ \cmidrule(l){4-8} 
                         &                         &                                                                               & Sofmax\_col. & Pow. Func. & ReLU $\quad$  & ELU+1 $\quad$ & Ours $\quad$ \\ \midrule
\multirow{5}{*}{SWaT}    & Train. Time/Epoch (s)   & 20.6                                                                          & 34.3(30.4)   & 38.7(34.5)  & 33.6(29.3) & 34.1(30.2) & 36.7(33.1)  \\
                         & Infer. Time (s)         & 0.094                                                                         & 0.112        & 0.143       & 0.111      & 0.112      & 0.13        \\
                         & GPU Memory (G)          & 1.4                                                                           & 3.1(1.7)     & 3.6(2.2)    & 3.1(1.7)   & 3.3(1.9)   & 3.3(1.9)    \\
                         & FLOPS (M)               & 483                                                                           & 484          & 484         & 484        & 484        & 484         \\
                         & \# Params. (M)          & 4.84                                                                          & 4.84         & 4.84        & 4.84       & 4.84       & 4.84        \\ \midrule
\multirow{5}{*}{WADI}    & Train. Time/Epoch (s)   & 33.5                                                                          & 54.9(48.8)   & 61.6(55.0)  & 53.8(47.0) & 54.5(48.4) & 58.9(52.9)  \\
                         & Infer. Time (s)         & 0.047                                                                         & 0.053        & 0.065       & 0.055      & 0.054      & 0.057       \\
                         & GPU Memory (G)          & 1.5                                                                           & 3.2(1.7)     & 3.6(2.2)    & 3.2(1.7)   & 3.3(1.9)   & 3.3(1.9)    \\
                         & FLOPS (M)               & 498                                                                           & 499          & 498         & 498        & 498        & 499         \\
                         & \# Params. (M)          & 4.98                                                                          & 4.98         & 4.98        & 4.98       & 4.98       & 4.98        \\ \midrule
\multirow{5}{*}{PSM}     & Train. Time/Epoch (s)   & 55.5                                                                          & 91.4(80.8)   & 102.2(91.1) & 89.3(77.7) & 90.6(80.2) & 98.3(87.8)  \\
                         & Infer. Time (s)         & 0.018                                                                         & 0.022        & 0.026       & 0.021      & 0.021      & 0.024       \\
                         & GPU Memory (G)          & 1.45                                                                          & 3.1(1.7)     & 3.6(2.1)    & 3.2(1.7)   & 3.3(1.8)   & 3.3(1.9)    \\
                         & FLOPS (M)               & 478                                                                           & 478          & 478         & 478        & 478        & 479         \\
                         & \# Params. (M)          & 4.78                                                                          & 4.78         & 4.78        & 4.78       & 4.78       & 4.78        \\ \midrule
\multirow{5}{*}{MSL}     & Training Time (s)       & 24                                                                            & 40.2(35.8)   & 45.4(40.5)  & 39.5(34.5) & 40.1(35.6) & 42.9(38.9)  \\
                         & Infer. Time (s)         & 0.017                                                                         & 0.02         & 0.025       & 0.02       & 0.019      & 0.022       \\
                         & GPU Memory (G)          & 1.46                                                                          & 3.1(1.7)     & 3.6(2.2)    & 3.1(1.7)   & 3.3(1.8)   & 3.3(1.9)    \\
                         & FLOPS (M)               & 484                                                                           & 485          & 484         & 484        & 484        & 485         \\
                         & \# Params. (M)          & 4.84                                                                          & 4.84         & 4.84        & 4.84       & 4.84       & 4.84        \\ \midrule
\multirow{5}{*}{SMAP}    & Train. Time/Epoch (s)   & 58.1                                                                          & 96.9(85.9)   & 108.6(96.8) & 94.9(82.6) & 96.2(85.2) & 103.9(93.1) \\
                         & Infer. Time (s)         & 0.083                                                                         & 0.096        & 0.126       & 0.101      & 0.098      & 0.112       \\
                         & GPU Memory (G)          & 1.4                                                                           & 3.1(1.7)     & 3.6(2.1)    & 3.1(1.7)   & 3.2(1.8)   & 3.3(1.9)    \\
                         & FLOPS (M)               & 478                                                                           & 478          & 478         & 478        & 478        & 479         \\
                         & \# Params. (M)          & 4.78                                                                          & 4.78         & 4.78        & 4.78       & 4.78       & 4.78        \\ \midrule
\multirow{5}{*}{SMD}     & Train. Time/Epoch (s)   & 29.7                                                                          & 48.9(43.4)   & 55.1(49.1)  & 47.9(41.7) & 48.6(43.0) & 52.5(47.1)  \\
                         & Infer. Time (s)         & 0.132                                                                         & 0.157        & 0.192       & 0.158      & 0.152      & 0.184       \\
                         & GPU Memory (G)          & 1.4                                                                           & 3.1(1.7)     & 3.6(2.2)    & 3.1(1.7)   & 3.2(1.8)   & 3.3(1.9)    \\
                         & FLOPS (M)               & 481                                                                           & 481          & 481         & 481        & 481        & 482         \\
                         & \# Params. (M)          & 4.81                                                                          & 4.81         & 4.81        & 4.81       & 4.81       & 4.81        \\ \bottomrule
\end{tabular}
\caption{Training and inference statistics for our model and its variants. FLOPS and parameter numbers are obtained using the {\tt thop} module. GPU memory indicates the GPU usage during training. The values inside the brackets are obtained by explicitly calculating the attention matrix, which improves training efficiency and GPU utilization over the implicit counterpart. }
\label{gpu_state}
\end{table*}


\end{document}

