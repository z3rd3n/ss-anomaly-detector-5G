#!/usr/bin/env python3
# main.py
from utils import start_logging, bring_approach, count_trainable_parameters
from data.dataLoader import ParquetSequenceDataset, custom_collate_fn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import logging

def main(params, train_func, detect_func):    
    logging.info("Building model...")
    model = params.build_model()
    trainable_params = count_trainable_parameters(model)
    logging.info(f'Number of trainable parameters: {trainable_params/1000:.1f}K')

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
    if params.optimizer_name == 'Adam':
        optimizer = optim.AdamW(model.parameters(), lr=params.learning_rate)
    if params.optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

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
    if params.detect:
        logging.info("Starting detection...")
        anomalies_df, fig_path = detect_func(
            params, 
            model, 
            val_loader, 
        )
        anomalies_df.to_csv(f"{params.output_dir}_p{params.p}q{str(params.q)[-2:]}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main script with specified approach.")
    parser.add_argument('--approach', type=str, default='subAdjacent', help='Specify the approach to use.')
    args = parser.parse_args()
        

    config, train_func, detect_func = bring_approach(args)
    start_logging(config, args.approach)
    main(config, train_func, detect_func)