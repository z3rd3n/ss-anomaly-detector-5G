# hyperparam_search.py

import optuna
import logging
from utils import start_logging

# Import your existing code
from subAdjacent.configClass import Config
from data.dataLoader import ParquetSequenceDataset, custom_collate_fn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def objective(trial: optuna.trial.Trial) -> float:
    """
    Objective function that Optuna will call multiple times.
    It will set different hyperparams on the Config, run training on
    a small subset of data, and return the validation loss.
    """
    # 1. Create a fresh config
    config = Config()
    config.train = True  # We do want to train in these trials
    config.num_epochs = 10  # Increased epochs for better convergence

    # 2. Suggest hyperparameters
    config.learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    config.dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
    config.model_dim = trial.suggest_categorical("model_dim", [64, 128, 256, 512, 1024])
    config.n_heads = trial.suggest_categorical("n_heads", [2, 4, 8, 16])
    config.e_layers = trial.suggest_int("e_layers", 1, 12)
    config.k_value = trial.suggest_float("k_value", 0.1, 10.0, log=True)
    config.one_side = trial.suggest_categorical("one_side", [True, False])
    config.seq_len = trial.suggest_int("seq_len", 20, 200, step=10)
    config.activation = trial.suggest_categorical("activation", ['relu', 'gelu', 'tanh'])

    stride_ratio_pairs = {
        "1/8": 1.0/8,
        "1/4": 1.0/4,
        "1/2": 1.0/2,
        "3/4": 3.0/4,
        "full": 1.0  # Non-overlapping
    }
    chosen_stride_key = trial.suggest_categorical("stride_ratio_key", list(stride_ratio_pairs.keys()))
    stride_ratio = stride_ratio_pairs[chosen_stride_key]
    config.stride = max(1, int(config.seq_len * stride_ratio))

    span_ratio_pairs = {
        "low_1/16_high_1/8": (1.0/16, 1.0/8),
        "low_1/8_high_1/4": (1.0/8, 1.0/4),
        "low_1/4_high_1/2": (1.0/4, 1.0/2),
        "low_1/2_high_3/4": (1.0/2, 3.0/4),
        "low_3/4_high_7/8": (3.0/4, 7.0/8),
        "low_1/8_high_3/8": (1.0/8, 3.0/8),
    }
    chosen_key = trial.suggest_categorical("span_ratio_key", list(span_ratio_pairs.keys()))
    low_ratio, high_ratio = span_ratio_pairs[chosen_key]

    span_low = int(config.seq_len * low_ratio)
    span_high = int(config.seq_len * high_ratio)
    span_low = max(1, span_low)
    span_high = max(span_low + 1, span_high)
    config.span = (span_low, span_high)

    config.optimizer_name = trial.suggest_categorical("optimizer_name", ["Adam", "AdamW", "SGD"])
    config.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    config.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    train_dataset, val_dataset = ParquetSequenceDataset.create_train_val_splits(
        parquet_path=config.parquet_path,
        feature_columns=config.feature_columns,
        seq_len=config.seq_len,
        stride=config.stride,
        validation_ratio=config.validation_ratio,
        seed=config.seed
    )

    # Optionally limit the dataset size for faster trials
    # Remove or adjust if computational resources allow
    if len(train_dataset.file_ids) > 50:
        train_dataset.file_ids = train_dataset.file_ids[:30]
        train_dataset.num_files = len(train_dataset.file_ids)

    if len(val_dataset.file_ids) > 10:
        val_dataset.file_ids = val_dataset.file_ids[:10]
        val_dataset.num_files = len(val_dataset.file_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=config.pin_memory,
        drop_last=True
    )

    model = config.build_model()

    if config.optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:  # SGD
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_val_loss = run_training_for_trial(config, model, optimizer, scheduler, train_loader, val_loader)
    return best_val_loss

def run_training_for_trial(config, model, optimizer, scheduler, train_loader, val_loader):
    from subAdjacent.run_epoch import train_one_epoch, validate_one_epoch
    import torch

    criterion_mse = torch.nn.MSELoss(reduction='none')
    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            config.device,
            criterion_mse,
            config.span,
            config.one_side,
            lambda_sacon=config.k_value,
            max_grad_norm=config.max_grad_norm
        )

        val_rec_loss, val_total_loss = validate_one_epoch(
            model,
            val_loader,
            config.device,
            criterion_mse,
            config.span,
            config.one_side,
            lambda_sacon=config.k_value
        )

        #scheduler.step(val_total_loss)

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss

        # Early stopping logic (optional)
        if epoch >= 5 and val_total_loss > best_val_loss - 1e-4:
            break

    return best_val_loss

def main():
    start_logging()
    
    study_name = "optimization_study"
    storage_name = "sqlite:///optuna_study.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='minimize',
        load_if_exists=True
    )
    
    study.optimize(
        objective,
        n_trials=30,  # Increased number of trials
        show_progress_bar=True,
        n_jobs=2  # Parallel execution if possible
    )
    
    logging.info("\n=== Optuna Dashboard Instructions ===")
    logging.info("To view the dashboard, run the following command in your terminal:")
    logging.info("optuna-dashboard sqlite:///optuna_study.db")
    logging.info("Then open http://127.0.0.1:8080 in your browser")

if __name__ == "__main__":
    main()