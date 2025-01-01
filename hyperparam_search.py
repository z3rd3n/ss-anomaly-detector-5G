# hyperparam_search.py

import optuna
import logging
from utils import start_logging

# For Optuna plotting
import optuna.visualization.matplotlib as optuna_plot
import matplotlib.pyplot as plt
import os

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
    config.num_epochs = 2  # Fewer epochs for faster search

    # 2. Suggest hyperparameters
    # Here are examples; you can add or remove based on your needs:
    #config.learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    #config.dropout       = trial.suggest_float("dropout", 0.0, 0.2, step=0.05)
    #config.model_dim     = trial.suggest_categorical("model_dim", [32, 64, 128, 256])
    #config.n_heads       = trial.suggest_categorical("n_heads", [2, 4, 8])
    #config.e_layers      = trial.suggest_int("e_layers", 2, 8)
    #config.k_value       = trial.suggest_float("k_value", 0.01, 2.0, log=True)
    config.one_side      = trial.suggest_categorical("one_side", [True, False])
    #config.batch_size    = trial.suggest_categorical("batch_size", [32, 64, 128])
    config.seq_len       = trial.suggest_categorical("seq_len", [16, 32, 50, 64, 100])
    #config.activation    = trial.suggest_categorical("activation", ['relu', 'gelu'])


    span_ratio_pairs = {
    "low_1/16_high_1/8": (1.0/16, 1.0/8),
    "low_1/8_high_1/4": (1.0/8, 1.0/4),
    "low_1/4_high_1/2": (1.0/4, 1.0/2),
    "low_1/2_high_3/4": (1.0/2, 3.0/4),
    "low_3/4_high_7/8": (3.0/4, 7.0/8),
    "low_1/8_high_3/8": (1.0/8, 3.0/8),
    # etc. add more if you want
    }

    chosen_key = trial.suggest_categorical("span_ratio_key", list(span_ratio_pairs.keys()))
    low_ratio, high_ratio = span_ratio_pairs[chosen_key]

    # 3. Compute the integer spans
    span_low  = int(config.seq_len * low_ratio)
    span_high = int(config.seq_len * high_ratio)

    # 4. Ensure they are at least 1 and distinct
    span_low  = max(1, span_low)
    span_high = max(span_low + 1, span_high)

    config.span = (span_low, span_high)

    
    #optimizer_name       = trial.suggest_categorical("optimizer_name", ["Adam", "AdamW", "SGD"])
    #weight_decay         = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    train_dataset, val_dataset = ParquetSequenceDataset.create_train_val_splits(
        parquet_path=config.parquet_path,
        feature_columns=config.feature_columns,
        seq_len=config.seq_len,
        stride=config.stride,  # incorporate the chosen stride
        validation_ratio=config.validation_ratio,
        seed=config.seed
    )


    if len(train_dataset.file_ids) > 10:
        train_dataset.file_ids = train_dataset.file_ids[:25]
        train_dataset.num_files = len(train_dataset.file_ids)

    if len(val_dataset.file_ids) > 2:
        val_dataset.file_ids = val_dataset.file_ids[:5]
        val_dataset.num_files = len(val_dataset.file_ids)

    # 5. Create DataLoaders
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

    # 6. Build model
    model = config.build_model()

    # 7. Choose optimizer based on the suggestion
    if config.optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:  # SGD
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)

    # 8. Optionally set up a scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    # 9. Train the model (returns the final val loss)
    best_val_loss = run_training_for_trial(config, model, optimizer, scheduler, train_loader, val_loader)
    return best_val_loss


def run_training_for_trial(config, model, optimizer, scheduler, train_loader, val_loader):
    """
    A simplified version of your training loop that returns the final validation loss
    for this trial. We replicate the logic from train_model but
    store/return the best validation loss for Optuna.
    """
    from subAdjacent.run_epoch import train_one_epoch, validate_one_epoch
    import torch

    criterion_mse = torch.nn.MSELoss(reduction='none')
    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        # Train
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

        # Validate
        val_rec_loss, val_total_loss = validate_one_epoch(
            model,
            val_loader,
            config.device,
            criterion_mse,
            config.span,
            config.one_side,
            lambda_sacon=config.k_value
        )

        # Track best
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss


    return best_val_loss


def main():
    start_logging()
    optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize')  # Minimizing validation loss
    study.optimize(
        objective,
        n_trials=25,
        show_progress_bar=True,
    )

    logging.info("Hyperparameter search complete.")
    logging.info(f"Best trial value (val_loss): {study.best_trial.value}")
    logging.info(f"Best trial params: {study.best_trial.params}")

    plot_dir = "optuna_plots"
    os.makedirs(plot_dir, exist_ok=True)

    # 2) Plot the optimization history
    fig1 = optuna_plot.plot_optimization_history(study)
    fig1.savefig(os.path.join(plot_dir, "optimization_history.png"))
    plt.close(fig1)

    # 3) Plot the parallel coordinate plot
    fig2 = optuna_plot.plot_parallel_coordinate(study)
    fig2.savefig(os.path.join(plot_dir, "parallel_coordinate.png"))
    plt.close(fig2)

    # 4) Plot the hyperparameter slice plots
    fig3 = optuna_plot.plot_slice(study)
    fig3.savefig(os.path.join(plot_dir, "slice_plot.png"))
    plt.close(fig3)

    # 5) Plot the parameter importances
    try:
        fig4 = optuna_plot.plot_param_importances(study)
        fig4.savefig(os.path.join(plot_dir, "param_importances.png"))
        plt.close(fig4)
    except ValueError:
        # Sometimes param importances can't be computed if we have categorical params only, etc.
        logging.warning("Could not plot param importances (possibly all categorical params).")

    # 6) Plot contour (shows 2D relationships among hyperparameters)
    fig5 = optuna_plot.plot_contour(study)
    fig5.savefig(os.path.join(plot_dir, "contour_plot.png"))
    plt.close(fig5)

    logging.info(f"Optuna plots saved to '{plot_dir}' directory.")


if __name__ == "__main__":
    main()