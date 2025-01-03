import optuna
import logging
from utils import start_logging
from typing import Dict, Any
import torch
from approaches.subAdjacent.run_epoch import train_one_epoch, validate_one_epoch
from data.dataLoader import ParquetSequenceDataset, custom_collate_fn
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from approaches.subAdjacent.configClass import Config
import os
from datetime import datetime

def log_epoch_results(epoch: int, train_loss: float, val_rec_loss: float, val_total_loss: float):
    """Log epoch-wise results"""
    logging.info(f"\nEpoch {epoch} Results:")
    logging.info(f"Training Loss: {train_loss:.6f}")
    logging.info(f"Validation Reconstruction Loss: {val_rec_loss:.6f}")
    logging.info(f"Validation Total Loss: {val_total_loss:.6f}")

def log_trial_start(trial_number: int, params: Dict[str, Any]):
    """Log trial start with all trial parameters"""
    logging.info(f"\n=== Starting Trial {trial_number} ===")
    logging.info("Trial parameters:")
    for key, value in params.items():
        logging.info(f"{key}: {value}")

def log_config_details(config: Config):
    """Log all configuration parameters"""
    logging.info("\n=== Configuration Details ===")
    for key, value in vars(config).items():
        logging.info(f"{key}: {value}")

def create_search_space(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """
    Define the hyperparameter search space separately for better organization
    and visualization in Optuna Dashboard.
    """
    # Architecture parameters
    params = {
        "model_dim": trial.suggest_categorical("architecture/model_dim", [64, 128, 256, 512]),
        "n_heads": trial.suggest_categorical("architecture/n_heads", [2, 4, 8]),
        "e_layers": trial.suggest_int("architecture/e_layers", 1, 4),
        "dropout": trial.suggest_float("architecture/dropout", 0.0, 0.5, step=0.1),
        
        # Sequence parameters
        "seq_len": trial.suggest_categorical("sequence/seq_len", [50, 100, 200]),
        "one_side": trial.suggest_categorical("sequence/one_side", [True, False]),
        "negative_qk": trial.suggest_categorical("sequence/negative_qk", [True, False]),
        
        # Span ratio parameters
        "span_ratio_key": trial.suggest_categorical("span/ratio_key", [
            "low_1/16_high_1/8",
            "low_1/8_high_1/4",
            "low_1/4_high_1/2",
            "low_1/2_high_3/4",
            "low_3/4_high_7/8",
            "low_1/8_high_3/8"
        ])
    }
    return params

def objective(trial: optuna.trial.Trial) -> float:
    """
    Objective function with improved logging and trial management.
    """
    torch.cuda.empty_cache()
    config = Config()
    config.train = True
    config.num_epochs = 3

    # Get hyperparameters from search space
    params = create_search_space(trial)

    log_trial_start(trial.number, params)
    
    # Update config with suggested parameters
    for key, value in params.items():
        if key != "span_ratio_key":  # Handle span ratio separately
            setattr(config, key, value)
    
    # Handle span ratio calculation
    span_ratio_pairs = {
        "low_1/16_high_1/8": (1.0/16, 1.0/8),
        "low_1/8_high_1/4": (1.0/8, 1.0/4),
        "low_1/4_high_1/2": (1.0/4, 1.0/2),
        "low_1/2_high_3/4": (1.0/2, 3.0/4),
        "low_3/4_high_7/8": (3.0/4, 7.0/8),
        "low_1/8_high_3/8": (1.0/8, 3.0/8),
    }
    low_ratio, high_ratio = span_ratio_pairs[params["span_ratio_key"]]
    span_low = max(1, int(config.seq_len * low_ratio))
    span_high = max(span_low + 1, int(config.seq_len * high_ratio))
    config.span = (span_low, span_high)

    # Create datasets and dataloaders
    train_dataset, val_dataset = ParquetSequenceDataset.create_train_val_splits(
        parquet_path=config.parquet_path,
        feature_columns=config.feature_columns,
        seq_len=config.seq_len,
        stride=config.stride,
        validation_ratio=config.validation_ratio,
        seed=config.seed
    )

    logging.info(f"\nDataset Information:")
    logging.info(f"Original train dataset files: {len(train_dataset.file_ids)}")
    logging.info(f"Original validation dataset files: {len(val_dataset.file_ids)}")

    # Limit dataset size for faster trials
    if len(train_dataset.file_ids) > 1:
        train_dataset.file_ids = train_dataset.file_ids[:10]
        train_dataset.num_files = len(train_dataset.file_ids)
    if len(val_dataset.file_ids) > 1:
        val_dataset.file_ids = val_dataset.file_ids[:2]
        val_dataset.num_files = len(val_dataset.file_ids)

    logging.info(f"\nDataset Information:")
    logging.info(f"Subset train dataset files: {len(train_dataset.file_ids)}")
    logging.info(f"Subset validation dataset files: {len(val_dataset.file_ids)}")

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
        optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer_name == "AdamW":
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:  # SGD
        optimizer = SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    criterion_mse = torch.nn.MSELoss(reduction='none')
    best_val_loss = float('inf')
    logging.info("\n=== Starting Training ===")
    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, config.device, 
            criterion_mse, config.span, config.one_side,
            lamda_rec=config.lamda_rec, lambda_sacon=config.k_value,
            max_grad_norm=config.max_grad_norm
        )
        
        val_rec_loss, val_total_loss = validate_one_epoch(
            model, val_loader, config.device, 
            criterion_mse, config.span, config.one_side,
            lamda_rec=config.lamda_rec, lambda_sacon=config.k_value
        )

        log_epoch_results(epoch, train_loss, val_rec_loss, val_total_loss)

        # Report intermediate values for better visualization
        trial.report(val_total_loss, epoch)
        trial.report(train_loss, epoch)
        
        # Log metrics for each epoch
        trial.set_user_attr(f"train_loss_epoch_{epoch}", float(train_loss))
        trial.set_user_attr(f"val_loss_epoch_{epoch}", float(val_total_loss))
        trial.set_user_attr(f"val_rec_loss_epoch_{epoch}", float(val_rec_loss))
        
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            trial.set_user_attr("best_epoch", epoch)
            logging.info(f"New best validation loss: {best_val_loss:.6f}")

        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Log final metrics
    trial.set_user_attr("final_train_loss", float(train_loss))
    trial.set_user_attr("final_val_loss", float(val_total_loss))
    trial.set_user_attr("best_val_loss", float(best_val_loss))

    # Log final metrics
    logging.info("\n=== Trial Summary ===")
    logging.info(f"Final Training Loss: {train_loss:.6f}")
    logging.info(f"Final Validation Loss: {val_total_loss:.6f}")
    logging.info(f"Best Validation Loss: {best_val_loss:.6f}")


    del model
    torch.cuda.empty_cache()
    return best_val_loss

def main():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('results', f'optuna_optimization_{current_time}')
    os.makedirs(output_dir, exist_ok=True)

    start_logging()

    config = Config()
    log_config_details(config)
    
    logging.info(f"\n Inital Optimizer Configuration:")
    logging.info(f"Type: {config.optimizer_name}")
    logging.info(f"Learning Rate: {config.learning_rate}")
    logging.info(f"Weight Decay: {config.weight_decay}")

    study_name = "optimization_study"
    storage_name = f"sqlite:///{output_dir}/optuna_study.db"
    
    # Log study configuration
    logging.info("\n=== Starting Hyperparameter Optimization ===")
    logging.info(f"Study Name: {study_name}")
    logging.info(f"Storage: {storage_name}")
    
    # Create study with pruner and sampler
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
        interval_steps=1
    )
    
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,
        multivariate=True,
        constant_liar=True
    )

    logging.info("\nPruner Configuration:")
    logging.info(f"Type: MedianPruner")
    logging.info(f"N startup trials: 5")
    logging.info(f"N warmup steps: 5")
    
    logging.info("\nSampler Configuration:")
    logging.info(f"Type: TPESampler")
    logging.info(f"N startup trials: 10")
    logging.info(f"Multivariate: True")
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='minimize',
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler
    )
    
    # Add study user attributes for better organization
    study.set_user_attr("description", "Hyperparameter optimization for sequence model")
    study.set_user_attr("model_version", "1.0")
    study.set_user_attr("dataset", "sequence_data")
    
    study.optimize(
        objective,
        n_trials=5,
        show_progress_bar=True,
        gc_after_trial=True
    )
    
    # Log best trial information
    best_trial = study.best_trial
    logging.info("\n=== Best Trial Results ===")
    logging.info(f"Best trial number: {best_trial.number}")
    logging.info(f"Best trial value: {best_trial.value}")
    logging.info("\nBest trial parameters:")
    for key, value in best_trial.params.items():
        logging.info(f"  {key}: {value}")
    
    logging.info("\n=== Study Statistics ===")
    logging.info(f"Number of completed trials: {len(study.trials)}")
    logging.info(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    logging.info(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    logging.info("\n=== Optuna Dashboard Instructions ===")
    logging.info("To view the dashboard, run the following command in your terminal:")
    logging.info(f"optuna-dashboard {storage_name}")
    logging.info("Then open http://127.0.0.1:8080 in your browser")

if __name__ == "__main__":
    main()