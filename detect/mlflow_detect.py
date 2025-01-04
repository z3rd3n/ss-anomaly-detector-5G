import os
import torch
import mlflow
import logging
import glob
from mlflow.tracking import MlflowClient
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from approaches.subAdjacent.configClass import Config
from approaches.subAdjacent.trainer import detect_anomalies
from detect.mlflow_utils import download_artifact
from detect.log_finder import extract_params_from_log, bring_log_file
from utils import load_checkpoint
from data.dataLoader import ParquetSequenceDataset, custom_collate_fn
from torch.utils.data import DataLoader

def detect_in_existing_run(
    run_id: str,
    detection_artifact_path: str = "detection_results"
):
    """
    1) Resumes the MLflow run with the given run_id
    2) Downloads checkpoint_best.pt from MLflow
    3) Builds the same model & optimizer
    4) Loads checkpoint
    5) Runs detect_anomalies
    6) Logs new artifacts under `detection_artifact_path` in the same run
    """

    mlflow.set_experiment("subAdjacent")  
    mlflow.start_run(run_id=run_id)
    logging.info(f"Resumed MLflow run: {run_id}")
    local_log_folder = download_artifact(run_id, "logs", dst_path="temp")

    params = Config()
    log_file = bring_log_file(local_log_folder) 
    params = extract_params_from_log(log_file, params)
    # update params with log_params dict
    
    model = params.build_model()
    params.output_dir = "temp"  

    os.makedirs(params.output_dir, exist_ok=True)
    local_ckpt_path = download_artifact(run_id, "checkpoints/checkpoint_best.pt", dst_path=params.output_dir)

    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"Selected device: {device}")
    checkpoint = torch.load(local_ckpt_path, map_location=device , weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    learning_rate = checkpoint.get("learning_rate", params.learning_rate)
    weight_decay = checkpoint.get("weight_decay", params.weight_decay)
    if checkpoint["optimizer_name"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    elif checkpoint["optimizer_name"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        raise ValueError(f"Unsupported optimizer: {checkpoint['optimizer_name']}")
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    logging.info(f"Checkpoint loaded (epoch={epoch+1}, loss={loss:.4f})")

    logging.info("Creating train and validation datasets...")
    _, val_dataset = ParquetSequenceDataset.create_train_val_splits(
        parquet_path=params.parquet_path,
        feature_columns=params.feature_columns,
        seq_len=params.seq_len,
        validation_ratio=params.validation_ratio,
        seed=params.seed
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=params.pin_memory,
        drop_last=True
    )

    detect_anomalies(params, model, optimizer, val_loader, mlflow=True)

    
    for file_path in glob.glob(os.path.join(params.output_dir, "*.*")):
        mlflow.log_artifact(file_path, artifact_path=detection_artifact_path)
        logging.info(f"Logged detect artifact => {file_path}")
    # delete after logging
    os.rmdir(params.output_dir)
    mlflow.end_run()
    logging.info(f"Detection completed for run {run_id}.")


if __name__ == "__main__":
    # Example usage
    # 1) supply the run_id you want to resume
    # 2) optionally choose a subfolder for detection artifacts
    run_id = "4c1cb482a928404ba86b2cba82e21e3e"  # your real run_id from MLflow
    detect_in_existing_run(run_id, "detect_anomalies")
