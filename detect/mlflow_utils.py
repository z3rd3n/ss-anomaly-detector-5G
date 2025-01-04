# mlflow_utils.py
import mlflow
import mlflow.pytorch
import os
import logging
from mlflow.tracking import MlflowClient
import glob
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from detect.log_finder import find_log_with_anomalies

def start_mlflow_run(experiment_name: str, run_name:str) -> None:
    """
    Sets or creates an MLflow experiment and starts a run under it.
    """
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name)
    logging.info(f"Started MLflow run under experiment: {experiment_name}, run name: {run_name}")

def log_params_from_config(config_obj: object) -> None:
    """
    Logs all attributes from a config class to MLflow as paramseters.
    """
    cfg_dict = vars(config_obj)
    for k, v in cfg_dict.items():
        # Only log basic Python types (str, int, float, bool, etc.)
        if isinstance(v, (str, int, float, bool, type(None))):
            mlflow.log_param(k, v)
        else:
            # For more complex objects, log them as string or skip
            mlflow.log_param(k, str(v))

def log_torch_model(model, artifact_path: str = "models", **kwargs) -> None:
    """
    Logs a PyTorch model to MLflow.
    """
    mlflow.pytorch.log_model(model, artifact_path, **kwargs)
    logging.info(f"Model logged to MLflow at artifact path: {artifact_path}")

def log_checkpoint_artifact(input_dir) -> None:
    checkpoint_path = os.path.join(input_dir,"checkpoints", "checkpoint_best.pt")
    artifact_path = "checkpoints"
    if os.path.exists(checkpoint_path):
        mlflow.log_artifact(checkpoint_path, artifact_path)
        logging.info(f"Checkpoint artifact logged: {checkpoint_path}")
    else:
        logging.warning(f"Checkpoint path does not exist: {checkpoint_path}")

def log_plots(input_dir: str) -> None:
    for file in glob.glob(os.path.join(input_dir, "*.png")):
        artifact_path = "attention_plots" if "attention" in file else "results"
        mlflow.log_artifact(file, artifact_path)

def log_log(input_dir: str) -> None:
    log_file = find_log_with_anomalies(input_dir)
    if log_file:
        mlflow.log_artifact(log_file, "logs")
    else:
        logging.warning(f"No log file found in {input_dir}")

def log_anomalies(input_dir: str) -> None:
    for file in glob.glob(os.path.join(input_dir, "*.csv")):
        mlflow.log_artifact(file, "anomalies")

def end_mlflow_run() -> None:
    mlflow.end_run()

def download_artifact(run_id: str, artifact_path: str, dst_path: str = None) -> str:
    """
    Downloads an artifact (model/checkpoint) from MLflow to a local directory.
    Returns the local path where the file is downloaded.
    """
    client = MlflowClient()
    local_path = client.download_artifacts(run_id, artifact_path, dst_path or ".")
    logging.info(f"Downloaded artifact {artifact_path} to local path: {local_path}")
    return local_path
