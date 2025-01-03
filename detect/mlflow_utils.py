# mlflow_utils.py
import mlflow
import mlflow.pytorch
import os
import logging
from mlflow.tracking import MlflowClient

def start_mlflow_run(experiment_name: str, run_name: str = None) -> None:
    """
    Sets or creates an MLflow experiment and starts a run under it.
    """
    local_tracking_dir = "/workspaces/thesis/detect/mlruns"
    mlflow.ui(port=5000, host="0.0.0.0")
    mlflow.set_tracking_uri(f"file://{local_tracking_dir}")
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name)
    # Ensure this directory exists

    
    logging.info(f"Started MLflow run under experiment: {experiment_name}, run name: {run_name}")

def log_params_from_config(config_obj: object) -> None:
    """
    Logs all attributes from a config class to MLflow as parameters.
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

def log_checkpoint_artifact(checkpoint_path: str, artifact_path: str = "checkpoints") -> None:
    """
    Logs a .pt checkpoint file as an MLflow artifact. 
    You can later download it to do detection with different parameters.
    """
    if os.path.exists(checkpoint_path):
        mlflow.log_artifact(checkpoint_path, artifact_path)
        logging.info(f"Checkpoint artifact logged: {checkpoint_path}")
    else:
        logging.warning(f"Checkpoint path does not exist: {checkpoint_path}")

def log_plot(plot_path: str, artifact_path: str = "") -> None:
    """
    Logs a single plot (PNG) file to MLflow artifacts. 
    """
    if os.path.exists(plot_path):
        mlflow.log_artifact(plot_path, artifact_path)
        logging.info(f"Plot artifact logged: {plot_path}")
    else:
        logging.warning(f"Plot path does not exist: {plot_path}")

def end_mlflow_run() -> None:
    """
    Ends the MLflow run.
    """
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
