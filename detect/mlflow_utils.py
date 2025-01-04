# mlflow_utils.py
import mlflow
import mlflow.pytorch
import os
import logging
from mlflow.tracking import MlflowClient
import glob

def start_mlflow_run(experiment_name: str, params) -> None:
    """
    Sets or creates an MLflow experiment and starts a run under it.
    """
    run_name = (
    f"q{str(params.q)[-2:]}::p{params.p}::s{params.seq_len}::stride{params.stride if params.stride is not None else 'st-1'}::h{params.n_heads}::"
    f"e{params.e_layers}::d{params.model_dim}::batch{params.batch_size}::"
    f"k{str(params.k_value)}::dr{int(params.dropout * 100)}::rec{params.lamda_rec}::"
    f"span{params.span[0]}::{params.span[1]}::"
    f"side{1 if params.one_side else 0}::negQK{1 if params.negative_qk else 0}"
    f"::grad{params.max_grad_norm}"
    f"::lr{str(params.learning_rate)}::wd{str(params.weight_decay)}"
    f"::opt{params.optimizer_name}"
    f"::fun{params.activation}"
)

    local_tracking_dir = "/workspaces/thesis/detect/mlruns"
    mlflow.set_tracking_uri(f"file://{local_tracking_dir}")
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
        
    existing_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tag.mlflow.runName = '{run_name}'"
    )
    
    if existing_runs:
        mlflow.start_run(run_id=existing_runs[0].info.run_id)
    else:
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
    artifact_path = "best_model"
    if os.path.exists(checkpoint_path):
        mlflow.log_artifact(checkpoint_path, artifact_path)
        logging.info(f"Checkpoint artifact logged: {checkpoint_path}")
    else:
        logging.warning(f"Checkpoint path does not exist: {checkpoint_path}")

def log_plots(input_dir: str) -> None:
    for file in glob.glob(os.path.join(input_dir, "*.png")):
        artifact_path = "attention_plots" if "attention" in file else "results"
        mlflow.log_artifact(file, artifact_path)

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
