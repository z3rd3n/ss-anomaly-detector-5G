# detect/mlflow_report.py
import logging
import sys, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from approaches.subAdjacent.configClass import Config
from detect.mlflow_utils import *


def main(input_dir: str): 

    params = Config()
    model = params.build_model()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    start_mlflow_run(experiment_name="subAdjacent", run_name="detect")
    log_params_from_config(params)
    log_plots(input_dir)
    log_log(input_dir)
    log_anomalies(input_dir)
    log_torch_model(model, "model")
    log_checkpoint_artifact(input_dir)


if __name__ == "__main__":
    folder_name = "detect/reportable/results"
    main(folder_name)
    end_mlflow_run()

