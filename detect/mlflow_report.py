# detect/mlflow_report.py
import logging
import sys, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from approaches.subAdjacent.configClass import Config
from detect.mlflow_utils import *


def main(input_dir: str): 

    #params = Config()
    #model = params.build_model()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    start_mlflow_run(experiment_name="subAdjacent", run_name="detect_best_trial")
    #log_params_from_config(params)
    #log_plots(input_dir)
    #log_log(input_dir)
    # add mlflow the only log file inside input_dir
    log_file = [f for f in os.listdir(input_dir) if f.endswith('.log')]
    if log_file:
        log_file_path = os.path.join(input_dir, log_file[0])
        mlflow.log_artifact(log_file_path, artifact_path="logs")

    #log_anomalies(input_dir)
    #log_torch_model(model, "model")
    log_checkpoint_artifact(input_dir)

    # add mlflow configClass.py
    mlflow.log_artifact(os.path.join(input_dir, "configClass.py"))



if __name__ == "__main__":
    folder_name = "detect/reportable/results"
    main(folder_name)
    end_mlflow_run()

