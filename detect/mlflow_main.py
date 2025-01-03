# subAdjacent/detector.py
import logging
import torch
import sys, os
from tqdm import tqdm
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from approaches.subAdjacent.configClass import Config
#from approaches.subAdjacent.model import anomalyTransformer

from detect.mlflow_utils import (
    start_mlflow_run,
    end_mlflow_run,
    log_params_from_config,
    download_artifact,
    log_plot
)


def main():

    params = Config()
    model = params.build_model()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Start MLflow run
    start_mlflow_run(experiment_name="subAdjacent", run_name="detector")

    # Log params
    log_params_from_config(params)

    



    # End MLflow run
    end_mlflow_run()


if __name__ == "__main__":
    main()
