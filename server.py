from logging import INFO

import pandas as pd 
import xgboost as xgb
from typing import Dict, Optional, Tuple
import flwr as fl
from flwr.server.strategy import FedXgbBagging
from flwr.common import Parameters, Scalar
from flwr.common.logger import log

import data_handler

# FL experimental settings
num_rounds = 20
num_clients = 8
BASE_DATASET_PATH = "processed_data_2"
LEARNING_RATE = 0.1

BST_PARAMS = {
    "objective": "binary:logistic",
    "eta": LEARNING_RATE,  # Learning rate
    "max_depth": 8,
    "eval_metric": "auc",
    "nthread": 16,
    "num_parallel_tree": 1,
    "subsample": 1,
    "tree_method": "hist",
}


def transform_dataset_to_dmatrix(data: pd.DataFrame) -> xgb.core.DMatrix:
    """Transform dataset to DMatrix format for xgboost."""
    x, y = data_handler.get_x_y(df=data, reshape=False)
    new_data = xgb.DMatrix(x, label=y)
    return new_data


def get_evaluate_fn():
    """Return a function for centralised evaluation."""

    test_data = data_handler.load_dataset(paths=[f"{BASE_DATASET_PATH}/test.csv"])
    test_data = transform_dataset_to_dmatrix(test_data)

    def evaluate_fn(
        server_round: int, parameters: Parameters, config: Dict[str, Scalar]
    ):
        # If at the first round, skip the evaluation
        if server_round == 0:
            return 0, {}
        else:
            bst = xgb.Booster(params=BST_PARAMS)
            for para in parameters.tensors:
                para_b = bytearray(para)

            # Load global model
            bst.load_model(para_b)
            # Run evaluation
            eval_results = bst.eval_set(
                evals=[(test_data, "valid")],
                iteration=bst.num_boosted_rounds() - 1,
            )
            auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
            log(INFO, f"AUC = {auc} at round {server_round}")

            return 0, {"test_auc": auc}

    return evaluate_fn


def config_func(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
        "all_rounds": str(num_rounds)
    }
    return config


# Define strategy
strategy = FedXgbBagging(
    min_fit_clients=num_clients,
    min_available_clients=num_clients,
    min_evaluate_clients=num_clients,
    fraction_evaluate=1.0,
    evaluate_metrics_aggregation_fn=None,
    on_evaluate_config_fn=config_func,
    on_fit_config_fn=config_func,
    evaluate_function=get_evaluate_fn(),

)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
)
