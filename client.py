import argparse
import warnings
from logging import INFO
import xgboost as xgb

import pandas as pd
from sklearn.model_selection import train_test_split
import flwr as fl
from flwr.common.logger import log
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)

import data_handler


BASE_DATASET_PATH = "processed_data_3"
LEARNING_RATE = 0.1
SAMPLE_SIZE = 20


warnings.filterwarnings("ignore", category=UserWarning)

# Define arguments parser for the client/partition ID.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--partition-id",
    default=0,
    type=int,
    help="Partition ID used for the current client.",
)
parser.add_argument(
    "--n-clients",
    type=int,
    default=1,
)
args = parser.parse_args()


def transform_dataset_to_dmatrix(data: pd.DataFrame) -> xgb.core.DMatrix:
    """Transform dataset to DMatrix format for xgboost."""
    x, y = data_handler.get_x_y(df=data, reshape=False)
    new_data = xgb.DMatrix(x, label=y)
    return new_data


# Load the partition for this `partition_id`
log(INFO, "Loading partition...")
paths = data_handler.get_paths(
    base_path=BASE_DATASET_PATH,
    client_num=args.partition_id,
    num_of_clients=args.n_clients
)
data = data_handler.load_dataset(paths=paths, sample_size=-1)
train, valid = train_test_split(data, test_size=0.05)

num_train, num_val = len(train), len(valid)

# Hyper-parameters for xgboost training
num_local_round = 1
params = {
    "objective": "binary:logistic",
    "eta": LEARNING_RATE,  # Learning rate
    "max_depth": 8,
    "eval_metric": "auc",
    "nthread": 16,
    "num_parallel_tree": 1,
    "subsample": 1,
    "tree_method": "hist",
}


# Define Flower client
class XgbClient(fl.client.Client):
    def __init__(
        self,
        train_data,
        valid_data,
        num_train,
        num_val,
        num_local_round,
        params,
    ):
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params
        self.train_data = train_data
        self.valid_data = valid_data
        self.train_dmatrix = None
        self.valid_dmatrix = transform_dataset_to_dmatrix(valid_data)

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def _local_boost(self, bst_input):
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for sever aggregation
        bst = bst_input[
            bst_input.num_boosted_rounds()
            - self.num_local_round : bst_input.num_boosted_rounds()
        ]

        return bst

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        all_rounds = int(ins.config["all_rounds"])
        train_step = len(self.train_data) // all_rounds
        # train_data = self.train_data[(global_round-1)*train_step:global_round*train_step]
        train_data = self.train_data.sample(n=SAMPLE_SIZE)

        # Reformat data to DMatrix for xgboost
        log(INFO, "Reformatting data...")
        self.train_dmatrix = transform_dataset_to_dmatrix(train_data)

        if global_round == 1:
            # First round local training
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
        else:
            bst = xgb.Booster(params=self.params)
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )


# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=XgbClient(
        train,
        valid,
        num_train,
        num_val,
        num_local_round,
        params,
    ).to_client(),
)
