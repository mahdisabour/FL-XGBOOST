# Federated Learning with XGBoost (FL-XGBOOST)

This repository contains an example/quickstart for running federated learning (FL) with XGBoost using the Flower (flwr) framework.

The project shows a simple server and client implementation where each client trains locally on a partition of a dataset and the server aggregates model updates.

## Table of contents
- Project overview
- Requirements
- Install
- Run (single-machine simulation)
- Scripts
- Code overview & important functions
- Data format & preprocessing
- Troubleshooting
- Contact

## Project overview

Key files:

- `server.py` — Flower server, defines the federated strategy and evaluation function.
- `client.py` — Flower client; loads a partition of the dataset, performs local XGBoost training and returns updates.
- `data_handler.py` — Helpers for loading, preprocessing and partitioning CSV data.
- `plot_data.py` — Simple log parser + plotting helper for results.
- `run.sh` — Convenience script to launch the server and N clients on a single machine (for simulation).
- `run-all-tests.sh` — Example script that runs `run.sh` for a set of client counts and appends logs to `results.txt`.

## Requirements

- Python 3.8+ (assumption based on dependencies; if you use a different Python version, test locally).
- Packages listed in `pyproject.toml`: Flower (`flwr`), `flwr-datasets`, and `xgboost`.

You can see the exact versions in `pyproject.toml`:

```toml
dependencies = [
		"flwr>=1.8.0,<2.0",
		"flwr-datasets>=0.1.0,<1.0.0",
		"xgboost>=2.0.0,<3.0.0",
]
```

## Install

Recommended (virtual environment):

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
# Install required packages directly (from pyproject or explicit list):
pip install "flwr>=1.8.0,<2.0" "flwr-datasets>=0.1.0,<1.0.0" "xgboost>=2.0.0,<3.0.0"
# OR (if you prefer editable install when pyproject supports it):
pip install -e .
```

If you add further dependencies, update `pyproject.toml` or install them into your virtualenv.

## Run (single-machine simulation)

This project is designed to let you simulate federated learning locally by starting one server process and multiple client processes on the same machine.

Basic usage (example with 3 clients):

```bash
./run.sh N_CLIENTS=3
```

What `run.sh` does:

- Starts `server.py` in the background and writes server output to `./logs/file_xgboost_<N>.log`.
- Sleeps briefly to allow the server to come up.
- Starts `N_CLIENTS` background `client.py` processes (each with their partition id).
- Waits for all processes. Ctrl+C will stop the set of background processes.

Advanced / manual: you can also start the server and clients manually.

Start the server (defaults to 1 client if not changed):

```bash
python server.py --n-clients 3
```

Start a client (for partition id 0 out of 3):

```bash
python client.py --partition-id=0 --n-clients=3
```

Repeat the client command (with partition-id 1, 2, ...) for additional clients.

## Scripts

- `run.sh N_CLIENTS=<n>` — convenience wrapper to start server and `n` clients locally.
- `run-all-tests.sh` — example script to run `run.sh` for configured values (appends logs to `results.txt`).

## Code overview & important parts

Below are the most important places to look and what they do.

- `server.py`
	- Configures global training using a Flower strategy: `FedXgbBagging`.
	- `get_evaluate_fn()` builds a centralized evaluation function that loads the global model and evaluates it on the test data. This function returns a tuple: (loss, {metrics}).
	- `config_func(rnd)` returns per-round configuration passed to clients (e.g., `global_round`, `all_rounds`).
	- Key constants: `BASE_DATASET_PATH`, `BST_PARAMS` (XGBoost hyperparameters), and `num_rounds`.

- `client.py`
	- Implements `XgbClient`, a Flower client class.
	- `fit(self, ins: FitIns) -> FitRes` is the core training routine:
		- Receives global config and (optionally) parameters from server.
		- Prepares local data (samples a subset in this example), transforms it to XGBoost `DMatrix` format.
		- For round 1, trains a fresh booster; for later rounds, loads the global model and performs local boosting via `_local_boost`.
		- Returns `FitRes` with serialized local model bytes inside `Parameters(tensors=[...])` and `num_examples`.
	- `get_parameters` returns an empty parameters object in this implementation (parameters are exchanged via raw model bytes in `FitRes` / custom strategy behavior).

- `data_handler.py`
	- Helpers for loading CSV files and preprocessing.
	- `load_dataset(paths, preprocess=False, sample_size=1)` concatenates CSVs and optional sampling/preprocessing.
	- `get_paths(num_of_clients, client_num, base_path)` computes which CSV files belong to a given client partition.
	- `get_x_y(df, binary=True, reshape=True)` prepares X and y arrays and performs scaling. Note: it assumes a column named ` Label` for labels and maps to `classify_label`.
	- `split_and_save_data_frame(df, output_path, file_counts)` can be used to split a large dataframe into per-client files.

- `plot_data.py`
	- A small utility that parses a `results.txt`/log file and plots accuracy lines per client count. Useful for quick visualization of results.

## Data format & preprocessing

- The code expects prepared CSVs in a folder like `processed_data_3/` with files representing partitions (the code uses `BASE_DATASET_PATH = "processed_data_3"`).
- `data_handler.get_paths()` picks files from that folder (excluding files that contain "test" in the filename) and divides them across clients.
- `get_x_y()` expects a label column named exactly ` Label` (note the leading space). It creates a `classify_label` column and drops the original label from features.
- If you need to prepare raw data, use `data_handler.preprocess_data()` and `split_and_save_data_frame()` to produce per-client CSVs.

## Parameters & tuning

- XGBoost hyperparameters are defined in `BST_PARAMS` (server) and `params` (client) — adjust `eta`, `max_depth`, and others to fit your data.
- `num_rounds` in `server.py` controls the total federated rounds; `num_local_round` in `client.py` controls how many boosting rounds each client performs per round.

## Troubleshooting

- If clients can't connect, ensure the server is running on `0.0.0.0:8080` and the clients connect to `127.0.0.1:8080` (the default in this project).
- If your dataset is large, consider reducing `SAMPLE_SIZE` in `client.py` or increasing system resources.
- If you see label-related errors, confirm that your CSVs contain the ` Label` column and that features are numeric.

## Running tests / sample run

There is an example `run-all-tests.sh` which runs `run.sh` for a configured client count and appends logs to `results.txt`.

To run the example test script locally:

```bash
./run-all-tests.sh
```

Note: these scripts spawn background processes and write logs to `./logs/` — ensure that directory exists and you have write permission.

## Assumptions made

- Python 3.8+ is available.
- You will run this locally for simulation and have sufficient CPU/RAM for XGBoost.

