"""Helper functions for clustering channels and setting up logging and directories."""

import logging
import os
import sys
from typing import List


class Config:
    """Loads parameters from config.yaml into global object."""

    def __init__(self):
        self.model_architecture: str = "ESN"  # neural network architecture LSTM, ESN
        self.num_sampler: int = 200
        self.num_sampler2: int = 20
        self.cpu: int = 1
        self.cpu_adaptive: int = 1
        self.train: bool = True  # train new or existing model for each channel
        self.predict: bool = (
            True  # generate new predicts or, if False, use predictions stored locally
        )
        self.use_id: int = 0
        self.cuda_id: int = 0
        self.batch_size: int = 70  # number of values to evaluate in each batch
        self.window_size: int = (
            30  # number of trailing batches to use in error calculation
        )
        self.header: list = [  # Columns headers for output file
            "run_id",
            "chan_id",
            "spacecraft",
            "num_anoms",
            "anomaly_sequences",
            "class",
            "true_positives",
            "false_positives",
            "false_negatives",
            "tp_sequences",
            "fp_sequences",
            "gaussian_p-value",
            "num_values",
            "normalized_error",
            "eval_time",
            "scores",
        ]
        self.smoothing_perc: float = (
            0.05  # determines window size used in EWMA smoothing (percentage of total values for channel)
        )
        self.error_buffer: int = (
            100  # number of values surrounding an error that are brought into the sequence (promotes grouping on nearby sequences
        )

        # model parameters
        self.loss_metric: str = "mse"
        self.optimizer: str = "adam"
        self.learning_rate: float = 0.001
        self.validation_split: float = 0.2
        self.dropout: float = 0.3
        self.lstm_batch_size: int = 64
        self.esn_batch_number: int = 32
        self.weight_decay: float = 0
        self.epochs: int = (
            15  # maximum number of epochs allowed (if early stopping criteria not met)
        )
        self.layers: list = [
            80,
            80,
        ]  # network architecture [<neurons in hidden layer>, <neurons in hidden layer>]
        self.patience: int = (
            5  # Number of consequetive training iterations to allow without decreasing the val_loss by at least min_delta
        )
        self.min_delta: float = 0.0003
        self.l_s: int = (
            250  # num previous timesteps provided to model to predict future values
        )
        self.n_predictions: int = 10  # number of steps ahead to predict

        # Parameters only for ESN
        self.activation: str = (
            "tanh"  # Name of the activation function from `torch` (e.g. `torch.tanh`)
        )
        self.leakage: float = 1  # The value of the leaking parameter `alpha`
        self.input_scaling: float = (
            0.9  # The value for the desired scaling of the input (must be `<= 1`)
        )
        self.rho: float = (
            0.99  # The desired spectral radius of the recurrent matrix (must be `< 1`)
        )
        self.kernel_initializer: str = (
            "uniform"  # The kind of initialization of the input transformation. Default: `'uniform'`
        )
        self.recurrent_initializer: str = (
            "normal"  # The kind of initialization of the recurrent matrix. Default: `'normal'`
        )
        self.net_gain_and_bias: bool = (
            False  # If ``True``, the network uses additional ``g`` (gain) and ``b`` (bias) parameters. Default: ``False``
        )
        self.bias: bool = False  # If ``False``, the layer does not use bias weights `b`
        self.l2: list = [1e-10]  # The value of l2 regularization

        # The value of l2 regularization
        self.p: float = (
            0.13  # minimum percent decrease between max errors in anomalous sequences (used for pruning)
        )


def make_dirs(_id: str):
    """Create directories for storing data in repo (using datetime ID) if they don't
    already exist."""

    paths: List[str] = [
        "data",
        f"data/{_id}",
        "data/logs",
        f"data/{_id}/models",
        f"data/{_id}/smoothed_errors",
        f"data/{_id}/y_hat",
    ]

    for p in paths:
        if not os.path.isdir(p):
            os.mkdir(p)


def setup_logging() -> logging.Logger:
    """Configure logging object to track parameter settings, training, and evaluation.

    Args:
        config(obj): Global object specifying system runtime params.

    Returns:
        logger (obj): Logging object
        _id (str): Unique identifier generated from datetime for storing data/models/results
    """

    logger = logging.getLogger("telemanom")
    logger.setLevel(logging.INFO)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    logger.addHandler(stdout)

    return logger
