"""Helper functions for clustering channels and setting up logging and directories."""

import logging
import os
import sys
from typing import (
    Any,
    Dict,
    List,
)

import yaml  # type: ignore[import-untyped]

# Remove the line where logger is redefined
sys.path.append("../telemanom")


class Config:  # pylint: disable=too-many-public-methods
    """Loads parameters from config.yaml into global object."""

    def __init__(self, path_to_config: str):
        """Initialize the Config object.

        Args:
            path_to_config (str): The path to the config.yaml file.
        """

        if os.path.isfile(path_to_config):
            pass
        else:
            path_to_config = f"../{path_to_config}"

        with open(path_to_config, "r", encoding="utf-8") as f:
            self.dictionary: Dict[str, Any] = yaml.load(
                f.read(), Loader=yaml.FullLoader
            )

    @property
    def train(self) -> bool:
        """Return whether to train model."""
        return self.dictionary.get("train", False)

    @property
    def predict(self) -> bool:
        """Return whether to predict model."""
        return self.dictionary.get("predict", False)

    @property
    def p(self) -> int:
        """Return the threshold p."""
        return self.dictionary.get("p", 1)

    @property
    def use_id(self) -> str:
        """Return use ID."""
        return self.dictionary.get("use_id", None)

    @property
    def l_s(self) -> int:
        """Return sequence length."""
        return self.dictionary.get("l_s", 100)

    @property
    def n_predictions(self) -> int:
        """Return number of predictions."""
        return self.dictionary.get("n_predictions", 1)

    @property
    def batch_size(self) -> int:
        """Return batch size."""
        return self.dictionary.get("batch_size", 32)

    @property
    def validation_split(self) -> float:
        """Return validation split."""
        return self.dictionary.get("validation_split", 0.2)

    @property
    def activation(self) -> str:
        """Return activation function."""
        return self.dictionary.get("activation", "tanh")

    @property
    def bias(self) -> bool:
        """Return bias."""
        return self.dictionary.get("bias", True)

    @property
    def cuda_id(self) -> int:
        """Return CUDA ID."""
        return self.dictionary.get("cuda_id", 0)

    @property
    def kernel_initializer(self) -> str:
        """Return kernel initializer."""
        return self.dictionary.get("kernel_initializer", "glorot_uniform")

    @property
    def recurrent_initializer(self) -> str:
        """Return recurrent initializer."""
        return self.dictionary.get("recurrent_initializer", "orthogonal")

    @property
    def net_gain_and_bias(self) -> bool:
        """Return net gain and bias."""
        return self.dictionary.get("net_gain_and_bias", False)

    @property
    def patience(self) -> int:
        """Return patience."""
        return self.dictionary.get("patience", 10)

    @property
    def epochs(self) -> int:
        """Return number of epochs."""
        return self.dictionary.get("epochs", 100)

    @property
    def window_size(self) -> int:
        """Return window size."""
        return self.dictionary.get("window_size", 10)

    @property
    def error_buffer(self) -> int:
        """Return error buffer."""
        return self.dictionary.get("error_buffer", 100)

    @property
    def smoothing_perc(self) -> int:
        """Return smoothing percentage."""
        return self.dictionary.get("smoothing_perc", 50)

    @property
    def smoothing(self) -> int:
        """Return smoothing."""
        return self.dictionary.get("smoothing", 10)

    @property
    def learning_rate(self) -> float:
        """Return learning rate."""
        return self.dictionary.get("learning_rate", 0.001)

    @learning_rate.setter
    def learning_rate(self, value: float):
        """Set learning rate."""
        self.dictionary["learning_rate"] = value

    @property
    def model_architecture(self) -> str:
        """Return model architecture."""
        return self.dictionary.get("model_architecture", "ESN")

    @model_architecture.setter
    def model_architecture(self, value: str):
        """Set model architecture."""
        self.dictionary["model_architecture"] = value

    @property
    def l2(self) -> List[float]:
        """Return L2 regularization."""
        return self.dictionary.get("l2", [0.0])

    @l2.setter
    def l2(self, value: List[float]):
        """Set L2 regularization."""
        self.dictionary["l2"] = value

    @property
    def layers(self) -> List[int]:
        """Return number of layers."""
        return self.dictionary.get("layers", [1])

    @layers.setter
    def layers(self, value: List[int]):
        """Set number of layers."""
        self.dictionary["layers"] = value

    @property
    def leakage(self) -> float:
        """Return leakage."""
        return self.dictionary.get("leakage", 1.0)

    @leakage.setter
    def leakage(self, value: float):
        """Set leakage."""
        self.dictionary["leakage"] = value

    @property
    def input_scaling(self) -> float:
        """Return input scaling."""
        return self.dictionary.get("input_scaling", 1.0)

    @input_scaling.setter
    def input_scaling(self, value: float):
        """Set input scaling."""
        self.dictionary["input_scaling"] = value

    @property
    def rho(self) -> float:
        """Return spectral radius."""
        return self.dictionary.get("rho", 0.9)

    @rho.setter
    def rho(self, value: float):
        """Set spectral radius."""
        self.dictionary["rho"] = value

    @property
    def lr(self) -> float:
        """Return learning rate."""
        return self.dictionary.get("lr", 0.001)

    @lr.setter
    def lr(self, value: float):
        """Set learning rate."""
        self.dictionary["lr"] = value

    @property
    def weight_decay(self) -> float:
        """Return weight decay."""
        return self.dictionary.get("weight_decay", 0.0)

    @weight_decay.setter
    def weight_decay(self, value: float):
        """Set weight decay."""
        self.dictionary["weight_decay"] = value

    @property
    def dropout(self) -> float:
        """Return dropout."""
        return self.dictionary.get("dropout", 0.0)

    @dropout.setter
    def dropout(self, value: float):
        """Set dropout."""
        self.dictionary["dropout"] = value


def make_dirs(_id: str):
    """Create directories for storing data in repo (using datetime ID) if they don't
    already exist."""

    config = Config("config.yaml")

    if not config.train or not config.predict:
        if not os.path.isdir(f"data/{config.use_id}"):
            raise ValueError(
                f"Run ID {_id} is not valid. "
                "If loading prior models or predictions, must provide valid ID."
            )

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
