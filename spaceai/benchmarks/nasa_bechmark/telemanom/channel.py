"""Load and shape channel values (predicted and actual) for ingestion into LSTM or
ESN."""

import logging
import os
import sys

import numpy as np
import torch
from telemanom.helpers import Config
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

sys.path.append("spaceai/spaice-autocl-main/telemanom")

logger = logging.getLogger("telemanom")


class Channel:  # pylint: disable=too-many-instance-attributes
    """Load and shape channel values (predicted and actual) for ingestion into LSTM or
    ESN."""

    def __init__(self, config: Config, chan_id: str, train_with_val: bool = True):
        """Load and reshape channel values (predicted and actual).

        Args:
            config (object): Config object containing parameters for processing
            chan_id (str): channel id

        Attributes:
            id (str): channel id
            config (object): see Args
            X_train (np.ndarray): training inputs with dimensions
                [timesteps, l_s, input dimensions)
            X_test (np.ndarray): test inputs with dimensions
                [timesteps, l_s, input dimensions)
            y_train (np.ndarray): actual channel training values with dimensions
                [timesteps, n_predictions, 1)
            y_test (np.ndarray): actual channel test values with dimensions
                [timesteps, n_predictions, 1)
            train (np.ndarray): train data loaded from .npy file
            test (np.ndarray): test data loaded from .npy file
        """

        self.id: str = chan_id
        self.config: Config = config
        self.X_train: np.ndarray  # pylint: disable=invalid-name
        self.y_train: np.ndarray
        self.X_valid: np.ndarray  # pylint: disable=invalid-name
        self.y_valid: np.ndarray
        self.X_test: np.ndarray  # pylint: disable=invalid-name
        self.y_test: np.ndarray
        self.y_hat: np.ndarray
        self.train: np.ndarray
        self.test: np.ndarray
        self.train_with_val: bool = train_with_val
        self.train_loader: DataLoader
        self.valid_loader: DataLoader
        self.test_loader: DataLoader

    def shape_data(self, arr: np.ndarray, train: bool = True):
        """Shape raw input streams for ingestion into LSTM or ESN. config.l_s specifies
        the sequence length of prior timesteps fed into the model at each timestep t.

        Args:
            arr (np.ndarray): array of input streams with
                dimensions [timesteps, 1, input dimensions]
            train (bool): If shaping training data, this indicates
                data can be shuffled
        """

        data_tmp: list = []
        for i in range(len(arr) - self.config.l_s - self.config.n_predictions):
            data_tmp.append(arr[i : i + self.config.l_s + self.config.n_predictions])
        data: np.ndarray = np.array(data_tmp)

        assert len(data.shape) == 3

        if train:
            np.random.shuffle(data)
            self.X_train = data[:, : -self.config.n_predictions, :]
            self.y_train = data[
                :, -self.config.n_predictions :, 0
            ]  # telemetry value is at position 0

            if self.train_with_val:
                # Split the dataset into training and validation sets
                # based on the validation_split ratio
                valid_size: int = int(len(self.X_train) * self.config.validation_split)
                train_dataset: TensorDataset = TensorDataset(
                    torch.Tensor(self.X_train[valid_size:]),
                    torch.Tensor(self.y_train[valid_size:]),
                )
                valid_dataset: TensorDataset = TensorDataset(
                    torch.Tensor(self.X_train[:valid_size]),
                    torch.Tensor(self.y_train[:valid_size]),
                )

                # Create DataLoaders for validation sets
                self.valid_loader = DataLoader(
                    valid_dataset, batch_size=self.config.batch_size, shuffle=False
                )
            else:
                train_dataset = TensorDataset(
                    torch.Tensor(self.X_train), torch.Tensor(self.y_train)
                )

            # Create DataLoaders for training sets
            self.train_loader = DataLoader(
                train_dataset, batch_size=self.config.batch_size, shuffle=True
            )

        else:
            self.X_test = data[:, : -self.config.n_predictions, :]
            self.y_test = data[
                :, -self.config.n_predictions :, 0
            ]  # telemetry value is at position 0

            test_dataset: TensorDataset = TensorDataset(
                torch.Tensor(self.X_test), torch.Tensor(self.y_test)
            )
            self.test_loader = DataLoader(
                test_dataset, batch_size=self.config.batch_size, shuffle=False
            )

    def load_data(self):
        """Load train and test data from local."""
        try:
            self.train = np.load(os.path.join("data", "train", f"{self.id}.npy"))
            self.test = np.load(os.path.join("data", "test", f"{self.id}.npy"))

        except FileNotFoundError as e:
            logger.critical(e)
            logger.critical(
                "Source data not found, may need to add data to repo: <link>"
            )

        self.shape_data(self.train)
        self.shape_data(self.test, train=False)
