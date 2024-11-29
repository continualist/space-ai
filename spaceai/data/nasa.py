import ast
import logging
import math
import os
import tarfile
from typing import (
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import torch

from .anomaly_dataset import AnomalyDataset
from .utils import download_file


class NASA(AnomalyDataset):
    """NASA benchmark dataset for anomaly detection.

    The dataset consists of multivariate time series data collected from NASA's SMAP and
    MSL spacecrafts telemetry data. The data is used to detect anomalies in the
    spacecrafts' telemetry data and evaluate the performance of anomaly detection
    algorithms.
    """

    resource = "https://www.dropbox.com/s/uv9ojw353qwzqht/SMAP.tar.gz?dl=1"

    channel_ids = [
        "A-1",
        "A-2",
        "A-3",
        "A-4",
        "A-5",
        "A-6",
        "A-7",
        "A-8",
        "A-9",
        "B-1",
        "C-1",
        "C-2",
        "D-1",
        "D-11",
        "D-12",
        "D-13",
        "D-14",
        "D-15",
        "D-16",
        "D-2",
        "D-3",
        "D-4",
        "D-5",
        "D-6",
        "D-7",
        "D-8",
        "D-9",
        "E-1",
        "E-10",
        "E-11",
        "E-12",
        "E-13",
        "E-2",
        "E-3",
        "E-4",
        "E-5",
        "E-6",
        "E-7",
        "E-8",
        "E-9",
        "F-1",
        "F-2",
        "F-3",
        "F-4",
        "F-5",
        "F-7",
        "F-8",
        "G-1",
        "G-2",
        "G-3",
        "G-4",
        "G-6",
        "G-7",
        "M-1",
        "M-2",
        "M-3",
        "M-4",
        "M-5",
        "M-6",
        "M-7",
        "P-1",
        "P-10",
        "P-11",
        "P-14",
        "P-15",
        "P-2",
        "P-3",
        "P-4",
        "P-7",
        "R-1",
        "S-1",
        "S-2",
        "T-1",
        "T-10",
        "T-12",
        "T-13",
        "T-2",
        "T-3",
        "T-4",
        "T-5",
        "T-8",
        "T-9",
    ]

    def __init__(
        self,
        root: str,
        channel_id: str,
        mode: Literal["prediction", "anomaly"],
        overlapping: bool = False,
        seq_length: Optional[int] = 250,
        n_predictions: int = 1,
        train: bool = True,
        download: bool = True,
        drop_last: bool = True,
    ):
        """Initialize the dataset for a given channel.

        Args:
            channel_id (str): the ID of the channel to be used

            seq_length (int): the size of the sliding window
            train (bool): whether to use the training or test data
            download (bool): whether to download the dataset
            drop_last (bool): whether to drop the last incomplete sequence
        """
        super().__init__(root)
        if seq_length is None or seq_length < 1:
            raise ValueError(f"Invalid window size: {seq_length}")
        self.channel_id: str = channel_id
        self._mode: Literal["prediction", "anomaly"] = mode
        self.overlapping: bool = overlapping
        self.window_size: int = seq_length if seq_length else 250
        self.train: bool = train
        self.drop_last: bool = drop_last
        self.n_predictions: int = n_predictions

        if not channel_id in self.channel_ids:
            raise ValueError(f"Channel ID {channel_id} is not valid")

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        if self._mode == "anomaly" and self.overlapping:
            logging.warning(
                f"Channel {channel_id} is in anomaly mode and overlapping is set to True."
                " Anomalies will be repeated in the dataset."
            )

        self.data, self.anomalies = self.load_and_preprocess()

    def __getitem__(self, index: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Return the data at the given index."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds")
        first_idx = (
            index
            if self.overlapping
            else index * (self.window_size + self.n_predictions - 1)
        )
        last_idx = first_idx + self.window_size
        if last_idx > len(self.data) - self.n_predictions:
            last_idx = len(self.data) - self.n_predictions

        x, y_true = (
            torch.tensor(self.data[first_idx:last_idx]),
            torch.from_numpy(
                np.stack(
                    [
                        self.data[first_idx + i + 1 : last_idx + i + 1, 0]
                        for i in range(self.n_predictions)
                    ]
                )
            ).T,
        )
        return x, y_true

    def __len__(self) -> int:
        if self.overlapping:
            length = self.data.shape[0] - self.window_size - self.n_predictions + 1
            return length
        length = self.data.shape[0] / (self.window_size + self.n_predictions)
        if self.drop_last:
            return math.floor(length)
        return math.ceil(length)

    def _check_exists(self) -> bool:
        """Check if the dataset exists on the local filesystem."""
        return os.path.exists(os.path.join(self.split_folder, self.channel_id + ".npy"))

    def download(self):
        """Download the dataset.

        This method is called by the constructor by default.
        """

        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)
        tar_filepath = "data.tar.gz"
        download_file(self.resource, to=tar_filepath)
        tar = tarfile.open(tar_filepath, "r:gz")
        tar.extractall(path=self.root)
        tar.close()
        os.remove(tar_filepath)

        nasa_dir = os.path.join(self.root, "NASA")
        data_dir = os.path.join(nasa_dir, "data")
        os.mkdir(nasa_dir)
        os.rename(os.path.join(self.root, "SMAP"), data_dir)
        os.rename(
            os.path.join(data_dir, "labeled_anomalies.csv"),
            os.path.join(data_dir, "test", "anomalies.csv"),
        )

    def load_and_preprocess(self) -> Tuple[torch.Tensor, pd.DataFrame]:
        """Load and preprocess the dataset."""

        data = np.load(
            os.path.join(self.split_folder, f"{self.channel_id}.npy")
        ).astype(np.float32)
        if self._mode == "prediction":
            return data, None

        anomalies: list[list[int]] = []  # Normal by default (train)

        # Load the anomalies for the test data
        if not self.train:
            anomaly_df = pd.read_csv(os.path.join(self.split_folder, "anomalies.csv"))
            anomaly_df = anomaly_df[anomaly_df["chan_id"] == self.channel_id]
            anomaly_seq_df = anomaly_df["anomaly_sequences"]
            if len(anomaly_seq_df) > 0:
                anomalies = ast.literal_eval(anomaly_seq_df.values[0])
            else:
                logging.warning(f"No anomalies found for channel {self.channel_id}")

        return data, anomalies

    @property
    def split_folder(self) -> str:
        """Return the path to the folder containing the split data."""
        return os.path.join(self.raw_folder, "data", "train" if self.train else "test")

    @property
    def in_features_size(self) -> str:
        """Return the size of the input features."""
        return self.data.shape[-1]

    @property
    def mode(self) -> str:
        """Return the mode of the dataset."""
        return self._mode

    @mode.setter
    def mode(self, mode: Literal["prediction", "anomaly"]):
        """Set the mode of the dataset."""
        if mode not in ["prediction", "anomaly"]:
            raise ValueError(f"Invalid mode {mode}")
        self._mode = mode
        self.data, self.anomalies = self.load_and_preprocess()
