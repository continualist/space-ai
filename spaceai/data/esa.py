# pylint: disable=missing-module-docstring, too-many-lines
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import torch

from spaceai.utils.tools import download_and_extract_zip

from .anomaly_dataset import AnomalyDataset


class AnnotationLabel(
    Enum
):  # pylint: disable=missing-class-docstring, too-few-public-methods
    """Enuemeration of annotation labels for ESA dataset."""

    NOMINAL = 0
    ANOMALY = 1
    RARE_EVENT = 2
    GAP = 3
    INVALID = 4


@dataclass
class ESAMission:  # pylint: disable=too-many-instance-attributes
    """ESA mission dataclass with metadata of a single mission."""

    index: int
    """The index of the mission."""
    url_source: str
    """The URL source of the mission data."""
    dirname: str
    """The directory name of the mission data."""
    train_test_split: pd.Timestamp
    """The split date between training and testing data."""
    start_date: pd.Timestamp
    """The start date of the mission."""
    end_date: pd.Timestamp
    """The end date of the mission."""
    resampling_rule: pd.Timedelta
    """The resampling rule for the data."""
    monotonic_channel_range: tuple[int, int]
    """The range of monotonic channels."""
    parameters: list[str]
    """The list of parameters."""
    telecommands: list[str]
    """The list of telecommands."""
    target_channels: list[str]
    """The list of target channels."""

    @property
    def inner_dirpath(self):  # pylint: disable=missing-function-docstring
        return os.path.join(self.dirname, self.dirname)

    @property
    def all_channels(self):  # pylint: disable=missing-function-docstring
        return self.parameters + self.telecommands


class ESAMissions(Enum):
    """ESA missions enumeration that contains metadata of mission1 and mission2."""

    MISSION_1: ESAMission = ESAMission(
        index=1,
        url_source="https://zenodo.org/records/12528696/files/ESA-Mission1.zip?download=1",
        dirname="ESA-Mission1",
        train_test_split=pd.to_datetime("2007-01-01"),
        start_date=pd.to_datetime("2000-01-01"),
        end_date=pd.to_datetime("2014-01-01"),
        resampling_rule=pd.Timedelta(seconds=30),
        monotonic_channel_range=(4, 11),
        parameters=[f"channel_{i + 1}" for i in range(76)],
        telecommands=[f"telecommand_{i + 1}" for i in range(698)],
        target_channels=[
            f"channel_{i}"
            for i in [*list(range(12, 53)), *list(range(57, 67)), *list(range(70, 77))]
        ],
    )
    MISSION_2: ESAMission = ESAMission(
        index=2,
        url_source="https://zenodo.org/records/12528696/files/ESA-Mission2.zip?download=1",
        dirname="ESA-Mission2",
        train_test_split=pd.to_datetime("2001-10-01"),
        start_date=pd.to_datetime("2000-01-01"),
        end_date=pd.to_datetime("2003-07-01"),
        resampling_rule=pd.Timedelta(seconds=18),
        monotonic_channel_range=(29, 46),
        parameters=[f"channel_{i + 1}" for i in range(100)],
        telecommands=[f"telecommand_{i + 1}" for i in range(123)],
        target_channels=[
            f"channel_{i}"
            for i in [
                *list(range(9, 29)),
                *list(range(58, 60)),
                *list(range(70, 92)),
                *list(range(96, 99)),
            ]
        ],
    )


class ESA(
    AnomalyDataset,
):  # pylint: disable=too-few-public-methods, too-many-instance-attributes
    """ESA benchmark dataset for anomaly detection.

    The dataset consists of multivariate time series data collected from ESA's
    spacecrafts telemetry data. The data is used to detect anomalies in the spacecrafts'
    telemetry data and evaluate the performance of anomaly detection algorithms.
    """

    def __init__(
        self,
        root: str,
        mission: ESAMission,
        channel_id: str,
        mode: Literal["prediction", "anomaly"],
        overlapping: bool = False,
        seq_length: Optional[int] = 250,
        train: bool = True,
        download: bool = True,
        uniform_start_end_date: bool = False,
    ):  # pylint: disable=useless-parent-delegation, too-many-arguments
        """ESABenchmark class that preprocesses and loads ESA dataset for training and
        testing.

        Args:
            root (str): The root directory of the dataset.
            mission (ESAMission): The mission type of the dataset.
            channel_id (str): The channel ID to be used.
            mode (Literal["prediction", "anomaly"]): The mode of the dataset.
            overlapping (bool): The flag that indicates whether the dataset is overlapping.
            seq_length (Optional[int]): The length of the sequence for each sample.
            train (bool): The flag that indicates whether the dataset is for training or testing.
            download (bool): The flag that indicates whether the dataset should be downloaded.
            uniform_start_end_date (bool): The flag that indicates whether the dataset should be resampled to have uniform start and end date.
        """
        super().__init__(root)
        if seq_length is None or seq_length < 1:
            raise ValueError(f"Invalid window size: {seq_length}")

        if mode not in ["prediction", "anomaly"]:
            raise ValueError(f"Invalid mode {mode}")

        self.root = root
        self.mission = mission
        self.channel_id: str = channel_id
        self._mode: Literal["prediction", "anomaly"] = mode
        self.overlapping: bool = overlapping
        self.window_size: int = seq_length if seq_length else 250
        self.train: bool = train
        self.uniform_start_end_date: bool = uniform_start_end_date

        if not channel_id in self.mission.all_channels:
            raise ValueError(f"Channel ID {channel_id} is not valid")

        train_test_label = "train" if train else "test"
        self.preprocessed_channel_filepath = Path(
            os.path.abspath(
                os.path.join(
                    self.root,
                    "preprocessed",
                    mission.dirname,
                    f"{channel_id}_{train_test_label}.csv",
                )
            )
        )
        os.makedirs(os.path.dirname(self.preprocessed_channel_filepath), exist_ok=True)

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

        self.data, self.anomalies = self.load_and_preprocess(channel_id)

    def __getitem__(self, index: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Return the data at the given index."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds")
        first_idx = index if self.overlapping else index * self.window_size
        if first_idx + self.window_size + 1 > len(self.data):
            first_idx -= 1

        x, y_true = (
            torch.tensor(self.data[first_idx : first_idx + self.window_size]),
            torch.tensor(self.data[first_idx + 1 : first_idx + self.window_size + 1]),
        )
        if self._mode == "prediction":
            return x.unsqueeze(1), y_true.unsqueeze(1)
        anomalies = torch.tensor(
            self.anomalies[first_idx + 1 : first_idx + self.window_size + 1]
        ).int()
        return x.unsqueeze(1), y_true.unsqueeze(1), anomalies.unsqueeze(1)

    def __len__(self) -> int:
        length = self.data.shape[0] - self.window_size - 1
        if self.overlapping:
            return length
        return length // self.window_size

    def download(self):
        """Download the dataset from the given URL and extract it to the given
        directory."""
        if self._check_exists():
            return
        download_and_extract_zip(
            self.mission.url_source,
            os.path.join(self.root, self.mission.dirname),
            cleanup=True,
        )

    def _check_exists(self) -> bool:
        """Check if the dataset exists on the local filesystem."""
        return os.path.exists(os.path.join(self.root, self.mission.dirname))

    def _filter_train_test_(self, channel_df: pd.DataFrame) -> pd.DataFrame:
        """Filter the dataframe by train-test split.

        Args:
            channel_df (pd.DataFrame): The dataframe to filter.

        Returns:
            pd.DataFrame: The filtered dataframe.
        """
        if self.train:
            mask = channel_df.index <= self.mission.train_test_split
        else:
            mask = channel_df.index > self.mission.train_test_split
        return channel_df[mask].copy()

    def _apply_resampling_rule_(self, channel_df: pd.DataFrame) -> pd.DataFrame:
        """Resample the dataframe using zero order hold.

        Args:
            channel_df (pd.DataFrame): The dataframe to resample.

        Returns:
            pd.DataFrame: The resampled dataframe.
        """
        # Resample using zero order hold
        if self.uniform_start_end_date:
            start_date, end_date = self.mission.start_date, self.mission.end_date
        else:
            start_date, end_date = channel_df.index[0], channel_df.index[-1]
        first_index_resampled = pd.Timestamp(start_date).floor(
            freq=self.mission.resampling_rule
        )
        last_index_resampled = pd.Timestamp(end_date).ceil(
            freq=self.mission.resampling_rule
        )
        resampled_range = pd.date_range(
            first_index_resampled,
            last_index_resampled,
            freq=self.mission.resampling_rule,
        )
        final_param_df = channel_df.reindex(resampled_range, method="ffill")
        # Initialize the first sample
        final_param_df.iloc[0] = channel_df.iloc[0]
        return final_param_df

    def _load_parameter_dataset_(
        self,
        filepath: str,
        channel_id: str,
        labels_df: pd.DataFrame,
        anomaly_types_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Load and preprocess the parameter dataset.

        Args:
            filepath (str): The path to the parameter dataset.
            channel_id (str): The channel ID.
            labels_df (pd.DataFrame): The labels dataframe.
            anomaly_types_df (pd.DataFrame): The anomaly types dataframe.

        Returns:
            pd.DataFrame: The preprocessed parameter dataset.
        """
        channel_df = pd.read_pickle(filepath)
        channel_df["label"] = np.uint8(0)
        channel_df = channel_df.rename(columns={channel_id: "value"})

        # Take derivative of monotonic channels - part of preprocessing
        min_ch, max_ch = self.mission.monotonic_channel_range
        if min_ch <= int(channel_id.split("_")[1]) <= max_ch:
            channel_df.value = np.diff(
                channel_df.value, append=channel_df.value.iloc[-1]
            )

        # Change string values to categorical integers
        if channel_df["value"].dtype == "O":
            channel_df["value"] = pd.factorize(channel_df["value"])[0]

        # Fill labels
        for _, row in labels_df.iterrows():
            if row["Channel"] == channel_id:
                anomaly_type = anomaly_types_df.loc[
                    anomaly_types_df["ID"] == row["ID"]
                ]["Category"].values[0]
                if anomaly_type == "Anomaly":
                    label_value = AnnotationLabel.ANOMALY.value
                elif anomaly_type == "Rare Event":
                    label_value = AnnotationLabel.RARE_EVENT.value
                elif anomaly_type == "Communication Gap":
                    label_value = AnnotationLabel.GAP.value
                else:
                    label_value = AnnotationLabel.INVALID.value
                end_time = pd.Timestamp(row["EndTime"]).ceil(
                    freq=self.mission.resampling_rule
                )
                channel_df.loc[row["StartTime"] : end_time, "label"] = label_value
        return channel_df

    def _load_telecommand_dataset_(
        self, filepath: str, channel_id: str
    ) -> pd.DataFrame:
        """Load and preprocess the telecommand dataset.

        Args:
            filepath (str): The path to the telecommand dataset.
            channel_id (str): The channel ID.

        Returns:
            pd.DataFrame: The preprocessed telecommand dataset.
        """
        channel_df = pd.read_pickle(filepath)
        channel_df["label"] = np.uint8(0)
        channel_df = channel_df.rename(columns={channel_id: "value"})
        channel_df.index = pd.to_datetime(channel_df.index)
        channel_df = channel_df[~channel_df.index.duplicated()]

        # Encode telecommands as 0-1 peaks ensuring that they are not removed after resampling
        original_timestamps = channel_df.index.copy()
        for timestamp in original_timestamps:
            timestamp_before = timestamp - self.mission.resampling_rule
            if len(channel_df.loc[timestamp_before:timestamp]) == 1:
                channel_df.loc[timestamp_before] = 0
                channel_df = channel_df.sort_index()
            timestamp_after = timestamp + self.mission.resampling_rule
            if len(channel_df.loc[timestamp:timestamp_after]) == 1:
                channel_df.loc[timestamp_after] = 0
                channel_df = channel_df.sort_index()
        return channel_df

    def load_and_preprocess(
        self,
        channel_id: str,
    ) -> pd.DataFrame:
        """Preprocess the channel dataset by loading the raw channel dataset.

        Args:
            channel_id (str): The channel ID to preprocess.

        Returns:
            pd.DataFrame: The preprocessed channel dataset.
        """
        source_folder = os.path.join(self.root, self.mission.inner_dirpath)

        # Load dataframes from files of raw dataset
        labels_df = pd.read_csv(os.path.join(source_folder, "labels.csv"))
        for dcol in ["StartTime", "EndTime"]:
            labels_df[dcol] = pd.to_datetime(labels_df[dcol]).dt.tz_localize(None)

        anomaly_types_df = pd.read_csv(os.path.join(source_folder, "anomaly_types.csv"))

        if not self.preprocessed_channel_filepath.exists():
            # Load and format parameter (channel)
            if channel_id in self.mission.parameters:
                channel_df = self._load_parameter_dataset_(
                    os.path.join(source_folder, "channels", f"{channel_id}.zip"),
                    channel_id=channel_id,
                    labels_df=labels_df,
                    anomaly_types_df=anomaly_types_df,
                )

            # Load and format telecommand
            if channel_id in self.mission.telecommands:
                channel_df = self._load_telecommand_dataset_(
                    os.path.join(source_folder, "telecommands", f"{channel_id}.zip"),
                    channel_id=channel_id,
                )

            channel_df = self._filter_train_test_(channel_df)
            if len(channel_df) == 0:
                return []
            channel_df = self._apply_resampling_rule_(channel_df)

            channel_df["value"] = channel_df["value"].ffill().bfill().astype(np.float32)
            channel_df["label"] = channel_df["label"].ffill().bfill().astype(np.uint8)
            channel_df.to_csv(
                self.preprocessed_channel_filepath, index=False, lineterminator="\n"
            )
        else:
            with open(self.preprocessed_channel_filepath, encoding="utf-8") as fp:
                channel_df = pd.read_csv(fp)
        return channel_df["value"].values.astype(np.float32), channel_df[
            "label"
        ].values.astype(np.uint8)
