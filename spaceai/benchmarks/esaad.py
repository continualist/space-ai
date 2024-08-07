# pylint: disable=missing-module-docstring, too-many-lines
import os
import sys
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wget

from spaceai.benchmarks.benchmark import SpaceBenchmark


class AnnotationLabel(
    Enum
):  # pylint: disable=missing-class-docstring, too-few-public-methods
    """Enuemeration of annotation labels for ESAAD dataset."""

    NOMINAL = 0
    ANOMALY = 1
    RARE_EVENT = 2
    GAP = 3
    INVALID = 4


@dataclass
class ESAADMission:  # pylint: disable=too-many-instance-attributes
    """ESAAD mission dataclass with metadata of a single mission."""

    index: int
    url_source: str
    dirname: str
    train_test_split: str
    resampling_rule: pd.Timedelta
    monotonic_channel_range: tuple[int, int]
    parameters: list[str]
    telecommands: list[str]

    @property
    def inner_dirpath(self):  # pylint: disable=missing-function-docstring
        return os.path.join(self.dirname, self.dirname)

    @property
    def all_channels(self):  # pylint: disable=missing-function-docstring
        return self.parameters + self.telecommands


class ESAADMissions(Enum):
    """ESAAD missions enumeration that contains metadata of mission1 and mission2."""

    MISSION_1: ESAADMission = ESAADMission(
        index=1,
        url_source="https://zenodo.org/records/12528696/files/ESA-Mission1.zip?download=1",
        dirname="ESA-Mission1",
        train_test_split="2007-01-01",
        resampling_rule=pd.Timedelta(seconds=30),
        monotonic_channel_range=(4, 11),
        parameters=[f"channel_{i + 1}" for i in range(76)],
        telecommands=[f"telecommand_{i + 1}" for i in range(698)],
    )
    MISSION_2: ESAADMission = ESAADMission(
        index=2,
        url_source="https://zenodo.org/records/12528696/files/ESA-Mission2.zip?download=1",
        dirname="ESA-Mission2",
        train_test_split="2001-10-01",
        resampling_rule=pd.Timedelta(seconds=18),
        monotonic_channel_range=(29, 46),
        parameters=[f"channel_{i + 1}" for i in range(100)],
        telecommands=[f"telecommand_{i + 1}" for i in range(123)],
    )


class ESAADBenchmark(
    SpaceBenchmark,
    torch.utils.data.Dataset,
):  # pylint: disable=missing-class-docstring, too-few-public-methods, too-many-instance-attributes

    def __init__(
        self,
        root: str,
        mission: ESAADMission,
        channel_id: str,
        train: bool,
        window_size: int = 250,
    ):  # pylint: disable=useless-parent-delegation, too-many-arguments
        """ESAADBenchmark class that preprocesses and loads ESAAD dataset for training
        and testing.

        Args:
            root (str): The root directory of the dataset.
            mission (ESAADMission): The mission type of the dataset.
            channel_id (str): The channel ID to be used.
            train (bool): The flag that indicates whether the dataset is for training or testing.
            window_size (int): The size of the window for each sample.
        """
        super().__init__()
        self.root = root
        self.mission = mission
        self.train = train
        self.channel_id = channel_id
        self.window_size = window_size

        assert channel_id in self.mission.all_channels, "Invalid channel ID"

        self.preprocessed_folder = Path(
            os.path.abspath(os.path.join(self.root, "preprocessed"))
        )
        self.preprocessed_folder.mkdir(parents=True, exist_ok=True)
        self.preprocessed_mission_folder = (
            self.preprocessed_folder
            / f'{mission.dirname}_{"train" if train else "test"}'
        )
        self.preprocessed_mission_folder.mkdir(parents=True, exist_ok=True)
        self.preprocessed_channel_filepath = (
            self.preprocessed_mission_folder / f"{channel_id}.csv"
        )
        self.__download_dataset__(
            os.path.join(root, mission.dirname), mission.url_source
        )
        self.channel_data = self.__preprocess_channel_dataset__(channel_id)

    def __len__(self):
        return len(self.channel_data) // self.window_size

    def __getitem__(self, idx):
        start, end = idx * self.window_size, (idx + 1) * self.window_size
        x_data = self.channel_data.iloc[start:end]["value"].values
        y_data = self.channel_data.iloc[start:end]["label"].values
        return torch.from_numpy(x_data), torch.from_numpy(y_data)

    def __download_dataset__(self, root: str, url: str):
        """Download the dataset from the given URL and extract it to the given
        directory.

        Args:
            root (str): The base directory to save the dataset.
            url (str): The URL of the dataset to download.
        """
        if not os.path.exists(root):

            def bar_progress(current, total):
                progress_message = (
                    f"Downloading: {current / total * 100} [{current} / {total}] bytes"
                )
                sys.stdout.write("\r" + progress_message)
                sys.stdout.flush()

            zipfilepath = f"{root}.zip"
            # ssl._create_default_https_context = ssl._create_unverified_context
            wget.download(url, zipfilepath, bar=bar_progress)
            with zipfile.ZipFile(zipfilepath, "r") as zip_ref:
                zip_ref.extractall(root)
            os.remove(zipfilepath)

    def __filter_train_test__(self, channel_df: pd.DataFrame) -> pd.DataFrame:
        """Filter the dataframe by train-test split.

        Args:
            channel_df (pd.DataFrame): The dataframe to filter.

        Returns:
            pd.DataFrame: The filtered dataframe.
        """
        if self.train:
            mask = channel_df.index <= pd.to_datetime(self.mission.train_test_split)
        else:
            mask = channel_df.index > pd.to_datetime(self.mission.train_test_split)
        return channel_df[mask].copy()

    def __apply_esampling_rule__(self, channel_df: pd.DataFrame) -> pd.DataFrame:
        """Resample the dataframe using zero order hold.

        Args:
            channel_df (pd.DataFrame): The dataframe to resample.

        Returns:
            pd.DataFrame: The resampled dataframe.
        """
        resampling_rule = self.mission.resampling_rule
        # Resample using zero order hold
        first_index_resampled = pd.Timestamp(channel_df.index[0]).floor(
            freq=resampling_rule
        )
        last_index_resampled = pd.Timestamp(channel_df.index[-1]).ceil(
            freq=resampling_rule
        )
        resampled_range = pd.date_range(
            first_index_resampled, last_index_resampled, freq=resampling_rule
        )
        final_param_df = channel_df.reindex(resampled_range, method="ffill")
        # Initialize the first sample
        final_param_df.iloc[0] = channel_df.iloc[0]
        return final_param_df

    def __encode_telecommands__(self, channel_df: pd.DataFrame) -> pd.DataFrame:
        """Encode telecommands as 0-1 peaks ensuring that they are not removed after
        resampling.

        Args:
            channel_df (pd.DataFrame): The telecommands dataframe.

        Returns:
            pd.DataFrame: The telecommands dataframe with encoded telecommands.
        """
        resampling_rule = self.mission.resampling_rule
        # Encode telecommands as 0-1 peaks ensuring that they are not removed after resampling
        original_timestamps = channel_df.index.copy()
        for timestamp in original_timestamps:
            timestamp_before = timestamp - resampling_rule
            if len(channel_df.loc[timestamp_before:timestamp]) == 1:
                channel_df.loc[timestamp_before] = 0
                channel_df = channel_df.sort_index()
            timestamp_after = timestamp + resampling_rule
            if len(channel_df.loc[timestamp:timestamp_after]) == 1:
                channel_df.loc[timestamp_after] = 0
                channel_df = channel_df.sort_index()
        return channel_df

    def __load_parameter_dataset__(
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
                end_time = pd.Timestamp(row["EndTime"]).ceil(
                    freq=self.mission.resampling_rule
                )
                if anomaly_type == "Anomaly":
                    label_value = AnnotationLabel.ANOMALY.value
                elif anomaly_type == "Rare Event":
                    label_value = AnnotationLabel.RARE_EVENT.value
                elif anomaly_type == "Communication Gap":
                    label_value = AnnotationLabel.GAP.value
                else:
                    label_value = AnnotationLabel.INVALID.value
                channel_df.loc[row["StartTime"] : end_time, "label"] = label_value
        return channel_df

    def __load_telecommand_dataset__(
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
        return self.__encode_telecommands__(channel_df)

    def __preprocess_channel_dataset__(
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
                channel_df = self.__load_parameter_dataset__(
                    os.path.join(source_folder, "channels", f"{channel_id}.zip"),
                    channel_id=channel_id,
                    labels_df=labels_df,
                    anomaly_types_df=anomaly_types_df,
                )

            # Load and format telecommand
            if channel_id in self.mission.telecommands:
                channel_df = self.__load_telecommand_dataset__(
                    os.path.join(source_folder, "telecommands", f"{channel_id}.zip"),
                    channel_id=channel_id,
                )

            channel_df = self.__filter_train_test__(channel_df)
            if len(channel_df) == 0:
                return []
            channel_df = self.__apply_esampling_rule__(channel_df)

            channel_df["value"] = channel_df["value"].astype(np.float64).ffill().bfill()
            channel_df["label"] = channel_df["label"].ffill().bfill().astype(np.uint8)
            channel_df.to_csv(
                self.preprocessed_channel_filepath, index=False, lineterminator="\n"
            )
            return channel_df
        with open(self.preprocessed_channel_filepath, encoding="utf-8") as fp:
            return pd.read_csv(fp)


if __name__ == "__main__":
    ROOT: str = "./benchmarks"
    if not os.path.exists(ROOT):
        os.mkdir(ROOT)

    for mission_metadata in [
        ESAADMissions.MISSION_1.value,
        ESAADMissions.MISSION_2.value,
    ]:
        for is_train in [True, False]:
            dataset = ESAADBenchmark(
                root=ROOT,
                mission=mission_metadata,
                channel_id="channel_12",
                train=is_train,
                window_size=250,
            )
            print(f"{dataset.mission.dirname} Dataset length:", len(dataset))
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=64, shuffle=True
            )
            for i, (x, y) in enumerate(dataloader):
                print(i, x.shape, y.shape)
