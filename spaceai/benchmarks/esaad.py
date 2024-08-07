# pylint: disable=missing-module-docstring
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from enum import Enum
from glob import glob
from pathlib import Path
from typing import (
    Any,
    Callable,
)

import numpy as np
import pandas as pd
import torch
import wget
from tqdm import tqdm

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
class ESAADMission:
    """ESAAD mission dataclass with metadata of a single mission."""

    index: int
    url_source: str
    dirname: str
    test_data_split: str
    train_data_split: str
    resampling_rule: pd.Timedelta
    monotonic_channel_range: tuple[int, int]

    @property
    def inner_dirpath(self):  # pylint: disable=missing-function-docstring
        return os.path.join(self.dirname, self.dirname)


class ESAADMissions(Enum):
    """ESAAD missions enumeration that contains metadata of mission1 and mission2."""

    MISSION_1: ESAADMission = ESAADMission(
        index=1,
        url_source="https://zenodo.org/records/12528696/files/ESA-Mission1.zip?download=1",
        dirname="ESA-Mission1",
        train_data_split="2007-01-01",
        test_data_split="2007-01-01",
        resampling_rule=pd.Timedelta(seconds=30),
        monotonic_channel_range=(4, 11),
    )
    MISSION_2: ESAADMission = ESAADMission(
        index=2,
        url_source="https://zenodo.org/records/12528696/files/ESA-Mission2.zip?download=1",
        dirname="ESA-Mission2",
        train_data_split="2001-10-01",
        test_data_split="2001-10-01",
        resampling_rule=pd.Timedelta(seconds=18),
        monotonic_channel_range=(29, 46),
    )


class Cache:  # pylint: disable=too-few-public-methods
    """Cache class that stores a portion of loaded data in memory to retrieve them in a
    faster way."""

    def __init__(self, cache_size: int, load_value_function: Callable) -> None:
        self.cache_size = cache_size
        self.load_value_function = load_value_function
        self.cached_keys: list = []
        self.cache_dict: dict = {}

    def load(self, key: str) -> Any:
        """Load the value from the cache or load it from the function and store it in
        the cache.

        Args:
            key (str): The key of the value to load.

        Returns:
            Any: The loaded value.
        """
        if key in self.cached_keys:
            self.cached_keys.remove(key)
        else:
            self.cache_dict[key] = self.load_value_function(key)
        self.cached_keys.append(key)
        if len(self.cached_keys) > self.cache_size:
            del self.cache_dict[self.cached_keys[0]]
            self.cached_keys.pop(0)
        return self.cache_dict[key]


class ESAADBenchmark(
    SpaceBenchmark,
    torch.utils.data.Dataset,
):  # pylint: disable=missing-class-docstring, too-few-public-methods, too-many-instance-attributes

    def __init__(
        self,
        root: str,
        mission_type: ESAADMission,
        is_train: bool,
        window_size: int = 250,
        telecommands_min_priority: int = 3,
        n_buckets: int = 1000,
        cache_size: int = 10,
    ):  # pylint: disable=useless-parent-delegation, too-many-arguments
        """ESAADBenchmark class that preprocesses and loads ESAAD dataset for training
        and testing.

        Args:
            root (str): The root directory of the dataset.
            mission_type (ESAADMission): The mission type of the dataset.
            is_train (bool): The flag that indicates whether the dataset is for training or testing.
            window_size (int): The size of the window for each sample.
            telecommands_min_priority (int): The minimum priority of telecommands to be included.
            n_buckets (int): The number of buckets to divide the dataset.
            cache_size (int): The size of the cache.
        """
        super().__init__()
        self.root = root
        self.mission_type = mission_type
        self.is_train = is_train
        self.window_size = window_size
        self.telecommands_min_priority = telecommands_min_priority
        self.n_buckets = n_buckets

        self.preprocessed_folder = Path(
            os.path.abspath(os.path.join(self.root, "preprocessed"))
        )
        self.preprocessed_folder.mkdir(parents=True, exist_ok=True)
        self.preprocessed_mission_folder = (
            self.preprocessed_folder
            / f'{mission_type.dirname}_{"train" if is_train else "test"}'
        )
        self.preprocessed_mission_folder.mkdir(parents=True, exist_ok=True)
        self.metadata_filepath = self.preprocessed_mission_folder / "metadata.json"
        self.__download_dataset__(
            os.path.join(root, mission_type.dirname), mission_type.url_source
        )
        self.preprocess_dataset()
        with open(self.metadata_filepath, encoding="utf-8") as metadata_file:
            self.metadata = json.load(metadata_file)
        self.cache = Cache(cache_size=cache_size, load_value_function=pd.read_csv)

    def __len__(self):
        return int(np.floor(self.metadata["total_size"] / self.window_size))

    def __getitem__(self, idx):
        start, end = idx * self.window_size, (idx + 1) * self.window_size
        x_data, y_data = [], []
        for bucket_metadata in self.metadata["buckets"]:
            if start < bucket_metadata["end_index"]:
                starti, endi = 0, None
                if start > bucket_metadata["start_index"]:
                    starti = start - bucket_metadata["start_index"]
                if end < bucket_metadata["end_index"]:
                    endi = -(bucket_metadata["end_index"] - end)
                data = self.cache.load(bucket_metadata["filepath"]).drop(
                    columns=["timestamp"]
                )
                x_data.append(data[self.metadata["x_columns"]].iloc[starti:endi].values)
                y_data.append(data[self.metadata["y_columns"]].iloc[starti:endi].values)
            if end < bucket_metadata["end_index"]:
                break

        x_data = torch.from_numpy(np.concatenate(x_data))
        y_data = torch.from_numpy(np.concatenate(y_data))
        return x_data, y_data

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

    def __filter_traintest_and_resampling_rule__(
        self, param_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter the dataframe by train-test split and resample it using zero order
        hold.

        Args:
            param_df (pd.DataFrame): The dataframe to filter and resample.

        Returns:
            pd.DataFrame: The filtered and resampled dataframe.
        """
        resampling_rule = self.mission_type.resampling_rule
        if self.is_train:
            param_df = param_df[
                param_df.index <= pd.to_datetime(self.mission_type.train_data_split)
            ].copy()
        else:
            param_df = param_df[
                param_df.index > pd.to_datetime(self.mission_type.test_data_split)
            ].copy()

        if len(param_df) == 0:
            return []

        # Resample using zero order hold
        first_index_resampled = pd.Timestamp(param_df.index[0]).floor(
            freq=resampling_rule
        )
        last_index_resampled = pd.Timestamp(param_df.index[-1]).ceil(
            freq=resampling_rule
        )
        resampled_range = pd.date_range(
            first_index_resampled, last_index_resampled, freq=resampling_rule
        )
        final_param_df = param_df.reindex(resampled_range, method="ffill")
        # Initialize the first sample
        final_param_df.iloc[0] = param_df.iloc[0]
        return final_param_df

    def __encode_telecommands__(self, param_df: pd.DataFrame) -> pd.DataFrame:
        """Encode telecommands as 0-1 peaks ensuring that they are not removed after
        resampling.

        Args:
            param_df (pd.DataFrame): The telecommands dataframe.

        Returns:
            pd.DataFrame: The telecommands dataframe with encoded telecommands.
        """
        resampling_rule = self.mission_type.resampling_rule
        # Encode telecommands as 0-1 peaks ensuring that they are not removed after resampling
        original_timestamps = param_df.index.copy()
        for timestamp in original_timestamps:
            timestamp_before = timestamp - resampling_rule
            if len(param_df.loc[timestamp_before:timestamp]) == 1:
                param_df.loc[timestamp_before] = 0
                param_df = param_df.sort_index()
            timestamp_after = timestamp + resampling_rule
            if len(param_df.loc[timestamp:timestamp_after]) == 1:
                param_df.loc[timestamp_after] = 0
                param_df = param_df.sort_index()
        return param_df

    def __find_full_time_range__(
        self, params_dict: dict
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Find the full time range of the dataset.

        Args:
            params_dict (dict): The dictionary of dataframes.

        Returns:
            tuple[pd.Timestamp, pd.Timestamp]: The start and end time of the dataset.
        """
        # Find full dataset time range
        start_time = []
        end_time = []
        for df in params_dict.values():
            if len(df) == 0:
                continue
            start_time.append(df.index[0])
            end_time.append(df.index[-1])
        start_time = min(start_time)
        end_time = max(end_time)
        return start_time, end_time

    def preprocess_dataset(
        self,
    ):  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        """Preprocess the dataset by filtering, resampling, and saving it to the
        preprocessed folder."""
        resampling_rule = self.mission_type.resampling_rule
        date_format_str = "%Y-%m-%d %H:%M:%S"
        source_folder = os.path.join(self.root, self.mission_type.inner_dirpath)

        # Load dataframes from files of raw dataset
        labels_df = pd.read_csv(os.path.join(source_folder, "labels.csv"))
        for dcol in ["StartTime", "EndTime"]:
            labels_df[dcol] = pd.to_datetime(labels_df[dcol]).dt.tz_localize(None)
        anomaly_types_df = pd.read_csv(os.path.join(source_folder, "anomaly_types.csv"))
        telecommands_df = pd.read_csv(os.path.join(source_folder, "telecommands.csv"))

        extension = ".zip"
        all_parameter_names = sorted(
            [
                os.path.basename(file)[: -len(extension)]
                for file in glob(
                    os.path.join(source_folder, "channels", f"*{extension}")
                )
            ]
        )

        # Filter telecommands by priority
        telecommands_df = telecommands_df.loc[
            telecommands_df["Priority"] >= self.telecommands_min_priority
        ]
        all_telecommands_names = sorted(telecommands_df.Telecommand.to_list())

        # Initialize anomaly columns
        is_anomaly_columns = [
            f"is_anomaly_{param}"
            for param in all_parameter_names + all_telecommands_names
        ]

        if not self.metadata_filepath.exists():
            params_dict = {}

            # Load and format parameters (channels)
            for param in tqdm(all_parameter_names, desc="Preprocess channels"):
                param_df = pd.read_pickle(
                    os.path.join(source_folder, "channels", f"{param}{extension}")
                )
                param_df["label"] = np.uint8(0)
                param_df = param_df.rename(columns={param: "value"})

                # Take derivative of monotonic channels - part of preprocessing
                min_ch, max_ch = self.mission_type.monotonic_channel_range
                if min_ch <= int(param.split("_")[1]) <= max_ch:
                    param_df.value = np.diff(
                        param_df.value, append=param_df.value.iloc[-1]
                    )

                # Change string values to categorical integers
                if param_df["value"].dtype == "O":
                    param_df["value"] = pd.factorize(param_df["value"])[0]

                # Fill labels
                for _, row in labels_df.iterrows():
                    if row["Channel"] == param:
                        anomaly_type = anomaly_types_df.loc[
                            anomaly_types_df["ID"] == row["ID"]
                        ]["Category"].values[0]
                        end_time = pd.Timestamp(row["EndTime"]).ceil(
                            freq=resampling_rule
                        )
                        if anomaly_type == "Anomaly":
                            label_value = AnnotationLabel.ANOMALY.value
                        elif anomaly_type == "Rare Event":
                            label_value = AnnotationLabel.RARE_EVENT.value
                        elif anomaly_type == "Communication Gap":
                            label_value = AnnotationLabel.GAP.value
                        else:
                            label_value = AnnotationLabel.INVALID.value
                        param_df.loc[row["StartTime"] : end_time, "label"] = label_value

                params_dict[param] = self.__filter_traintest_and_resampling_rule__(
                    param_df
                )

            # Load and format telecommands
            for param in tqdm(all_telecommands_names, desc="Preprocess telecommands"):
                param_df = pd.read_pickle(
                    os.path.join(source_folder, "telecommands", f"{param}{extension}")
                )
                param_df["label"] = np.uint8(0)
                param_df = param_df.rename(columns={param: "value"})

                param_df.index = pd.to_datetime(param_df.index)
                param_df = param_df[~param_df.index.duplicated()]
                param_df = self.__encode_telecommands__(param_df)

                params_dict[param] = self.__filter_traintest_and_resampling_rule__(
                    param_df
                )

            # Initialize final dataframe
            start_time, end_time = self.__find_full_time_range__(params_dict)
            all_params = list(params_dict.keys())
            metadata = {
                "buckets": [],
                "start_time": start_time.strftime(date_format_str),
                "end_time": end_time.strftime(date_format_str),
                "total_size": 0,
                "x_columns": all_params,
                "y_columns": is_anomaly_columns,
            }
            full_index = pd.date_range(start_time, end_time, freq=resampling_rule)
            data_df = pd.DataFrame(index=full_index)

            for param in tqdm(all_params, desc="Create total dataframe"):
                df = params_dict.pop(param)
                df = df.rename(columns={"value": param, "label": f"is_anomaly_{param}"})
                if len(df) == 0:
                    data_df[param] = np.uint8(0)
                    data_df[f"is_anomaly_{param}"] = np.uint8(0)
                    continue
                data_df[df.columns] = df.reindex(data_df.index)
                data_df[param] = data_df[param].astype(np.float64).ffill().bfill()
                data_df[f"is_anomaly_{param}"] = (
                    data_df[f"is_anomaly_{param}"].ffill().bfill().astype(np.uint8)
                )

            data_df["timestamp"] = data_df.index.strftime(date_format_str)
            new_columns_order = [
                "timestamp",
                *all_parameter_names,
                *all_telecommands_names,
                *is_anomaly_columns,
            ]
            data_df = data_df[new_columns_order].reset_index(drop=True)

            bucket_size = np.ceil(len(data_df) / self.n_buckets)
            for n_bucket in tqdm(range(self.n_buckets), desc="Save buckets"):
                df = data_df.loc[n_bucket * bucket_size : (n_bucket + 1) * bucket_size]
                bucket_filepath = (
                    self.preprocessed_mission_folder / f"data_{n_bucket}.csv"
                )
                metadata["buckets"].append(
                    {
                        "filepath": str(bucket_filepath),
                        "size": len(df),
                        "start_index": metadata["total_size"],
                        "end_index": metadata["total_size"] + len(df),
                    }
                )
                metadata["total_size"] += len(df)
                df.to_csv(bucket_filepath, index=False, lineterminator="\n")
            with open(self.metadata_filepath, "w", encoding="utf-8") as metadata_file:
                json.dump(metadata, metadata_file)


if __name__ == "__main__":
    ROOT: str = "./benchmarks"
    if not os.path.exists(ROOT):
        os.mkdir(ROOT)

    datasets = [
        ESAADBenchmark(
            root=ROOT, mission_type=ESAADMissions.MISSION_1.value, is_train=True
        ),
    ]
    """[ ESAADBenchmark(root=ROOT, mission_type=ESAADMissions.MISSION_1.value,
    is_train=False), ESAADBenchmark( root=ROOT,
    mission_type=ESAADMissions.MISSION_2.value, is_train=True), ESAADBenchmark(
    root=ROOT, mission_type=ESAADMissions.MISSION_2.value, is_train=False) ]"""
    for dataset in datasets:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        for i, (x, y) in enumerate(dataloader):
            print(i, x.shape, y.shape)
        break
