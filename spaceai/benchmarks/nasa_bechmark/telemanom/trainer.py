"""Top-level class for running anomaly detection over a group of channels."""

import ast
import csv
import glob
import os
import time
from datetime import datetime as dt
from typing import (
    Dict,
    List,
    Optional,
)

import numpy as np
import pandas as pd
from telemanom import helpers
from telemanom.channel import Channel
from telemanom.config import Config
from telemanom.modeling import Model

logger = helpers.setup_logging()


class Trainer:
    """Class for training models on channels."""

    def __init__(
        self,
        labels_path: str,
        result_path: str = "results/",
    ):
        """
        Args:
            labels_path (str): path to .csv containing labeled anomaly ranges
                for group of channels to be processed
            result_path (str): directory indicating where to stick result .csv
            config_path (str): path to config.yaml
        Attributes:
            labels_path (str): see Args
            results (List[Dict]): holds dicts of results for each channel
            result_df (Optional[pd.DataFrame]): results converted to pandas dataframe
            chan_df (Optional[pd.DataFrame]): holds all channel information from labels .csv
            config (Config): Channel class object containing train/test data
            y_hat (Optional[np.ndarray]): predicted channel values
            id (str): datetime id for tracking different runs
            result_path (str): see Args
        """
        self.labels_path: str = labels_path
        self.results: List[Dict] = []
        self.result_df: pd.DataFrame
        self.chan_df: pd.DataFrame
        self.final_result_path: str = "final_results"
        self.config: Config = Config()
        self.y_hat: Optional[np.ndarray] = None
        self.id: str = (
            self.config.model_architecture
            + "_"
            + dt.now().strftime("%Y-%m-%d_%H.%M.%S")
        )
        helpers.make_dirs(self.id)

        self.result_path: str = os.path.join(
            result_path, self.config.model_architecture
        )
        if self.labels_path:
            self.chan_df = pd.read_csv(self.labels_path)
        else:
            chan_ids = [x.split(".")[0] for x in os.listdir("./data/test/")]
            self.chan_df = pd.DataFrame({"chan_id": chan_ids})
        logger.info("%s channels found for processing.", len(self.chan_df))

    def retrain(self) -> None:
        """Run retraining on channels and log results."""
        result_csv_files: List[str] = glob.glob(os.path.join(self.result_path, "*.csv"))
        for csv_files in result_csv_files:
            with open(csv_files, "r", encoding="utf-8") as csvfile:
                self.result_df = pd.read_csv(csvfile)
                self.id = str(self.result_df.iloc[0, 0])
                final_result_path: str = os.path.join(
                    self.final_result_path, self.id, "time_tracking_results"
                )
                os.makedirs(final_result_path, exist_ok=True)
                time_tracking_csv: str = os.path.join(
                    final_result_path, "time_tracking.csv"
                )

                file_exists: bool = os.path.exists(time_tracking_csv)
                with open(
                    time_tracking_csv, mode="a", newline="", encoding="utf-8"
                ) as file:
                    csv_writer = csv.writer(file)

                    # If the file doesn't exist, write the header
                    if not file_exists:
                        csv_writer.writerow(["ID", "Stage", "Time (seconds)"])

                    start_time: float = time.time()
                    for index, row in self.result_df.iterrows():
                        best_hps = ast.literal_eval(row["best_hps"])

                        logger.info("Stream # %s: %s", index, row["chan_id"])
                        channel = Channel(self.config, row["chan_id"], False)
                        channel.load_data()

                        if "ESN" in row["run_id"]:
                            self.config.model_architecture = "ESN"
                            self.config.leakage = best_hps["leakage"]
                            self.config.input_scaling = best_hps["input_scaling"]
                            self.config.rho = best_hps["rho"]
                            self.config.l2 = best_hps["l2"]
                            self.config.layers = [
                                best_hps["hidden_size_1"],
                                best_hps["hidden_size_2"],
                            ]
                        else:  # LSTM
                            self.config.learning_rate = best_hps["lr"]
                            self.config.weight_decay = best_hps["weight_decay"]
                            self.config.layers = [
                                best_hps["hidden_size_1"],
                                best_hps["hidden_size_2"],
                            ]
                            self.config.model_architecture = "LSTM"
                            self.config.dropout = best_hps["dropout"]

                        model = Model(self.config, row["run_id"], channel)
                        model.train_new(channel)
                        channel = model.batch_predict(self.final_result_path, channel)
                    end_time: float = time.time()
                    elapsed_time: float = end_time - start_time
                    csv_writer.writerow([self.id, "stage2_retrain", elapsed_time])
                    print(f"stage2_retrain took {elapsed_time:.2f} seconds")
