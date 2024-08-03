"""Top-level class for running anomaly detection over a group of channels."""

import ast
import csv
import glob
import os
import time
from datetime import datetime as dt
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from telemanom import helpers
from telemanom.channel import Channel
from telemanom.errors import Errors
from telemanom.config import Config
from telemanom.helpers import (
    evaluate_sequences,
    log_final_stats,
)
from telemanom.modeling import Model

logger = helpers.setup_logging()


class Detector:
    """Top-level class for running anomaly detection over a group of channels with."""

    def __init__(
        self,
        labels_path: str,
        result_path: str = "results/",
    ):
        """Top-level class for running anomaly detection over a group of channels with
        values stored in .npy files.

        Also evaluates performance against a set of labels
        if provided.
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
            result_tracker (Dict[str, int]): if labels provided, holds results throughout
                processing for logging
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
        self.result_tracker: Dict[str, int] = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }
        self.config: Config = Config()
        self.y_hat: Optional[np.ndarray] = None
        self.id: str = (
            self.config.model_architecture
            + "_"
            + dt.now().strftime("%Y-%m-%d_%H.%M.%S")
        )
        helpers.make_dirs(self.id)

        # # add logging FileHandler based on ID
        # hdlr: logging.FileHandler = logging.FileHandler(f"data/logs/{self.id}.log")
        # formatter: logging.Formatter = logging.Formatter(
        #     "%(asctime)s %(levelname)s %(message)s"
        # )
        # hdlr.setFormatter(formatter)
        # logger.addHandler(hdlr)
        self.result_path: str = os.path.join(
            result_path, self.config.model_architecture
        )
        if self.labels_path:
            self.chan_df = pd.read_csv(self.labels_path)
        else:
            chan_ids = [x.split(".")[0] for x in os.listdir("./data/test/")]
            self.chan_df = pd.DataFrame({"chan_id": chan_ids})
        logger.info("%s channels found for processing.", len(self.chan_df))

    def run_stage2_retrain(self) -> None:
        """Run retraining on channels and log results."""
        result_stage1_csv_files: List[str] = glob.glob(
            os.path.join(self.result_path, "*.csv")
        )
        print("result_stage1_csv_files", self.result_path, result_stage1_csv_files)
        for csv_files in result_stage1_csv_files:
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

    def run_stage3_anomaly(
        self,
    ) -> None:
        """Run anomaly detection on channels and log results."""
        result_stage1_csv_files: List[str] = glob.glob(
            os.path.join(self.result_path, "*.csv")
        )
        for csv_files in result_stage1_csv_files:
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
                    tot_test_mes: float = 0.0
                    for index, row in self.chan_df.iterrows():
                        logger.info("Stream # %s: %s", index, row["chan_id"])
                        channel = Channel(self.config, row["chan_id"], False)
                        channel.load_data()
                        channel.y_hat = np.load(
                            os.path.join(
                                self.final_result_path,
                                self.id,
                                "y_hat",
                                f"{channel.id}.npy",
                            )
                        )

                        mse: float = float(
                            mean_squared_error(channel.y_test[:, 0], channel.y_hat)
                        )
                        tot_test_mes += mse
                        print(
                            f"Mean Squared Error for channel {row['chan_id']} : {mse} "
                            f"{tot_test_mes}"
                        )

                        errors = Errors(
                            channel, self.config, self.id, self.final_result_path
                        )
                        errors.process_batches(channel)

                        result_row: Dict[str, Any] = {
                            "run_id": self.id,
                            "chan_id": row["chan_id"],
                            "num_train_values": len(channel.x_train),
                            "num_test_values": len(channel.x_test),
                            "n_predicted_anoms": len(errors.e_seq),
                            "normalized_pred_error": errors.normalized,
                            "anom_scores": errors.anom_scores,
                            "test_mse": mse,
                        }

                        if self.labels_path:
                            result_row = {
                                **result_row,
                                **evaluate_sequences(
                                    errors, row, self.result_tracker, logger
                                ),
                            }
                            result_row["spacecraft"] = row["spacecraft"]
                            result_row["anomaly_sequences"] = row["anomaly_sequences"]
                            result_row["class"] = row["class"]
                            self.results.append(result_row)

                            logger.info(
                                "Total true positives: %s",
                                self.result_tracker["true_positives"],
                            )
                            logger.info(
                                "Total false positives: %s",
                                self.result_tracker["false_positives"],
                            )
                            logger.info(
                                "Total false negatives: %s\n",
                                self.result_tracker["false_negatives"],
                            )

                        else:
                            result_row["anomaly_sequences"] = errors.e_seq
                            self.results.append(result_row)

                            logger.info(
                                "%s anomalies found", result_row["n_predicted_anoms"]
                            )
                            logger.info(
                                "anomaly sequences start/end indices: %s",
                                result_row["anomaly_sequences"],
                            )
                            logger.info(
                                "number of test values: %s",
                                result_row["num_test_values"],
                            )
                            logger.info(
                                "anomaly scores: %s\n", result_row["anom_scores"]
                            )

                        self.result_df = pd.DataFrame(self.results)
                        final_result_path = os.path.join(
                            self.final_result_path, self.id, "final_result"
                        )
                        os.makedirs(final_result_path, exist_ok=True)
                        self.result_df.to_csv(
                            os.path.join(final_result_path, f"{self.id}.csv"),
                            index=False,
                        )

                    log_final_stats(
                        logger, self.labels_path, self.result_tracker, self.result_df
                    )
                    end_time: float = time.time()
                    elapsed_time: float = end_time - start_time
                    csv_writer.writerow([self.id, "stage3_anomaly", elapsed_time])
                    print(f"Total Mean Squared Error : {tot_test_mes}")
                    print(f"stage3_anomaly took {elapsed_time:.2f} seconds")
