"""Top-level class for running anomaly detection over a group of channels."""

import csv
import glob
import logging
import os
import time
from typing import (
    Any,
    Dict,
    List,
)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from telemanom.channel import Channel
from telemanom.config import Config
from telemanom.errors import Errors
from telemanom.helpers import (
    evaluate_sequences,
    log_final_stats,
)

logger = logging.getLogger("telemanom")


class Detector:
    """Top-level class for running trained anomaly detection models over a group of
    channels with values stored in .npy files.

    Also evaluates performance against a set of labels.
    """

    def __init__(
        self,
        run_id: str,
        labels_path: str,
        chan_df: pd.DataFrame,
        config: Config,
        result_path: str,
        final_result_path: str,
    ):
        """
        Args:
            run_id (str): The ID for tracking different runs.
            labels_path (str): The path to .csv containing labeled anomaly ranges for
                the group of channels to be processed.
            chan_df (pd.DataFrame): The DataFrame that holds all channel information
                from the labels .csv.
            config (Config): The Channel class object containing train/test data.
            result_path (str): The directory indicating where to store the result .csv.
            final_result_path (str): The path for the final result .csv.
        Attributes:
            run_id (str): see Args.
            labels_path (str): see Args.
            chan_df (pd.DataFrame): see Args.
            config (Config): see Args.
            result_path (str): see Args.
            final_result_path (str): see Args.
            result_tracker (Dict[str, int]): if labels provided, holds results throughout
                processing for logging
        """
        self.id: str = run_id
        self.labels_path: str = labels_path
        self.chan_df: pd.DataFrame = chan_df
        self.config: Config = config
        self.results: List[Dict] = []
        self.result_df: pd.DataFrame
        self.result_path: str = result_path
        self.final_result_path: str = final_result_path
        self.result_tracker: Dict[str, int] = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

    def anomaly_detection(
        self,
    ) -> None:
        """Run anomaly detection on channels and log results."""
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
