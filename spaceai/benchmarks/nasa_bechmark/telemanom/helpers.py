"""Helper functions for clustering channels and setting up logging and directories."""

import ast
import logging
import os
import sys
from typing import (
    Any,
    Dict,
    List,
    Union,
)

import numpy as np
import pandas as pd
from telemanom.errors import Errors


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


def log_final_stats(logger, labels_path, result_tracker, result_df):
    """Log final stats at end of experiment."""
    if labels_path:
        logger.info("Final Totals:")
        logger.info("-----------------")
        logger.info("True Positives: %s", result_tracker["true_positives"])
        logger.info("False Positives: %s", result_tracker["false_positives"])
        logger.info("False Negatives: %s", result_tracker["false_negatives"])
        try:
            precision: float = float(result_tracker["true_positives"]) / (
                float(
                    result_tracker["true_positives"] + result_tracker["false_positives"]
                )
            )
            recall: float = float(result_tracker["true_positives"]) / (
                float(
                    result_tracker["true_positives"] + result_tracker["false_negatives"]
                )
            )
            f1: float = 2 * ((precision * recall) / (precision + recall))
            logger.info("Precision: %.2f", precision)
            logger.info("Recall: %.2f", recall)
            logger.info("F1 Score: %.2f\n", f1)
        except ZeroDivisionError:
            logger.info("Precision: NaN")
            logger.info("Recall: NaN")
            logger.info("F1 Score: NaN\n")
    else:
        logger.info("Final Totals:")
        logger.info("-----------------")
        logger.info("Total channel sets evaluated: %s", len(result_df))
        logger.info("Total anomalies found: %s", result_df["n_predicted_anoms"].sum())
        logger.info(
            "Avg normalized prediction error: %s",
            result_df["normalized_pred_error"].mean(),
        )
        logger.info(
            "Total number of values evaluated: %s\n",
            result_df["num_test_values"].sum(),
        )


def evaluate_sequences(
    errors: Errors,
    label_row: pd.Series,
    result_tracker: Dict[str, int],
    logger: logging.Logger,
) -> Dict[str, Union[int, List]]:
    """Compare identified anomalous sequences with labeled anomalous sequences.

    Args:
        errors (Errors): Errors class object containing detected anomaly
            sequences for a channel
        label_row (pd.Series): Contains labels and true anomaly details
            for a channel
    Returns:
        result_row (Dict[str, Union[int, List]]): anomaly detection accuracy and results
    """
    result_row: Dict[str, Any] = {
        "false_positives": 0,
        "false_negatives": 0,
        "true_positives": 0,
        "fp_sequences": [],
        "tp_sequences": [],
        "num_true_anoms": 0,
    }

    matched_true_seqs: List[int] = []
    label_row["anomaly_sequences"] = ast.literal_eval(label_row["anomaly_sequences"])
    result_row["num_true_anoms"] += len(label_row["anomaly_sequences"])
    result_row["scores"] = errors.anom_scores
    if len(errors.e_seq) == 0:
        result_row["false_negatives"] = result_row["num_true_anoms"]
    else:
        true_indices_grouped: List[List[int]] = [
            list(range(e[0], e[1] + 1)) for e in label_row["anomaly_sequences"]
        ]
        true_indices_flat: set = set(
            [i for group in true_indices_grouped for i in group]
        )
        for e_seq in errors.e_seq:
            i_anom_predicted: set = set(range(e_seq[0], e_seq[1] + 1))
            matched_indices: List[int] = list(i_anom_predicted & true_indices_flat)
            valid: bool = len(matched_indices) > 0
            if valid:
                result_row["tp_sequences"].append(e_seq)
                true_seq_index: List[int] = [
                    i
                    for i in range(len(true_indices_grouped))
                    if len(
                        np.intersect1d(list(i_anom_predicted), true_indices_grouped[i])
                    )
                    > 0
                ]
                if not true_seq_index[0] in matched_true_seqs:
                    matched_true_seqs.append(true_seq_index[0])
                    result_row["true_positives"] += 1
            else:
                result_row["fp_sequences"].append([e_seq[0], e_seq[1]])
                result_row["false_positives"] += 1
        result_row["false_negatives"] = len(
            np.delete(label_row["anomaly_sequences"], matched_true_seqs, axis=0)
        )
    logger.info(
        "Channel Stats: TP: %s  FP: %s  FN: %s",
        result_row["true_positives"],
        result_row["false_positives"],
        result_row["false_negatives"],
    )
    for key, _ in result_row.items():
        if key in result_tracker:
            result_tracker[key] += result_row[key]
    return result_row
