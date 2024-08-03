"""Error class for calculating and scoring prediction errors."""

import logging
import os
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
)

import more_itertools as mit
import numpy as np
import pandas as pd
from telemanom.channel import Channel
from telemanom.config import Config

logger = logging.getLogger("telemanom")


class Errors:
    """Batch processing of errors between actual and predicted values for a channel."""

    def __init__(self, channel: Channel, config: Config, run_id: str, result_path: str):
        """Batch processing of errors between actual and predicted values for a channel.

        Args:
            channel (obj): Channel class object containing train/test data
            for X,y for a single channel
            config (obj): Config object containing parameters for processing
            run_id (str): Datetime referencing set of predictions in use
            result_path (str): Path to save the results

        Attributes:
            config (obj): see Args
            window_size (int): number of trailing batches to use in error calculation
            n_windows (int): number of windows in test values for channel
            i_anom (np.ndarray): indices of anomalies in channel
            test values
            E_seq (List[Tuple[int, int]]): array of (start, end) indices for
            each continuous anomaly sequence in test values
            anom_scores (List[dict]): score indicating relative severity of
            each anomaly sequence in E_seq
            e (List[float]): errors in prediction (predicted - actual)
            e_s (np.ndarray): exponentially-smoothed errors in prediction
            normalized (float): prediction errors as a percentage of the range of the channel values
        """
        self.config = config
        self.window_size: int = self.config.window_size
        self.n_windows: int = int(
            (channel.y_test.shape[0] - (self.config.batch_size * self.window_size))
            / self.config.batch_size
        )
        self.i_anom: np.ndarray = np.array([])
        self.e_seq: List[Tuple[int, int]] = []
        self.anom_scores: List[dict] = []
        self.run_id: str = run_id
        self.result_path: str = result_path
        # raw prediction error
        self.e: List[float] = [
            abs(y_h - y_t[0]) for y_h, y_t in zip(channel.y_hat, channel.y_test)
        ]
        smoothing_window: int = int(
            self.config.batch_size
            * self.config.window_size
            * self.config.smoothing_perc
        )
        if not len(channel.y_hat) == len(channel.y_test):
            raise ValueError(
                f"len(y_hat) != len(y_test): {len(channel.y_hat)}, {len(channel.y_test)}",
                len(channel.y_hat),
                len(channel.y_test),
            )
        # smoothed prediction error
        self.e_s: np.ndarray = (
            pd.DataFrame(self.e).ewm(span=smoothing_window).mean().values.flatten()
        )
        # for values at beginning < sequence length, just use avg
        if not channel.id == "C-2":  # anomaly occurs early in window
            self.e_s[: self.config.l_s] = [
                np.mean(self.e_s[: self.config.l_s * 2])
            ] * self.config.l_s
        result_path = os.path.join(self.result_path, self.run_id, "smoothed_errors")
        os.makedirs(result_path, exist_ok=True)
        np.save(os.path.join(result_path, f"{channel.id}.npy"), np.array(self.e_s))
        self.normalized: float = np.mean(self.e / np.ptp(channel.y_test))
        logger.info("normalized prediction error: %.2f", self.normalized)

    def adjust_window_size(self, channel):
        """Decrease the historical error window size (h) if number of test values is
        limited.

        Args:
            channel (obj): Channel class object containing train/test data
            for X,y for a single channel
        """
        while self.n_windows < 0:
            self.window_size -= 1
            self.n_windows = int(
                (channel.y_test.shape[0] - (self.config.batch_size * self.window_size))
                / self.config.batch_size
            )
            if self.window_size == 1 and self.n_windows < 0:
                raise ValueError(
                    f"Batch_size ({self.config.batch_size}) larger than y_test "
                    f"(len={channel.y_test.shape[0]}). Adjust in config.yaml.",
                    self.config.batch_size,
                    channel.y_test.shape[0],
                )

    def merge_scores(self):
        """If anomalous sequences from subsequent batches are adjacent they will
        automatically be combined.

        This combines the scores for these initial adjacent sequences (scores are
        calculated as each batch is processed) where applicable.
        """
        merged_scores: List[float] = []
        score_end_indices: List[int] = []

        for _, score in enumerate(self.anom_scores):
            if not score["start_idx"] - 1 in score_end_indices:
                merged_scores.append(score["score"])
                score_end_indices.append(score["end_idx"])

    def process_batches(self, channel):
        """Top-level function for the Error class that loops through batches of values
        for a channel.

        Args:
            channel (obj): Channel class object containing train/test data
            for X,y for a single channel
        """
        self.adjust_window_size(channel)
        for i in range(0, self.n_windows + 1):
            prior_idx: int = i * self.config.batch_size
            idx: int = (self.config.window_size * self.config.batch_size) + (
                i * self.config.batch_size
            )
            if i == self.n_windows:
                idx = channel.y_test.shape[0]
            window = ErrorWindow(channel, self.config, prior_idx, idx, self, i)
            window.find_epsilon()
            window.find_epsilon(inverse=True)
            window.compare_to_epsilon(self)
            window.compare_to_epsilon(self, inverse=True)
            if len(window.i_anom) == 0 and len(window.i_anom_inv) == 0:
                continue
            window.prune_anoms()
            window.prune_anoms(inverse=True)
            if len(window.i_anom) == 0 and len(window.i_anom_inv) == 0:
                continue
            window.i_anom = np.sort(
                np.unique(np.append(window.i_anom, window.i_anom_inv))
            ).astype("int")
            window.score_anomalies(prior_idx)
            # update indices to reflect true indices in full set of values
            self.i_anom = np.append(self.i_anom, window.i_anom + prior_idx)
            self.anom_scores = self.anom_scores + window.anom_scores
        if len(self.i_anom) > 0:
            # group anomalous indices into continuous sequences
            groups: List[List[int]] = [
                list(group) for group in mit.consecutive_groups(self.i_anom)
            ]
            self.e_seq: List[Tuple[int, int]] = [
                (int(g[0]), int(g[-1])) for g in groups if not g[0] == g[-1]
            ]
            # additional shift is applied to indices so that they represent the
            # position in the original data array, obtained from the .npy files,
            # and not the position on y_test (See PR #27).
            self.e_seq = [
                (e_seq[0] + self.config.l_s, e_seq[1] + self.config.l_s)
                for e_seq in self.e_seq
            ]
            self.merge_scores()


class ErrorWindow:
    """Data and calculations for a specific window of prediction errors."""

    def __init__(
        self, channel, config, start_idx: int, end_idx: int, errors, window_num: int
    ):
        """Data and calculations for a specific window of prediction errors.

        Includes finding thresholds, pruning, and scoring anomalous sequences
        for errors and inverted errors (flipped around mean) - significant drops
        in values can also be anomalous.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
            config (obj): Config object containing parameters for processing
            start_idx (int): Starting index for window within full set of
                channel test values
            end_idx (int): Ending index for window within full set of channel
                test values
            errors (arr): Errors class object
            window_num (int): Current window number within channel test values

        Attributes:
            i_anom (arr): indices of anomalies in window
            i_anom_inv (arr): indices of anomalies in window of inverted
                telemetry values
            E_seq (arr of tuples): array of (start, end) indices for each
                continuous anomaly sequence in window
            E_seq_inv (arr of tuples): array of (start, end) indices for each
                continuous anomaly sequence in window of inverted telemetry
                values
            non_anom_max (float): highest smoothed error value below epsilon
            non_anom_max_inv (float): highest smoothed error value below
                epsilon_inv
            config (obj): see Args
            anom_scores (arr): score indicating relative severity of each
                anomaly sequence in E_seq within a window
            window_num (int): see Args
            sd_lim (int): default number of standard deviations to use for
                threshold if no winner or too many anomalous ranges when scoring
                candidate thresholds
            sd_threshold (float): number of standard deviations for calculation
                of best anomaly threshold
            sd_threshold_inv (float): same as above for inverted channel values
            e_s (arr): exponentially-smoothed prediction errors in window
            e_s_inv (arr): inverted e_s
            sd_e_s (float): standard deviation of e_s
            mean_e_s (float): mean of e_s
            epsilon (float): threshold for e_s above which an error is
                considered anomalous
            epsilon_inv (float): threshold for inverted e_s above which an error
                is considered anomalous
            y_test (arr): Actual telemetry values for window
            sd_values (float): st dev of y_test
            perc_high (float): the 95th percentile of y_test values
            perc_low (float): the 5th percentile of y_test values
            inter_range (float): the range between perc_high - perc_low
            num_to_ignore (int): number of values to ignore initially when
                looking for anomalies
        """

        self.i_anom: np.ndarray = np.array([])
        self.e_seq: List[Tuple[int, int]] = []
        self.non_anom_max: float = -1000000
        self.i_anom_inv: np.ndarray = np.array([])
        self.e_seq_inv: List[Tuple[int, int]] = []
        self.non_anom_max_inv: float = -1000000

        self.config = config
        self.anom_scores: List[dict] = []

        self.window_num: int = window_num

        self.sd_lim: float = 12.0
        self.sd_threshold: float = self.sd_lim
        self.sd_threshold_inv: float = self.sd_lim

        self.e_s: np.ndarray = errors.e_s[start_idx:end_idx]

        self.mean_e_s: float = float(np.mean(self.e_s))
        self.sd_e_s: float = float(np.std(self.e_s))
        self.e_s_inv: np.ndarray = np.array(
            [self.mean_e_s + (self.mean_e_s - e) for e in self.e_s]
        )

        self.epsilon: float = self.mean_e_s + self.sd_lim * self.sd_e_s
        self.epsilon_inv: float = self.mean_e_s + self.sd_lim * self.sd_e_s

        self.y_test: np.ndarray = channel.y_test[start_idx:end_idx]
        self.sd_values: float = float(np.std(self.y_test))

        self.perc_high: float
        self.perc_low: float
        self.perc_high, self.perc_low = np.percentile(self.y_test, [95, 5])
        self.inter_range: float = self.perc_high - self.perc_low

        # ignore initial error values until enough history for processing
        self.num_to_ignore: int = self.config.l_s * 2
        # if y_test is small, ignore fewer
        if len(channel.y_test) < 2500:
            self.num_to_ignore = self.config.l_s
        if len(channel.y_test) < 1800:
            self.num_to_ignore = 0

    def find_epsilon(self, inverse: bool = False) -> None:
        """
        Find the anomaly threshold that maximizes function representing
        tradeoff between:
            a) number of anomalies and anomalous ranges
            b) the reduction in mean and st dev if anomalous points are removed
            from errors
        (see https://arxiv.org/pdf/1802.04431.pdf)

        Args:
            inverse (bool): If true, epsilon is calculated for inverted errors
        """
        e_s: np.ndarray = self.e_s if not inverse else self.e_s_inv

        max_score: float = -10000000

        for z in np.arange(2.5, self.sd_lim, 0.5):
            epsilon: float = self.mean_e_s + (self.sd_e_s * z)

            pruned_e_s: np.ndarray = e_s[e_s < epsilon]

            i_anom: np.ndarray = np.argwhere(e_s >= epsilon).reshape(
                -1,
            )
            buffer: np.ndarray = np.arange(1, self.config.error_buffer)
            i_anom = np.sort(
                np.concatenate(
                    (
                        i_anom,
                        np.array([i + buffer for i in i_anom]).flatten(),
                        np.array([i - buffer for i in i_anom]).flatten(),
                    )
                )
            )
            i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
            i_anom = np.sort(np.unique(i_anom))

            if len(i_anom) > 0:
                # group anomalous indices into continuous sequences
                groups: List[List[int]] = [
                    list(group) for group in mit.consecutive_groups(i_anom)
                ]
                e_seq: List[Tuple[int, int]] = [
                    (g[0], g[-1]) for g in groups if not g[0] == g[-1]
                ]

                mean_perc_decrease: float = float(
                    (self.mean_e_s - np.mean(pruned_e_s)) / self.mean_e_s
                )
                sd_perc_decrease: float = float(
                    (self.sd_e_s - np.std(pruned_e_s)) / self.sd_e_s
                )
                score: float = (mean_perc_decrease + sd_perc_decrease) / (
                    len(e_seq) ** 2 + len(i_anom)
                )

                # sanity checks / guardrails
                if (
                    score >= max_score
                    and len(e_seq) <= 5
                    and len(i_anom) < (len(e_s) * 0.5)
                ):
                    max_score = score
                    if not inverse:
                        self.sd_threshold = z
                        self.epsilon = self.mean_e_s + z * self.sd_e_s
                    else:
                        self.sd_threshold_inv = z
                        self.epsilon_inv = self.mean_e_s + z * self.sd_e_s

    def compare_to_epsilon(self, errors_all: "Errors", inverse: bool = False) -> None:
        """Compare smoothed error values to epsilon (error threshold) and group
        consecutive errors together into sequences.

        Args:
            errors_all (Errors): Errors class object containing list of all
            previously identified anomalies in test set
        """

        e_s: np.ndarray = self.e_s if not inverse else self.e_s_inv
        epsilon: float = self.epsilon if not inverse else self.epsilon_inv

        # Check: scale of errors compared to values too small?
        if (
            not (
                self.sd_e_s > (0.05 * self.sd_values)
                or max(self.e_s) > (0.05 * self.inter_range)
            )
            or not max(self.e_s) > 0.05
        ):
            return

        i_anom: np.ndarray = np.argwhere(
            (e_s >= epsilon) & (e_s > 0.05 * self.inter_range)
        ).reshape(
            -1,
        )

        if len(i_anom) == 0:
            return
        buffer: np.ndarray = np.arange(1, self.config.error_buffer + 1)
        i_anom = np.sort(
            np.concatenate(
                (
                    i_anom,
                    np.array([i + buffer for i in i_anom]).flatten(),
                    np.array([i - buffer for i in i_anom]).flatten(),
                )
            )
        )
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]

        # if it is first window, ignore initial errors (need some history)
        if self.window_num == 0:
            i_anom = i_anom[i_anom >= self.num_to_ignore]
        else:
            i_anom = i_anom[i_anom >= len(e_s) - self.config.batch_size]

        i_anom = np.sort(np.unique(i_anom))

        # capture max of non-anomalous values below the threshold
        # (used in filtering process)
        batch_position: int = self.window_num * self.config.batch_size
        window_indices: np.ndarray = np.arange(0, len(e_s)) + batch_position
        adj_i_anom: np.ndarray = i_anom + batch_position
        window_indices = np.setdiff1d(
            window_indices, np.append(errors_all.i_anom, adj_i_anom)
        )
        candidate_indices: np.ndarray = np.unique(window_indices - batch_position)
        non_anom_max: float = np.max(np.take(e_s, candidate_indices))

        # group anomalous indices into continuous sequences
        groups: List[List[int]] = [
            list(group) for group in mit.consecutive_groups(i_anom)
        ]
        e_seq: List[Tuple[int, int]] = [
            (g[0], g[-1]) for g in groups if not g[0] == g[-1]
        ]

        if inverse:
            self.i_anom_inv = i_anom
            self.e_seq_inv = e_seq
            self.non_anom_max_inv = non_anom_max
        else:
            self.i_anom = i_anom
            self.e_seq = e_seq
            self.non_anom_max = non_anom_max

    def prune_anoms(self, inverse: bool = False) -> None:
        """Remove anomalies that don't meet minimum separation from the next closest
        anomaly or error value.

        Args:
            inverse (bool): If true, epsilon is calculated for inverted errors
        """

        e_seq: Union[List[Tuple[int, int]], np.ndarray] = (
            self.e_seq if not inverse else self.e_seq_inv
        )
        e_s: np.ndarray = self.e_s if not inverse else self.e_s_inv
        non_anom_max: float = (
            self.non_anom_max if not inverse else self.non_anom_max_inv
        )

        if len(e_seq) == 0:
            return

        e_seq_max: np.ndarray = np.array([max(e_s[e[0] : e[1] + 1]) for e in e_seq])
        e_seq_max_sorted: np.ndarray = np.sort(e_seq_max)[::-1]
        e_seq_max_sorted = np.append(e_seq_max_sorted, [non_anom_max])

        i_to_remove: np.ndarray = np.array([])
        for i in range(0, len(e_seq_max_sorted) - 1):
            if (e_seq_max_sorted[i] - e_seq_max_sorted[i + 1]) / e_seq_max_sorted[
                i
            ] < self.config.p:
                i_to_remove = np.append(
                    i_to_remove, np.argwhere(e_seq_max == e_seq_max_sorted[i])
                )
            else:
                i_to_remove = np.array([])
        i_to_remove[::-1].sort()

        i_to_remove = np.array(i_to_remove, dtype=int)
        if len(i_to_remove) > 0:
            e_seq = np.delete(e_seq, i_to_remove, axis=0)

        if len(e_seq) == 0 and inverse:
            self.i_anom_inv = np.array([])
            return
        if len(e_seq) == 0 and not inverse:
            self.i_anom = np.array([])
            return

        indices_to_keep: np.ndarray = np.concatenate(
            [range(e_seq[0], e_seq[-1] + 1) for e_seq in e_seq]
        )

        if not inverse:
            mask: np.ndarray = np.isin(self.i_anom, indices_to_keep)
            self.i_anom = self.i_anom[mask]
        else:
            mask_inv: np.ndarray = np.isin(self.i_anom_inv, indices_to_keep)
            self.i_anom_inv = self.i_anom_inv[mask_inv]

    def score_anomalies(self, prior_idx: int) -> None:
        """Calculate anomaly scores based on max distance from epsilon for each
        anomalous sequence.

        Args:
            prior_idx (int): starting index of window within full set of test
                values for channel
        """

        groups: List[List[int]] = [
            list(group) for group in mit.consecutive_groups(self.i_anom)
        ]

        for e_seq in groups:

            score_dict: Dict[str, Any] = {
                "start_idx": e_seq[0] + prior_idx,
                "end_idx": e_seq[-1] + prior_idx,
                "score": 0,
            }

            score: float = max(
                [
                    abs(self.e_s[i] - self.epsilon) / (self.mean_e_s + self.sd_e_s)
                    for i in range(e_seq[0], e_seq[-1] + 1)
                ]
            )
            inv_score: float = max(
                [
                    abs(self.e_s_inv[i] - self.epsilon_inv)
                    / (self.mean_e_s + self.sd_e_s)
                    for i in range(e_seq[0], e_seq[-1] + 1)
                ]
            )

            # the max score indicates whether anomaly was from regular
            # or inverted errors
            score_dict["score"] = max([score, inv_score])
            self.anom_scores.append(score_dict)
