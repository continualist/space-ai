import logging
from typing import (
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import more_itertools as mit
import numpy as np

from .error_based_detector import ErrorBasedDetector

logger = logging.getLogger("telemanom")


class Telemanom(ErrorBasedDetector):
    """Batch processing of errors between actual and predicted values for a channel.
    This class is used to performs the business logic for identifying anomalies in a
    window of errors by applying Non-Parametric Dynamic Thresholding (NDT) algorithm.

    The default parameters are set to the values employed in the original paper.
    """

    SD_LIM = 12.0

    def __init__(
        self,
        low_perc: float,
        high_perc: float,
        channel_volatility: float,
        window_size: int = 2100,
        n_eval_per_window: int = 70,
        smoothing_perc: float = 0.05,
        error_offset: int = 100,
        pred_buffer: int = 250,
        ignore_first_n_factor: float = 2,
        pruning_factor: float = 0.12,
        force_early_anomaly: bool = False,
        adjust_ewma: bool = True,
        batch_size_ewma: int = 1000,
    ):
        """Batch processing of errors between actual and predicted values for a channel.

        Args:
            low_perc (float): lower percentile for channel values. We suggest setting
                this parameter by using `np.percentile(y_test, 5)`.
            high_perc (float): upper percentile for channel values. We suggest setting
                this parameter by using `np.percentile(y_test, 95)`.
            channel_volatility (float): standard deviation of channel values. We suggest
                setting this parameter by using `np.std(y_test)`.
            window_size (int): size of the window of errors employed for the evaluation.
            n_eval_per_window (int): actual number of points to evaluate in every window.
                This parameter identifies the last `n_eval_per_window` points in the
                window to evaluate. For the first window, this parameter is ignored
                and the first `window_size - ignore_first_n` points are evaluated.
            smoothing_perc (float): percentage of the window to use for smoothing the
                history of errors.
            n_evals_per_window (int): number of points to evaluate in every window.
            error_offset (int): number of indices to consider before and after
                an anomalous value.
            pred_buffer (int): number of points employed by the predictor to make predictions.
                This parameter is employed to denote the number initial points to ignore
                after prompting the predictor in the first window. This is the parameter
                l_s in the paper.
            ignore_first_n (int): number of initial values to ignore. The actual number
                is computed as `ignore_first_n_factor*l_s`. We suggest setting
                this parameter to 2 if the total length of the channel is > 2500, to
                1 if the total length of the channel is between 1800 and 2500, and to
                0 if the total length of the channel is < 1800.
            pruning_factor (float): factor for pruning the anomalies that don't meet the
                minimum separation according to the p-value. We suggest setting this
                parameter to 0.12.
            force_early_anomaly (bool): whether to force an anomaly in the first window
            adjust_ewma (bool): whether to adjust the EWMA smoothing parameter.
            batch_size_ewma (int): batch size for the EWMA smoothing.
        """
        if error_offset < 0:
            raise ValueError("error_offset must be greater than or equal to 0")
        if low_perc > high_perc:
            raise ValueError("low_perc must be less than high_perc")
        if pruning_factor < 0 or pruning_factor > 1:
            raise ValueError("pruning_factor must be between 0 and 1")

        self.window_size = window_size
        self._n_eval: int = n_eval_per_window
        self.smoothing_perc = smoothing_perc
        self.e_buf: int = error_offset
        self.perc_low: float = low_perc
        self.perc_high: float = high_perc
        self.inter_range: float = self.perc_high - self.perc_low
        self.sd_values: float = channel_volatility
        self.pred_buffer: int = pred_buffer
        self.ignore_first_n_factor: float = ignore_first_n_factor
        self.p: float = pruning_factor
        self.force_early_anomaly: bool = force_early_anomaly

        self.ewma = EWMA(
            int(window_size * smoothing_perc),
            adjust=adjust_ewma,
            batch_size=batch_size_ewma,
        )
        self.window = np.array([])
        self.eval_buffer = np.array([])
        self.n_window: int = 0
        self.first_pred: int = int(ignore_first_n_factor * pred_buffer)
        self.last_pred: int = self.first_pred

    def compute_error(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        reduce: Optional[Literal["mean", "min", "max"]] = None,
    ) -> Union[np.ndarray, float]:
        """Compute the error between the actual and predicted values.

        Args:
            y_pred (np.ndarray): predicted telemetry values
            y_hat (np.ndarray): actual telemetry values

        Returns:
            float: error between the actual and predicted values
        """
        err = np.abs(y_true - y_pred)
        if reduce is None:
            return err
        if reduce == "mean":
            return np.mean(err)
        if reduce == "min":
            return np.amin(err)
        if reduce == "max":
            return np.amax(err)

    def detect_anomalies(
        self, y_pred: np.ndarray, y_true: np.ndarray, **kwargs
    ) -> Optional[np.ndarray]:
        """Calculate the prediction errors between actual and predicted values.

        Args:
            y_true (np.ndarray): actual telemetry values
            y_pred (np.ndarray): predicted telemetry values

        Returns:
            np.ndarray: prediction errors
        """
        # Compute the errors and update the window
        err = self.compute_error(y_pred, y_true)
        err_s = self.ewma(err)
        self.eval_buffer = np.concatenate([self.eval_buffer, err_s])
        scores = None
        while len(self.eval_buffer) >= self.n_eval:
            moved = False
            # The window is not at its capacity yet
            if len(self.window) < self.window_size:
                missing = self.window_size - len(self.window)
                self.window = np.append(self.window, self.eval_buffer[:missing])
                if len(self.eval_buffer) <= missing:
                    self.eval_buffer = np.array([])
                else:
                    self.eval_buffer = self.eval_buffer[missing:]
                moved = True

            # The window is at its capacity and we have sufficient errors to evaluate
            elif len(self.eval_buffer) >= self.n_eval:
                self.window = np.append(self.window, self.eval_buffer[: self.n_eval])[
                    self.n_eval :
                ]
                self.eval_buffer = self.eval_buffer[self.n_eval :]
                moved = True

            # We updated the window and it is at its capacity
            if len(self.window) == self.window_size and moved:
                window_scores = self.process_window()
                scores = (
                    window_scores
                    if scores is None
                    else np.concatenate([scores, window_scores])
                )
                self.n_window += 1
        if scores is not None:
            self.last_pred += len(scores)
        return scores

    def process_window(self) -> np.ndarray:
        """Process the window of errors and identify the anomalies.

        Returns:
            np.ndarray: anomaly scores within the evaluated portion of the window
        """
        # Computing the statistics on the window and the inverted errors
        e_s = self.window
        if self.n_window == 0 and self.force_early_anomaly:
            e_s[: self.pred_buffer] = [
                np.mean(e_s[: self.pred_buffer * 2])
            ] * self.pred_buffer
        mean_e_s, sd_e_s = np.mean(e_s), np.std(e_s)
        e_s_inv = 2 * np.mean(e_s) - e_s  # Inverted errors

        # Finding the epsilon value for the errors and the inverted errors
        _, epsilon = self.find_epsilon(e_s, mean_e_s, sd_e_s)
        _, epsilon_inv = self.find_epsilon(e_s_inv, mean_e_s, sd_e_s)

        # Finding the anomalies by comparing the optimal epsilon found
        i_anom, e_seq, non_anom_max = self.compare_to_epsilon(e_s, epsilon, sd_e_s)
        i_anom_inv, e_seq_inv, non_anom_max_inv = self.compare_to_epsilon(
            e_s_inv, epsilon_inv, sd_e_s
        )
        if len(i_anom) == 0 and len(i_anom_inv) == 0:
            return np.zeros(self.n_eval)

        # Pruning the anomalies that don't meet the minimum separation according to the
        # p-value
        i_anom = self.prune_anomalies(e_s, e_seq, i_anom, non_anom_max)
        i_anom_inv = self.prune_anomalies(
            e_s_inv, e_seq_inv, i_anom_inv, non_anom_max_inv
        )
        if len(i_anom) == 0 and len(i_anom_inv) == 0:
            return np.zeros(self.n_eval)
        # Merging the anomalies in regular and inverted errors
        i_anom = np.sort(np.unique(np.append(i_anom, i_anom_inv))).astype("int")

        # Scoring the anomalies
        scores = self.score_anomalies(
            e_s, e_s_inv, epsilon, epsilon_inv, i_anom, mean_e_s, sd_e_s
        )

        return scores

    def find_epsilon(
        self, e_s: np.ndarray, mean_e_s: float, sd_e_s: float
    ) -> Tuple[float, float]:
        """
        Find the anomaly threshold that maximizes function representing
        tradeoff between:
            a) number of anomalies and anomalous ranges
            b) the reduction in mean and st dev if anomalous points are removed
            from errors
        (see https://arxiv.org/pdf/1802.04431.pdf)

        Args:
            e_s (np.ndarray): errors
            mean_e_s (float): mean of the errors
            sd_e_s (float): standard deviation of the errors

        Returns:
            Tuple[float, float]: anomaly threshold and epsilon
        """
        sd_threshold = self.SD_LIM
        epsilon = mean_e_s + self.SD_LIM * sd_e_s

        max_score = float("-inf")
        for z in np.arange(2.5, self.SD_LIM, 0.5):
            epsilon_z = mean_e_s + (sd_e_s * z)

            pruned_e_s = e_s[e_s < epsilon_z]

            i_anom = np.argwhere(e_s >= epsilon_z).reshape(-1)
            if len(i_anom) > 0:
                e_buf = self.e_buf
                i_anom = np.concatenate(
                    [np.arange(i - e_buf, i + e_buf) for i in i_anom]
                )
                i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
                i_anom = np.sort(np.unique(i_anom))

                # group anomalous indices into continuous sequences
                groups = [list(group) for group in mit.consecutive_groups(i_anom)]
                e_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

                mean_perc_decr = float((mean_e_s - np.mean(pruned_e_s)) / mean_e_s)
                sd_perc_decr = float((sd_e_s - np.std(pruned_e_s)) / sd_e_s)
                score = (mean_perc_decr + sd_perc_decr) / (
                    len(e_seq) ** 2 + len(i_anom)
                )

                # sanity checks / guardrails
                if (
                    score >= max_score
                    and len(e_seq) <= 5
                    and len(i_anom) < (len(e_s) * 0.5)
                ):
                    max_score, sd_threshold, epsilon = score, z, epsilon_z
        return sd_threshold, epsilon

    def compare_to_epsilon(
        self, e_s: np.ndarray, epsilon: float, sd_e_s: float
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], float]:
        """Compare smoothed error values to epsilon (error threshold) and group
        consecutive errors together into sequences.

        Args:
            e_s (np.ndarray): smoothed errors
            epsilon (float): error threshold
            sd_e_s (float): standard deviation of the errors

        Returns:
            Tuple[np.ndarray, List[Tuple[int, int]], float]: indices of anomalies,
                sequences of anomalies, and the maximum error among non-anomalous values
        """

        # Check: scale of errors compared to values too small?
        max_error = max(e_s)
        if (
            not (
                sd_e_s > (0.05 * self.sd_values)
                or max_error > (0.05 * self.inter_range)
            )
            or not max_error > 0.05
        ):
            return np.array([]), [], float("-inf")

        i_anom = np.argwhere(
            (e_s >= epsilon) & (e_s > 0.05 * self.inter_range)
        ).flatten()

        if len(i_anom) == 0:
            return np.array([]), [], float("-inf")
        e_buf = self.e_buf
        i_anom = np.concatenate([np.arange(i - e_buf, i + e_buf) for i in i_anom])
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
        i_anom = np.sort(np.unique(i_anom))

        # if it is first window, ignore initial errors (need some history)
        if self.n_window == 0 and self.num_to_ignore > 0:
            i_anom = i_anom[i_anom >= self.num_to_ignore]
        else:
            i_anom = i_anom[i_anom >= len(e_s) - self.n_eval]

        non_anom_max = np.amax(np.delete(e_s, i_anom))

        # group anomalous indices into continuous sequences
        groups: List[List[int]] = [
            list(group) for group in mit.consecutive_groups(i_anom)
        ]
        e_seq: List[Tuple[int, int]] = [
            (g[0], g[-1]) for g in groups if not g[0] == g[-1]
        ]
        return i_anom, e_seq, non_anom_max

    def prune_anomalies(
        self,
        e_s: np.ndarray,
        e_seq: List[Tuple[int, int]],
        i_anom: np.ndarray,
        non_anom_max: float,
    ) -> np.ndarray:
        """Remove anomalies that don't meet minimum separation from the next closest
        anomaly or error value.

        Args:
            e_s (np.ndarray): smoothed errors
            e_seq (List[Tuple[int, int]]): sequences of anomalies
            i_anom (np.ndarray): indices of anomalies
            non_anom_max (float): maximum error among non-anomalous values

        Returns:
            np.ndarray: indices of anomalies after pruning
        """
        e_seq_max: np.ndarray = np.array([max(e_s[e[0] : e[1] + 1]) for e in e_seq])
        e_seq_max_sorted: np.ndarray = np.sort(e_seq_max)[::-1]
        e_seq_max_sorted = np.append(e_seq_max_sorted, [non_anom_max])

        if len(e_seq) == 0:
            return np.array([])

        i_to_remove: np.ndarray = np.array([])
        for i in range(0, len(e_seq_max_sorted) - 1):
            if (e_seq_max_sorted[i] - e_seq_max_sorted[i + 1]) / e_seq_max_sorted[
                i
            ] < self.p:
                i_to_remove = np.append(
                    i_to_remove, np.argwhere(e_seq_max == e_seq_max_sorted[i])
                )
            else:
                i_to_remove = np.array([])
        i_to_remove[::-1].sort()

        i_to_remove = np.array(i_to_remove, dtype=int)
        if len(i_to_remove) > 0:
            e_seq = np.delete(e_seq, i_to_remove, axis=0)

        if len(e_seq) == 0:
            return np.array([])

        indices_to_keep = np.concatenate(
            [range(subseq[0], subseq[-1] + 1) for subseq in e_seq]
        )

        mask = np.isin(i_anom, indices_to_keep)
        return i_anom[mask]

    def score_anomalies(
        self,
        e_s: np.ndarray,
        e_s_inv: np.ndarray,
        epsilon: float,
        epsilon_inv: float,
        i_anom: np.ndarray,
        mean_e_s: float,
        sd_e_s: float,
    ) -> np.ndarray:
        """Calculate anomaly scores based on max distance from epsilon for each
        anomalous sequence.

        Args:
            e_s (np.ndarray): smoothed errors
            epsilon (float): error threshold
            i_anom (np.ndarray): indices of anomalies
            mean_e_s (float): mean of the errors
            sd_e_s (float): standard deviation of the errors

        Returns:
            List[Dict[str, float]]: anomaly scores
        """
        groups = [list(group) for group in mit.consecutive_groups(i_anom)]
        scores = np.zeros(self.n_eval)
        for e_seq in groups:
            min_idx, max_idx = e_seq[0], e_seq[-1] + 1
            score = np.amax(
                np.abs(e_s[min_idx:max_idx] - epsilon) / (mean_e_s + sd_e_s)
            )
            inv_score = np.amax(
                np.abs(e_s_inv[min_idx:max_idx] - epsilon_inv) / (mean_e_s + sd_e_s)
            )

            max_score = max([score, inv_score])
            scores[min_idx - i_anom[0] : max_idx - i_anom[0]] = max_score
        return scores

    def reset_state(self):
        self.ewma.reset()
        self.window = np.array([])
        self.eval_buffer = np.array([])
        self.n_window = 0
        self._predictor.reset_state()

    @property
    def n_eval(self) -> int:
        if self.n_window == 0:
            return self.window_size - self.num_to_ignore
        return self._n_eval

    @property
    def num_to_ignore(self) -> int:
        return int(self.ignore_first_n_factor * self.pred_buffer)


class EWMA:
    def __init__(self, window_size: int, adjust: bool = True, batch_size: int = 1000):
        self.window_size = window_size
        self.adjust = adjust
        self.batch_size = batch_size
        alpha = 2 / (window_size + 1.0)
        self.alpha_rev = 1 - alpha
        self.alpha = 1 if adjust else alpha
        self.last_ewma = 0 if self.adjust else None
        self.last_div = 0

    def reset(self):
        self.last_ewma = 0 if self.adjust else None
        self.last_div = 0

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                self.run(data[i : i + self.batch_size])
                for i in range(0, len(data), self.batch_size)
            ]
        )

    def run(self, data: np.ndarray) -> np.ndarray:
        n = data.shape[0]
        pows = self.alpha_rev ** np.arange(n + 1)

        if self.last_ewma is None:
            self.last_ewma = data[0]

        scale_arr = 1 / pows[:-1]
        offset = self.last_ewma * pows[1:]
        if self.adjust:
            offset_div = self.last_div * pows[1:]

        pw0 = self.alpha * self.alpha_rev ** (n - 1)

        mult = data * pw0 * scale_arr
        cumsums = mult.cumsum()
        ewma_result = offset + (cumsums * scale_arr[::-1])

        self.last_ewma = ewma_result[-1]

        if self.adjust:
            div = pows[:-1].cumsum() + offset_div
            self.last_div = div[-1]
            return ewma_result / div

        return ewma_result


__all__ = ["Telemanom"]
