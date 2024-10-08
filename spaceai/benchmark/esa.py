from __future__ import annotations

import os
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import more_itertools as mit
import numpy as np
import pandas as pd
from torch.utils.data import (
    DataLoader,
    Subset,
)

from spaceai.data import (
    ESA,
    ESAMission,
)
from spaceai.utils import data

if TYPE_CHECKING:
    from spaceai.models.predictors import SequenceModel
    from spaceai.models.anomaly import AnomalyDetector

from tqdm import tqdm

from .benchmark import Benchmark


class ESABenchmark(Benchmark):

    def __init__(
        self,
        run_id: str,
        exp_dir: str,
        seq_length: int = 250,
        data_root: str = "datasets",
    ):
        """Initializes a new ESA benchmark run.

        Args:
            run_id (str): A unique identifier for this run.
            exp_dir (str): The directory where the results of this run are stored.
            seq_length (int): The length of the sequences used for training and testing.
            data_root (str): The root directory of the ESA dataset.
        """
        super().__init__(run_id, exp_dir)
        self.data_root: str = data_root
        self.seq_length: int = seq_length
        self.all_results: List[Dict[str, Any]] = []

    def run(
        self,
        mission: ESAMission,
        channel_id: str,
        predictor: SequenceModel,
        detector: AnomalyDetector,
        fit_predictor_args: Optional[Dict[str, Any]] = None,
        perc_eval: Optional[float] = 0.2,
        restore_predictor: bool = False,
        overlapping_train: bool = True,
    ):
        """Runs the benchmark for a given channel.

        Args:
            mission (ESAMission): the mission to be used
            channel_id (str): the ID of the channel to be used
            predictor (SequenceModel): the sequence model to be trained
            detector (AnomalyDetector): the anomaly detector to be used
            fit_predictor_args (Optional[Dict[str, Any]]): additional arguments for the predictor's fit method
            perc_eval (Optional[float]): the percentage of the training data to be used for evaluation
            restore_predictor (bool): whether to restore the predictor from a previous run
            overlapping_train (bool): whether to use overlapping sequences for the training dataset
        """
        train_channel, test_channel = self.load_channel(
            mission,
            channel_id,
            overlapping_train=overlapping_train,
        )
        os.makedirs(self.run_dir, exist_ok=True)

        results: Dict[str, Any] = {"channel_id": channel_id}
        train_history = None
        if (
            os.path.exists(os.path.join(self.run_dir, f"predictor-{channel_id}.pt"))
            and restore_predictor
        ):
            print(f"Restoring predictor for channel {channel_id}...")
            predictor.load(os.path.join(self.run_dir, f"predictor-{channel_id}.pt"))

        elif fit_predictor_args is not None:
            print(f"Fitting the predictor for channel {channel_id}...")
            # Training the predictor
            batch_size = fit_predictor_args.pop("batch_size", 64)
            eval_channel = None
            if perc_eval is not None:
                # Split the training data into training and evaluation sets
                indices = np.arange(len(train_channel))
                np.random.shuffle(indices)
                eval_size = int(len(train_channel) * perc_eval)
                eval_channel = Subset(train_channel, indices[:eval_size])
                train_channel = Subset(train_channel, indices[eval_size:])
            train_loader = DataLoader(
                train_channel,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=data.seq_collate_fn(n_inputs=2, mode="batch"),
            )
            eval_loader = (
                DataLoader(
                    eval_channel,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=data.seq_collate_fn(n_inputs=2, mode="batch"),
                )
                if eval_channel is not None
                else None
            )
            t1 = time.time()
            train_history = predictor.fit(
                train_loader=train_loader,
                valid_loader=eval_loader,
                **fit_predictor_args,
            )
            t2 = time.time()
            results["train_time"] = t2 - t1
            print("Training time on channel", channel_id, ":", results["train_time"])
            train_history = pd.DataFrame.from_records(train_history).to_csv(
                os.path.join(self.run_dir, f"train_history-{channel_id}.csv"),
                index=False,
            )
            predictor.save(os.path.join(self.run_dir, f"predictor-{channel_id}.pt"))

        print(f"Predicting the test data for channel {channel_id}...")
        test_loader = DataLoader(
            test_channel,
            batch_size=1,
            shuffle=False,
            collate_fn=data.seq_collate_fn(n_inputs=3, mode="time"),
        )
        t1 = time.time()
        y_pred, y_trg, anomalies = zip(
            *[
                (
                    predictor(x).detach().squeeze().cpu().numpy(),
                    y.detach().squeeze().cpu().numpy(),
                    a.detach().squeeze().cpu().numpy(),
                )
                for x, y, a in tqdm(test_loader, desc="Predicting")
            ]
        )
        y_pred, y_trg, anomalies = [
            np.concatenate(seq) for seq in [y_pred, y_trg, anomalies]
        ]
        t2 = time.time()
        results["predict_time"] = t2 - t1
        results["test_loss"] = np.mean(((y_pred - y_trg) ** 2))  # type: ignore[operator]
        print("Test loss for channel", channel_id, ":", results["test_loss"])
        print("Prediction time for channel", channel_id, ":", results["predict_time"])

        # Testing the detector
        print("Detecting anomalies for channel", channel_id)
        t1 = time.time()
        y_scores = detector.detect_anomalies(y_pred, anomalies)
        t2 = time.time()
        results["detect_time"] = t2 - t1
        print("Detection time for channel", channel_id, ":", results["detect_time"])

        y_true = anomalies
        true_idx, pred_idx = np.where(y_true == 1), np.where(y_scores != 0)
        true_seq = [list(group) for group in mit.consecutive_groups(true_idx[0])]
        pred_seq = [list(group) for group in mit.consecutive_groups(pred_idx[0])]

        is_false_positive = [1 for _ in range(len(pred_seq))]
        results["n_anomalies"] = len(true_seq)
        results["n_detected"] = len(pred_seq)
        results["true_positives"] = 0
        results["false_positives"] = 0
        results["false_negatives"] = 0

        for t_seq in true_seq:
            detected = False
            for i, p_seq in enumerate(pred_seq):
                if len(set(t_seq) & set(p_seq)) > 0:
                    detected = True
                    is_false_positive[i] = 0
                    break
            if detected:
                results["true_positives"] += 1
            else:
                results["false_negatives"] += 1

        results["false_positives"] = sum(is_false_positive)
        results["precision"] = (
            results["true_positives"] / results["n_detected"]
            if results["n_detected"] > 0
            else 0
        )
        results["recall"] = (
            results["true_positives"] / results["n_anomalies"]
            if results["n_anomalies"] > 0
            else 0
        )
        results["f1"] = (
            (
                2
                * (results["precision"] * results["recall"])
                / (results["precision"] + results["recall"])
            )
            if results["precision"] + results["recall"] > 0
            else 0
        )
        if train_history is not None:
            results["train_loss"] = train_history[-1]["loss_train"]
            if eval_loader is not None:
                results["eval_loss"] = train_history[-1]["loss_eval"]
        print("Results for channel", channel_id)
        print(results)
        self.all_results.append(results)

        df = pd.DataFrame.from_records(self.all_results).to_csv(
            os.path.join(self.run_dir, "results.csv"), index=False
        )
        print(df)

    def load_channel(
        self, mission: ESAMission, channel_id: str, overlapping_train: bool = True
    ) -> Tuple[ESA, ESA]:
        """Load the training and testing datasets for a given channel.

        Args:
            channel_id (str): the ID of the channel to be used
            overlapping_train (bool): whether to use overlapping sequences for the training dataset

        Returns:
            Tuple[ESA, ESA]: training and testing datasets
        """
        train_channel = ESA(
            root=self.data_root,
            mission=mission,
            channel_id=channel_id,
            mode="prediction",
            overlapping=overlapping_train,
            seq_length=self.seq_length,
            drop_last=True,
        )

        test_channel = ESA(
            root=self.data_root,
            mission=mission,
            channel_id=channel_id,
            mode="anomaly",
            overlapping=False,
            seq_length=self.seq_length,
            train=False,
            drop_last=False,
        )

        return train_channel, test_channel
