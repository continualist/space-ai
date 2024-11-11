import os

import pandas as pd
from spaceai.data import ESA, ESAMissions
from spaceai.benchmark import ESABenchmark
from spaceai.models.anomaly import Telemanom
from spaceai.models.predictors import LSTM

import numpy as np
from torch import nn
from torch import optim

from spaceai.utils.callbacks import SystemMonitorCallback


def main():
    benchmark = ESABenchmark(
        run_id="esa_lstm",
        exp_dir="experiments",
        seq_length=250,
        n_predictions=10,
        data_root="datasets",
    )
    callbacks = [SystemMonitorCallback()]
    for mission_wrapper in ESAMissions:
        mission = mission_wrapper.value
        for channel_id in mission.target_channels:
            esa_channel = ESA(
                "datasets", 
                mission, 
                channel_id, 
                mode="anomaly", 
                train=False
            )

            detector = Telemanom(pruning_factor=0.13)
            predictor = LSTM(
                esa_channel.in_features_size, 
                [80, 80], 
                10, 
                reduce_out="first",
                dropout=0.3,
                washout=249,
            )
            predictor.build()

            benchmark.run(
                mission,
                channel_id,
                predictor,
                detector,
                fit_predictor_args=dict(
                    criterion=nn.MSELoss(),
                    optimizer=optim.Adam(predictor.model.parameters(), lr=0.001),
                    epochs=35,
                    patience_before_stopping=10,
                    min_delta=0.0003,
                    batch_size=64,
                    restore_best=False,
                ),
                overlapping_train=True,
                restore_predictor=False,
                callbacks=callbacks,
            )
        
    results_df = pd.read_csv(os.path.join(benchmark.run_dir, "results.csv"))
    tp = results_df['true_positives'].sum()
    fp = results_df['false_positives'].sum()
    fn = results_df['false_negatives'].sum()

    total_precision = tp / (tp + fp)
    total_recall = tp / (tp + fn)
    total_f1 = 2 * (total_precision * total_recall) / \
        (total_precision + total_recall)
    
    print("True Positives: ", tp)
    print("False Positives: ", fp)
    print("False Negatives: ", fn)
    print("Total Precision: ", total_precision)
    print("Total Recall: ", total_recall)
    print("Total F1: ", total_f1)


if __name__ == "__main__":
    main()
