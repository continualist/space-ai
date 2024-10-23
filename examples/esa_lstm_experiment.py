import os

import pandas as pd
from spaceai.data import ESA, ESAMissions
from spaceai.benchmark import ESABenchmark
from spaceai.models.anomaly import Telemanom
from spaceai.models.predictors import LSTM

import numpy as np
from torch import nn
from torch import optim


def main():

    benchmark = ESABenchmark("esa_lstm", "experiments", 250, "datasets")
    for mission_wrapper in ESAMissions:
        mission = mission_wrapper.value
        for channel_id in mission.target_channels:
            esa_channel = ESA(
                "datasets", mission, channel_id, mode="anomaly", train=False
            )
            y_test = esa_channel.data[esa_channel.window_size - 1:, 0]
            low_perc, high_perc = np.percentile(y_test, [5, 95])
            volatility = np.std(y_test)

            detector = Telemanom(low_perc, high_perc, volatility, pruning_factor=0.13)
            predictor = LSTM(esa_channel.in_features_size, [80, 80], 1, 0.3)
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
                ),
                overlapping_train=True,
                restore_predictor=True,
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
