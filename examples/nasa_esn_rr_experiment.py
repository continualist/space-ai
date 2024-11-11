import os

import pandas as pd
from spaceai.data import NASA
from spaceai.benchmark import NASABenchmark
from spaceai.models.anomaly import Telemanom
from spaceai.models.predictors import ESN

import numpy as np
from torch import nn
from torch import optim

from spaceai.utils.callbacks import SystemMonitorCallback


def main():
    benchmark = NASABenchmark(
        run_id="nasa_esn_rr", 
        exp_dir="experiments", 
        seq_length=250, 
        n_predictions=10,
        data_root="datasets",
    )
    callbacks = [SystemMonitorCallback()]

    channels = NASA.channel_ids
    for i, channel_id in enumerate(channels):
        print(f'{i+1}/{len(channels)}: {channel_id}')

        nasa_channel = NASA(
            "datasets", 
            channel_id, 
            mode="anomaly", 
            train=False, 
        )

        detector = Telemanom(
            pruning_factor=0.13, 
            force_early_anomaly=channel_id == 'C-2'
        )
        predictor = ESN(
            nasa_channel.in_features_size, 
            [80, 80],
            10,
            reduce_out="mean",
            gradient_based=False,
            washout=200,
        )
        predictor.build()

        benchmark.run(
            channel_id,
            predictor,
            detector,
            fit_predictor_args=dict(
                criterion=nn.MSELoss(),
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
