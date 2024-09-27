from spaceai.data import ESA, ESAMissions
from spaceai.benchmark import ESABenchmark
from spaceai.models.anomaly import Telemanom
from spaceai.models.predictors import LSTM

import numpy as np
from torch import nn
from torch import optim


def main():
    benchmark = ESABenchmark("esa", "experiments", 250, "datasets")
    for mission_wrapper in ESAMissions:
        mission = mission_wrapper.value
        for channel_id in mission.target_channels:
            esa_channel = ESA(
                "datasets", mission, channel_id, mode="anomaly", train=False
            )
            low_perc, high_perc = np.percentile(esa_channel.data, [5, 95])
            volatility = np.std(esa_channel.data)

            detector = Telemanom(low_perc, high_perc, volatility)
            predictor = LSTM(1, [5], 1, 0.3)
            predictor.build()

            benchmark.run(
                mission,
                channel_id,
                predictor,
                detector,
                fit_predictor_args=dict(
                    criterion=nn.MSELoss(),
                    optimizer=optim.Adam(predictor.model.parameters(), lr=0.001),
                    epochs=1,
                    patience_before_stopping=10,
                    min_delta=0.0003,
                    batch_size=64,
                ),
                overlapping_train=False,
            )


if __name__ == "__main__":
    main()
