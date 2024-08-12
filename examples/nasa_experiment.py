from spaceai.data import NASA
from spaceai.benchmark import NASABenchmark
from spaceai.models.anomaly import Telemanom
from spaceai.models.predictors import LSTM

import numpy as np
from torch import nn
from torch import optim


def main():

    benchmark = NASABenchmark("nasa", "experiments", 250, "datasets")

    for channel_id in NASA.channel_ids:
        nasa_channel = NASA(
            "datasets", NASA.channel_ids[0], mode="anomaly", train=False
        )
        low_perc, high_perc = np.percentile(nasa_channel.data, [5, 95])
        volatility = np.std(nasa_channel.data)

        detector = Telemanom(low_perc, high_perc, volatility)
        predictor = LSTM(1, [80, 80], 1, 0.3)
        predictor.build()

        benchmark.run(
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
        )


if __name__ == "__main__":
    main()
