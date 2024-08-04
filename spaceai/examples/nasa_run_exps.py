"""Main file to run experiments."""

import argparse
import sys

sys.path.append(
    "../../spaceai/benchmarks/nasa_bechmark",
)
from telemanom.detector import Detector
from telemanom.trainer import Trainer

parser = argparse.ArgumentParser(
    description="Parse path to anomaly labels if provided."
)
parser.add_argument("-l", "--labels_path", default=None, required=False)
args = parser.parse_args()

if __name__ == "__main__":

    trainer = Trainer(
        labels_path=args.labels_path,
        result_path="result_stage1",
    )
    trainer.retrain()

    detector = Detector(
        run_id=trainer.id,
        labels_path=trainer.labels_path,
        chan_df=trainer.chan_df,
        config=trainer.config,
        result_path=trainer.result_path,
        final_result_path=trainer.final_result_path,
    )
    detector.anomaly_detection()
