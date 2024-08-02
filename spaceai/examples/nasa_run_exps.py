"""Main file to run experiments."""

import argparse
import sys

sys.path.append(
    "/Users/lilanpei/Github/space-ai/spaceai/benchmarks/nasa_bechmark",
)
from telemanom.detector import Detector

parser = argparse.ArgumentParser(
    description="Parse path to anomaly labels if provided."
)
parser.add_argument("-l", "--labels_path", default=None, required=False)
args = parser.parse_args()

if __name__ == "__main__":

    detector = Detector(
        labels_path=args.labels_path,
        result_path="result_stage1",
    )
    detector.run_stage2_retrain()
    detector.run_stage3_anomaly()
