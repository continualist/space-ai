"""
    SPAICE research PJ
    Main file to run experiments
"""
from telemanom.detector import Detector
import argparse

parser = argparse.ArgumentParser(description='Parse path to anomaly labels if provided.')
parser.add_argument('-l', '--labels_path', default=None, required=False)
args = parser.parse_args()

if __name__ == '__main__':
    # LSTM
    detector = Detector(labels_path=args.labels_path, result_path='result_hpo_stage1', config_path='config.yaml')
    detector.run_stage1_hpo()
    detector.run_stage2_retrain()
    detector.run_stage2_retrain()
    detector.run_stage2_retrain()
    detector.run_stage3_anomaly()

    # ESN
    detector = Detector(labels_path=args.labels_path, result_path='result_hpo_stage1', config_path='config2.yaml')
    detector.run_stage1_hpo()
    detector.run_stage2_retrain()
    detector.run_stage2_retrain()
    detector.run_stage2_retrain()
    detector.run_stage3_anomaly()
