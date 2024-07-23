import os
import numpy as np
import pandas as pd
from datetime import datetime as dt
import logging
import glob
import time
import csv

from telemanom.helpers import Config
from telemanom.errors import Errors
import telemanom.helpers as helpers
from telemanom.channel import Channel
from telemanom.modeling import Model
from telemanom.hpo import HPO

from sklearn.metrics import mean_squared_error

logger = helpers.setup_logging()

class Detector:
    def __init__(self, labels_path=None, result_path='results/',
                 config_path='config.yaml'):
        """
        Top-level class for running anomaly detection over a group of channels
        with values stored in .npy files. Also evaluates performance against a
        set of labels if provided.

        Args:
            labels_path (str): path to .csv containing labeled anomaly ranges
                for group of channels to be processed
            result_path (str): directory indicating where to stick result .csv
            config_path (str): path to config.yaml

        Attributes:
            labels_path (str): see Args
            results (list of dicts): holds dicts of results for each channel
            result_df (dataframe): results converted to pandas dataframe
            chan_df (dataframe): holds all channel information from labels .csv
            result_tracker (dict): if labels provided, holds results throughout
                processing for logging
            config (obj):  Channel class object containing train/test data
                for X,y for a single channel
            y_hat (arr): predicted channel values
            id (str): datetime id for tracking different runs
            result_path (str): see Args
        """

        self.labels_path = labels_path
        self.results = []
        self.result_df = None
        self.chan_df = None
        self.final_result_path = "final_results"

        self.result_tracker = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }

        self.config = Config(config_path)
        self.y_hat = None

        if self.config.stage == "1HPO":
            self.id = self.config.stage + "_" + self.config.hpo_type + "_" + self.config.model_architecture + "_" + dt.now().strftime('%Y-%m-%d_%H.%M.%S')
        else: # TODO: manage the other stages, i.e. 2/3
            pass

        helpers.make_dirs(self.id)

        # add logging FileHandler based on ID
        hdlr = logging.FileHandler('data/logs/%s.log' % self.id)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

        self.result_path = os.path.join(result_path, self.config.model_architecture)

        if self.labels_path:
            self.chan_df = pd.read_csv(labels_path)
        else:
            chan_ids = [x.split('.')[0] for x in os.listdir('./data/test/')]
            self.chan_df = pd.DataFrame({"chan_id": chan_ids})

        logger.info("{} channels found for processing."
                    .format(len(self.chan_df)))

    def evaluate_sequences(self, errors, label_row):
        """
        Compare identified anomalous sequences with labeled anomalous sequences.

        Args:
            errors (obj): Errors class object containing detected anomaly
                sequences for a channel
            label_row (pandas Series): Contains labels and true anomaly details
                for a channel

        Returns:
            result_row (dict): anomaly detection accuracy and results
        """

        result_row = {
            'false_positives': 0,
            'false_negatives': 0,
            'true_positives': 0,
            'fp_sequences': [],
            'tp_sequences': [],
            'num_true_anoms': 0
        }

        matched_true_seqs = []

        label_row['anomaly_sequences'] = eval(label_row['anomaly_sequences'])
        result_row['num_true_anoms'] += len(label_row['anomaly_sequences'])
        result_row['scores'] = errors.anom_scores

        if len(errors.E_seq) == 0:
            result_row['false_negatives'] = result_row['num_true_anoms']

        else:
            true_indices_grouped = [list(range(e[0], e[1]+1)) for e in label_row['anomaly_sequences']]
            true_indices_flat = set([i for group in true_indices_grouped for i in group])

            for e_seq in errors.E_seq:
                i_anom_predicted = set(range(e_seq[0], e_seq[1]+1))

                matched_indices = list(i_anom_predicted & true_indices_flat)
                valid = True if len(matched_indices) > 0 else False

                if valid:

                    result_row['tp_sequences'].append(e_seq)

                    true_seq_index = [i for i in range(len(true_indices_grouped)) if
                                      len(np.intersect1d(list(i_anom_predicted), true_indices_grouped[i])) > 0]

                    if not true_seq_index[0] in matched_true_seqs:
                        matched_true_seqs.append(true_seq_index[0])
                        result_row['true_positives'] += 1

                else:
                    result_row['fp_sequences'].append([e_seq[0], e_seq[1]])
                    result_row['false_positives'] += 1

            result_row["false_negatives"] = len(np.delete(label_row['anomaly_sequences'],
                                                          matched_true_seqs, axis=0))

        logger.info('Channel Stats: TP: {}  FP: {}  FN: {}'.format(result_row['true_positives'],
                                                                   result_row['false_positives'],
                                                                   result_row['false_negatives']))

        for key, value in result_row.items():
            if key in self.result_tracker:
                self.result_tracker[key] += result_row[key]

        return result_row

    def log_final_stats(self):
        """
        Log final stats at end of experiment.
        """

        if self.labels_path:

            logger.info('Final Totals:')
            logger.info('-----------------')
            logger.info('True Positives: {}'
                        .format(self.result_tracker['true_positives']))
            logger.info('False Positives: {}'
                        .format(self.result_tracker['false_positives']))
            logger.info('False Negatives: {}'
                        .format(self.result_tracker['false_negatives']))
            try:
                self.precision = float(self.result_tracker['true_positives']) / (float(self.result_tracker['true_positives'] + self.result_tracker['false_positives']))
                self.recall = float(self.result_tracker['true_positives']) / (float(self.result_tracker['true_positives'] + self.result_tracker['false_negatives']))
                self.f1 = 2*((self.precision*self.recall)/(self.precision+self.recall))

                logger.info('Precision: {0:.2f}'.format(self.precision))
                logger.info('Recall: {0:.2f}'.format(self.recall))
                logger.info('F1 Score: {0:.2f}\n'.format(self.f1))

            except ZeroDivisionError:
                logger.info('Precision: NaN')
                logger.info('Recall: NaN')
                logger.info('F1 Score: NaN\n')

        else:
            logger.info('Final Totals:')
            logger.info('-----------------')
            logger.info('Total channel sets evaluated: {}'
                        .format(len(self.result_df)))
            logger.info('Total anomalies found: {}'
                        .format(self.result_df['n_predicted_anoms'].sum()))
            logger.info('Avg normalized prediction error: {}'
                        .format(self.result_df['normalized_pred_error'].mean()))
            logger.info('Total number of values evaluated: {}\n'
                        .format(self.result_df['num_test_values'].sum()))

    def run_stage1_hpo(self):
        hpo = HPO(self.config, self.id) # note: self.config=.yaml file

        for i, row in self.chan_df.iterrows():
            logger.info('Stream # {}: {}'.format(i + 1, row.chan_id))
            channel = Channel(self.config, row.chan_id)
            channel.load_data()

            start_time = time.time()  # Start time measurement
            best_metric, best_config = hpo.execute_hpo(channel,
                                                       num_samples=self.config.num_sampler,
                                                       num_samples2=self.config.num_sampler2,
                                                       resources={'cpu': self.config.cpu})
            end_time = time.time()  # End time measurement
            time_channel = end_time - start_time # time for channel (iteration)

            result_row = {
                'run_id': self.id,
                'hpo_type': self.config.hpo_type,
                'chan_id': row.chan_id,
                'best_metric': best_metric['MSE'],
                'time_channel': time_channel,
                'best_hps': best_config
            }
            if self.config.hpo_type == "adaptive":
                result_row['hps_importance'] = hpo.hps_importance

            self.results.append(result_row)

            self.result_df = pd.DataFrame(self.results)
            os.makedirs(self.result_path, exist_ok=True)
            self.result_df.to_csv(
                os.path.join(self.result_path, '{}.csv'.format(self.id)), index=False)

        # Final log
        logger.info('Final Totals:')
        logger.info('-----------------')
        logger.info('Total channel sets evaluated: {}'.format(len(self.result_df)))
        logger.info('Avg MSE: {}'.format(self.result_df['best_metric'].mean()))
        logger.info('Total TIME: {}'.format(self.result_df['time_channel'].sum()))


    def run_stage2_retrain(self):
        result_hpo_stage1_csv_files = glob.glob(os.path.join(self.result_path, "*.csv"))
        for csv_files in result_hpo_stage1_csv_files:
            with open(csv_files, 'r', encoding='utf-8') as csvfile:
                self.result_df = pd.read_csv(csvfile)
                self.id = self.result_df.iloc[0, 0]
                final_result_path = os.path.join(self.final_result_path, self.id, "time_tracking_results")
                os.makedirs(final_result_path, exist_ok=True)
                time_tracking_csv = os.path.join(final_result_path, "time_tracking.csv")

                file_exists = os.path.exists(time_tracking_csv)
                with open(time_tracking_csv, mode='a', newline='') as file:
                    csv_writer = csv.writer(file)

                    # If the file doesn't exist, write the header
                    if not file_exists:
                        csv_writer.writerow(['ID', 'Stage', 'Time (seconds)'])

                    start_time = time.time()
                    for index, row in self.result_df.iterrows():
                        best_hps = eval(row['best_hps'])

                        logger.info('Stream # {}: {}'.format(index, row['chan_id']))
                        channel = Channel(self.config, row['chan_id'], False)
                        channel.load_data()

                        if "ESN" in row['run_id']:
                            self.config.model_architecture = "ESN"
                            self.config.leakage = best_hps['leakage']
                            self.config.input_scaling = best_hps['input_scaling']
                            self.config.rho = best_hps['rho']
                            self.config.l2 = best_hps['l2']
                            self.config.layers = [best_hps['hidden_size_1'], best_hps['hidden_size_2']]
                        else:  # LSTM
                            self.config.learning_rate = best_hps['lr']
                            self.config.weight_decay = best_hps['weight_decay']
                            self.config.layers = [best_hps['hidden_size_1'], best_hps['hidden_size_2']]
                            self.config.model_architecture = "LSTM"
                            self.config.dropout = best_hps['dropout']

                        model = Model(self.config, row['run_id'], channel)
                        channel = model.batch_predict(self.final_result_path, channel)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    csv_writer.writerow([self.id, 'stage2_retrain', elapsed_time])
                    print(f"stage2_retrain took {elapsed_time:.2f} seconds")


    def run_stage3_anomaly(self):
        result_hpo_stage1_csv_files = glob.glob(os.path.join(self.result_path, "*.csv"))
        for csv_files in result_hpo_stage1_csv_files:
            with open(csv_files, 'r', encoding='utf-8') as csvfile:
                self.result_df = pd.read_csv(csvfile)
                self.id = self.result_df.iloc[0, 0]
                final_result_path = os.path.join(self.final_result_path, self.id, "time_tracking_results")
                os.makedirs(final_result_path, exist_ok=True)
                time_tracking_csv = os.path.join(final_result_path, "time_tracking.csv")
                
                file_exists = os.path.exists(time_tracking_csv)
                with open(time_tracking_csv, mode='a', newline='') as file:
                    csv_writer = csv.writer(file)

                    # If the file doesn't exist, write the header
                    if not file_exists:
                        csv_writer.writerow(['ID', 'Stage', 'Time (seconds)'])

                    start_time = time.time()
                    tot_test_mes = 0
                    for index, row in self.chan_df.iterrows():
                        logger.info('Stream # {}: {}'.format(index, row['chan_id']))
                        channel = Channel(self.config, row['chan_id'], False)
                        channel.load_data()
                        channel.y_hat = np.load(os.path.join(self.final_result_path, self.id, 'y_hat', '{}.npy'.format(channel.id)))

                        mse = mean_squared_error(channel.y_test[:,0], channel.y_hat)
                        tot_test_mes += mse
                        print(f"Mean Squared Error for channel {row['chan_id']} : {mse} {tot_test_mes}")
                        
                        errors = Errors(channel, self.config, self.id, self.final_result_path)
                        errors.process_batches(channel)

                        result_row = {
                            'run_id': self.id,
                            'chan_id': row['chan_id'],
                            'num_train_values': len(channel.X_train),
                            'num_test_values': len(channel.X_test),
                            'n_predicted_anoms': len(errors.E_seq),
                            'normalized_pred_error': errors.normalized,
                            'anom_scores': errors.anom_scores,
                            'test_mse': mse
                        }

                        if self.labels_path:
                            result_row = {**result_row,
                                        **self.evaluate_sequences(errors, row)}
                            result_row['spacecraft'] = row['spacecraft']
                            result_row['anomaly_sequences'] = row['anomaly_sequences']
                            result_row['class'] = row['class']
                            self.results.append(result_row)

                            logger.info('Total true positives: {}'
                                        .format(self.result_tracker['true_positives']))
                            logger.info('Total false positives: {}'
                                        .format(self.result_tracker['false_positives']))
                            logger.info('Total false negatives: {}\n'
                                        .format(self.result_tracker['false_negatives']))

                        else:
                            result_row['anomaly_sequences'] = errors.E_seq
                            self.results.append(result_row)

                            logger.info('{} anomalies found'
                                        .format(result_row['n_predicted_anoms']))
                            logger.info('anomaly sequences start/end indices: {}'
                                        .format(result_row['anomaly_sequences']))
                            logger.info('number of test values: {}'
                                        .format(result_row['num_test_values']))
                            logger.info('anomaly scores: {}\n'
                                        .format(result_row['anom_scores']))

                        self.result_df = pd.DataFrame(self.results)
                        final_result_path = os.path.join(self.final_result_path, self.id, "final_result")
                        os.makedirs(final_result_path, exist_ok=True)
                        self.result_df.to_csv(
                            os.path.join(final_result_path, '{}.csv'.format(self.id)),
                            index=False)

                    self.log_final_stats()
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    csv_writer.writerow([self.id, 'stage3_anomaly', elapsed_time])
                    print(f"Total Mean Squared Error : {tot_test_mes}")
                    print(f"stage3_anomaly took {elapsed_time:.2f} seconds")
