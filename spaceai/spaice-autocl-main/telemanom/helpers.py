import logging
import yaml
import json
import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger('telemanom')
sys.path.append('../telemanom')


class Config:
    """Loads parameters from config.yaml into global object

    """

    def __init__(self, path_to_config):

        self.path_to_config = path_to_config

        if os.path.isfile(path_to_config):
            pass
        else:
            self.path_to_config = '../{}'.format(self.path_to_config)

        with open(self.path_to_config, "r") as f:
            self.dictionary = yaml.load(f.read(), Loader=yaml.FullLoader)

        for k, v in self.dictionary.items():
            setattr(self, k, v)

    def build_group_lookup(self, path_to_groupings):

        channel_group_lookup = {}

        with open(path_to_groupings, "r") as f:
            groupings = json.loads(f.read())

            for subsystem in groupings.keys():
                for subgroup in groupings[subsystem].keys():
                    for chan in groupings[subsystem][subgroup]:
                        channel_group_lookup[chan["key"]] = {}
                        channel_group_lookup[chan["key"]]["subsystem"] = subsystem
                        channel_group_lookup[chan["key"]]["subgroup"] = subgroup

        return channel_group_lookup


def make_dirs(_id):
    '''Create directories for storing data in repo (using datetime ID) if they don't already exist'''

    config = Config("config.yaml")

    if not config.train or not config.predict:
        if not os.path.isdir('data/%s' %config.use_id):
            raise ValueError("Run ID {} is not valid. If loading prior models or predictions, must provide valid ID.".format(_id))

    paths = ['data', 'data/%s' %_id, 'data/logs', 'data/%s/models' %_id, 'data/%s/smoothed_errors' %_id, 'data/%s/y_hat' %_id]

    for p in paths:
        if not os.path.isdir(p):
            os.mkdir(p)

def setup_logging():
    '''Configure logging object to track parameter settings, training, and evaluation.
    
    Args:
        config(obj): Global object specifying system runtime params.

    Returns:
        logger (obj): Logging object
        _id (str): Unique identifier generated from datetime for storing data/models/results
    '''

    logger = logging.getLogger('telemanom')
    logger.setLevel(logging.INFO)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    logger.addHandler(stdout)

    return logger


class Clustering:
    def __init__(self, labels_path=None):
        """
            Helper Class to manage the clustering of channels

            Args:
            labels_path (str): path to .csv containing labeled anomaly ranges
                for group of channels to be processed

            Attributes:
            labels_path (str): see Args
            chan_df (dataframe): holds all channel information from labels .csv
        """

        self.labels_path = labels_path
        self.chan_df = None

        if self.labels_path:
            self.chan_df = pd.read_csv(labels_path)
        else:
            chan_ids = [x.split('.')[0] for x in os.listdir('./data/test/')]
            self.chan_df = pd.DataFrame({"chan_id": chan_ids})

        logger.info("{} channels found for processing."
                    .format(len(self.chan_df)))

    def align_channels(self):
        directory = 'data/test/'
        # load times series in dict
        time_series = {}
        for filename in os.listdir(directory):
            if filename.endswith(".npy"):
                channel_id = filename.split('.')[0]  # Estrarre l'ID del canale dal nome del file
                file_path = os.path.join(directory, filename)
                time_series[channel_id] = np.load(file_path)

        max_length = max(len(series) for series in time_series.values())
        aligned_time_series = {}
        for channel_id, series in time_series.items():
            aligned_series = np.resize(series, max_length)
            aligned_time_series[channel_id] = aligned_series

        return aligned_time_series

    def perform_correlation(self, aligned_time_series):
        correlation_matrix = np.zeros((len(aligned_time_series), len(aligned_time_series)))
        for i, (channel_id1, series1) in enumerate(aligned_time_series.items()):
            for j, (channel_id2, series2) in enumerate(aligned_time_series.items()):
                correlation, _ = pearsonr(series1, series2)
                correlation_matrix[i, j] = correlation

        return correlation_matrix

    def make_clustering(self, correlation_matrix, num_clusters=5):
        scaler = StandardScaler()
        correlation_matrix_scaled = scaler.fit_transform(correlation_matrix)

        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(correlation_matrix_scaled)
        cluster_labels = kmeans.labels_

        return cluster_labels

    def make_dict_cluster_channels(self, aligned_time_series, cluster_labels):
        series_cluster_map = {}
        for channel_id, cluster_label in zip(aligned_time_series.keys(), cluster_labels):
            if cluster_label not in series_cluster_map:
                series_cluster_map[cluster_label] = []
            series_cluster_map[cluster_label].append(channel_id)

        return series_cluster_map

    def clustering_channels(self, num_clusters=5):
        aligned_time_series = self.align_channels()
        correlation_matrix = self.perform_correlation(aligned_time_series)
        cluster_labels = self.make_clustering(correlation_matrix, num_clusters=num_clusters)
        series_cluster_map = self.make_dict_cluster_channels(aligned_time_series, cluster_labels)

        # Remove "T-10" from series_cluster_map to align with clustering.chan_df['chan_id'].values
        for cluster_label, cluster_channels in series_cluster_map.items():
            if 'T-10' in cluster_channels:
                cluster_channels.remove('T-10')

        return series_cluster_map


if __name__ == '__main__':
    # Test the clustering
    clustering = Clustering(labels_path='labeled_anomalies.csv')
    series_cluster_map = clustering.clustering_channels()
    print("Result: ", series_cluster_map)

    # Test the match between series_cluster_map and  clustering.chan_df ( channel_id vs ['chan_id'] )
    # Check if DataFrame is empty
    if clustering.chan_df.empty:
        print("DataFrame is empty. Unable to proceed.")
    else:
        tot_loop = 0
        for i, (cluster_label, cluster_channels) in enumerate(series_cluster_map.items()):
            print('\n Processing Cluster # {}:'.format(cluster_label))
            # Inner loop (channels)
            for j, channel_id in enumerate(cluster_channels):
                tot_loop += 1
                print('\n Loop: # {},  Cluster # {}, Stream # {}'.format(tot_loop, cluster_label, channel_id))
                # Check if channel_id exists in DataFrame
                if channel_id not in clustering.chan_df['chan_id'].values:
                    print(f"Channel ID {channel_id} does not exist in the DataFrame.")
                else:
                    # Access DataFrame with proper error handling
                    try:
                        label_row = clustering.chan_df[clustering.chan_df['chan_id'] == channel_id].iloc[0]
                        #print("label_row: ", label_row)
                    except IndexError as e:
                        print(f"Error accessing DataFrame: {e}")
                        sys.exit()

    print("\n -------------------------------------\n ")
    for i, row in clustering.chan_df.iterrows():
        print('Stream in Data Frame # {}: {}'.format(i + 1, row.chan_id))

    print("\n -------------------------------------\n ")

    # Check for any IDs in series_cluster_map that are not present in clustering.chan_df
    for cluster_label, cluster_channels in series_cluster_map.items():
        for channel_id in cluster_channels:
            if channel_id not in clustering.chan_df['chan_id'].values:
                print(f"Channel ID {channel_id} does not exist in clustering.chan_df.")


