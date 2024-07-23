import numpy as np
import os
import torch
import logging
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger('telemanom')


class Channel:
    def __init__(self, config, chan_id, train_with_val=True):
        """
        Load and reshape channel values (predicted and actual).

        Args:
            config (obj): Config object containing parameters for processing
            chan_id (str): channel id

        Attributes:
            id (str): channel id
            config (obj): see Args
            X_train (arr): training inputs with dimensions
                [timesteps, l_s, input dimensions)
            X_test (arr): test inputs with dimensions
                [timesteps, l_s, input dimensions)
            y_train (arr): actual channel training values with dimensions
                [timesteps, n_predictions, 1)
            y_test (arr): actual channel test values with dimensions
                [timesteps, n_predictions, 1)
            train (arr): train data loaded from .npy file
            test(arr): test data loaded from .npy file
        """

        self.id = chan_id
        self.config = config
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None
        self.y_hat = None
        self.train = None
        self.test = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.train_with_val = train_with_val

    def shape_data(self, arr, train=True):
        """Shape raw input streams for ingestion into LSTM or ESN. config.l_s specifies
        the sequence length of prior timesteps fed into the model at
        each timestep t.

        Args:
            arr (np array): array of input streams with
                dimensions [timesteps, 1, input dimensions]
            train (bool): If shaping training data, this indicates
                data can be shuffled
        """

        data = []
        for i in range(len(arr) - self.config.l_s - self.config.n_predictions):
            data.append(arr[i:i + self.config.l_s + self.config.n_predictions])
        data = np.array(data)

        assert len(data.shape) == 3

        if train:
            np.random.shuffle(data)
            self.X_train = data[:, :-self.config.n_predictions, :]
            self.y_train = data[:, -self.config.n_predictions:, 0]  # telemetry value is at position 0

            if self.train_with_val:
                # Split the dataset into training and validation sets based on the validation_split ratio
                valid_size = int(len(self.X_train) * self.config.validation_split)
                train_dataset = TensorDataset(torch.Tensor(self.X_train[valid_size:]), torch.Tensor(self.y_train[valid_size:]))
                valid_dataset = TensorDataset(torch.Tensor(self.X_train[:valid_size]), torch.Tensor(self.y_train[:valid_size]))
                
                # Create DataLoaders for validation sets
                self.valid_loader = DataLoader(valid_dataset, batch_size=self.config.batch_size, shuffle=False)
            else:
                train_dataset = TensorDataset(torch.Tensor(self.X_train), torch.Tensor(self.y_train))

            # Create DataLoaders for training sets
            self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            
        else:
            self.X_test = data[:, :-self.config.n_predictions, :]
            self.y_test = data[:, -self.config.n_predictions:, 0]  # telemetry value is at position 0

            test_dataset = TensorDataset(torch.Tensor(self.X_test), torch.Tensor(self.y_test))
            self.test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

    def load_data(self):
        """
        Load train and test data from local.
        """
        try:
            self.train = np.load(os.path.join("data", "train", "{}.npy".format(self.id)))
            self.test = np.load(os.path.join("data", "test", "{}.npy".format(self.id)))

        except FileNotFoundError as e:
            logger.critical(e)
            logger.critical("Source data not found, may need to add data to repo: <link>")

        self.shape_data(self.train)
        self.shape_data(self.test, train=False)
