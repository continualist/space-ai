"""Model class for training and predicting with LSTM or ESN models."""

import functools
import logging
import os
from typing import List

import numpy as np
import torch
from torch import (
    nn,
    optim,
)
from torch_esn.model import reservoir as esn
from torch_esn.optimization import ridge_regression
from tqdm import tqdm

# suppress PyTorch CPU speedup warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
logger = logging.getLogger("telemanom")


def ESN(reserviors):
    """Echo State Network model for predicting time series data."""
    return lambda x: functools.reduce(lambda res, f: f(res), reserviors, x)[:, -1:, :]


def mse(y_pred, y_target):
    """Mean squared error loss function."""
    return ((y_pred - y_target) ** 2).mean()


class LSTMModel(nn.Module):
    """LSTM model for predicting time series data."""

    def __init__(self, input_size, hidden_sizes, output_size, num_layers, dropout):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.lstms = nn.ModuleList()
        self.lstms.append(
            nn.LSTM(
                input_size,
                hidden_sizes[0],
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
            )
        )
        for i in range(1, len(hidden_sizes)):
            self.lstms.append(
                nn.LSTM(
                    hidden_sizes[i - 1],
                    hidden_sizes[i],
                    num_layers=num_layers,
                    dropout=dropout,
                    batch_first=True,
                )
            )
        self.fc = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        """Forward pass through the LSTM model."""
        for lstm in self.lstms:
            out, _ = lstm(x)
            x = out
        out = self.fc(out[:, -1, :])
        return out


class Model:
    """Model class for training and predicting with LSTM or ESN models."""

    def __init__(self, config, run_id, channel):
        self.config = config
        self.chan_id = channel.id
        self.run_id = run_id
        self.y_hat = np.array([])
        self.model = None
        self.esn = esn
        self.input_size = channel.X_train.shape[2]

        if torch.cuda.is_available() and config.cuda_id is not None:
            self.device = torch.device(f"cuda:{config.cuda_id}")
        else:
            self.device = torch.device("cpu")

        if self.config.model_architecture == "LSTM":
            self.model = LSTMModel(
                input_size=self.input_size,
                hidden_sizes=self.config.layers,
                output_size=self.config.n_predictions,
                num_layers=len(self.config.layers),
                dropout=self.config.dropout,
            )

            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=float(self.config.learning_rate),
                weight_decay=self.config.weight_decay,
            )
            self.valid_loss = None

        self.train_new(channel)

    def train_new(self, channel):
        """Train LSTM model according to specifications in config.yaml.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
            esn (bool): flag indicating an echo state network model
        """
        if self.config.model_architecture == "LSTM":
            self.model = self.model.to(self.device)
            criterion = nn.MSELoss()
            if channel.train_with_val:
                best_val_loss = float("inf")
            epochs_since_improvement = 0
            with tqdm(total=self.config.epochs) as pbar:
                for epoch in range(self.config.epochs):
                    self.model.train()  # Set the model to training mode
                    train_loss = 0.0
                    for inputs, targets in channel.train_loader:
                        inputs, targets = inputs.to(self.device), targets.to(
                            self.device
                        )
                        self.optimizer.zero_grad()
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        self.optimizer.step()
                        train_loss += loss.item() * inputs.size(0)

                    # Calculate average loss for the epoch
                    train_loss /= len(channel.train_loader.dataset)

                    if channel.train_with_val:
                        # Validate the model
                        self.model.eval()  # Set the model to evaluation mode
                        valid_loss = 0.0
                        with torch.no_grad():
                            for inputs, targets in channel.valid_loader:
                                inputs, targets = inputs.to(self.device), targets.to(
                                    self.device
                                )
                                outputs = self.model(inputs)
                                loss = criterion(outputs, targets)
                                valid_loss += loss.item() * inputs.size(0)

                        # Calculate average loss for the validation set
                        valid_loss /= len(channel.valid_loader.dataset)

                        pbar.set_description(
                            f"Epoch [{epoch+1}/{self.config.epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}"
                        )

                        if valid_loss < best_val_loss:
                            best_val_loss = valid_loss
                            epochs_since_improvement = 0
                        else:
                            epochs_since_improvement += 1

                        if epochs_since_improvement >= self.config.patience:
                            logger.info(f"Early stopping at epoch {epoch}")
                            break

                    pbar.update(1)

                if channel.train_with_val:
                    self.valid_loss = best_val_loss

        else:
            if not isinstance(self.config.l2, List):
                self.config.l2 = [self.config.l2]
            self.reserviors = []
            self.reserviors.append(
                esn.Reservoir(
                    self.input_size,
                    self.config.layers[0],
                    self.config.activation,
                    self.config.leakage,
                    self.config.input_scaling,
                    self.config.rho,
                    self.config.bias,
                    self.config.kernel_initializer,
                    self.config.recurrent_initializer,
                    self.config.net_gain_and_bias,
                )
            )
            for i in range(1, len(self.config.layers)):
                self.reserviors.append(
                    esn.Reservoir(
                        self.config.layers[i - 1],
                        self.config.layers[i],
                        self.config.activation,
                        self.config.leakage,
                        self.config.input_scaling,
                        self.config.rho,
                        self.config.bias,
                        self.config.kernel_initializer,
                        self.config.recurrent_initializer,
                        self.config.net_gain_and_bias,
                    )
                )

            if channel.train_with_val:
                self.W_readout, best_mse = ridge_regression.fit_and_validate_readout(
                    channel.train_loader,
                    channel.valid_loader,
                    self.config.l2,
                    mse,
                    "min",
                    None,
                    ESN(self.reserviors),
                    self.device,
                )
                return {"MSE": best_mse.item()}

            self.W_readout = ridge_regression.fit_readout(
                channel.train_loader,
                ESN(self.reserviors),
                self.config.l2,
                None,
                self.device,
            )

            return

    def aggregate_predictions(self, y_hat_batch, method="first"):
        """Aggregates predictions for each timestep. When predicting n steps ahead where
        n > 1, will end up with multiple predictions for a timestep.

        Args:
            y_hat_batch (arr): predictions shape (<batch length>, <n_preds)
            method (string): indicates how to aggregate for a timestep - "first"
                or "mean"
        """

        agg_y_hat_batch = np.array([])

        for t in range(len(y_hat_batch)):

            start_idx = t - self.config.n_predictions
            start_idx = start_idx if start_idx >= 0 else 0

            # predictions pertaining to a specific timestep lie along diagonal
            y_hat_t = np.flipud(y_hat_batch[start_idx : t + 1]).diagonal()

            if method == "first":
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == "mean":
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))

        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        self.y_hat = np.append(self.y_hat, agg_y_hat_batch)

    def batch_predict(self, path, channel):
        """Used trained LSTM model to predict test data arriving in batches.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Returns:
            channel (obj): Channel class object with y_hat values as attribute
        """

        num_batches = int(
            (channel.y_test.shape[0] - self.config.l_s) / self.config.batch_size
        )
        if num_batches < 0:
            raise ValueError(
                f"l_s ({self.config.l_s}) too large for stream length {channel.y_test.shape[0]}."
            )

        # predict each batch
        for X_test_batch, y_hat_batch in channel.test_loader:
            X_test_batch, y_hat_batch = X_test_batch.to(self.device), y_hat_batch.to(
                self.device
            )
            if self.config.model_architecture == "LSTM":
                y_hat_batch = self.model(X_test_batch).detach().cpu().numpy()
            else:
                all_W = [w.to(self.device) for w in self.W_readout]
                # Processing x
                x = ESN(self.reserviors)(X_test_batch)
                size_x = x.size()
                if len(size_x) > 2:
                    x = x.reshape(size_x[0], -1)

                for _, W in enumerate(all_W):
                    y_hat_batch = torch.matmul(x.to(W), W.t()).cpu()

            self.aggregate_predictions(y_hat_batch, "mean")
        self.y_hat = np.reshape(self.y_hat, (self.y_hat.size,))

        channel.y_hat = self.y_hat
        result_path = os.path.join(path, self.run_id, "y_hat")
        os.makedirs(result_path, exist_ok=True)
        np.save(os.path.join(result_path, f"{self.chan_id}.npy"), self.y_hat)

        return channel
