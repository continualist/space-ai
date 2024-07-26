"""Model class for training and predicting with LSTM or ESN models."""

import functools
import logging
import os
from typing import List

import numpy as np
import torch
from telemanom.channel import Channel
from telemanom.helpers import Config
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


def ESN(reserviors: List[esn.Reservoir]):  # pylint: disable=invalid-name
    """Echo State Network model for predicting time series data."""
    return lambda x: functools.reduce(lambda res, f: f(res), reserviors, x)[:, -1:, :]


def mse(y_pred: torch.Tensor, y_target: torch.Tensor) -> float:
    """Mean squared error loss function."""
    return ((y_pred - y_target) ** 2).mean().item()


class LSTMModel(nn.Module):
    """LSTM model for predicting time series data."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_sizes: List[int] = hidden_sizes
        self.num_layers: int = num_layers
        self.lstms: nn.ModuleList = nn.ModuleList()
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
        self.fc: nn.Linear = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LSTM model."""
        for lstm in self.lstms:
            out, _ = lstm(x)
            x = out
        out = self.fc(out[:, -1, :])
        return out


class Model:  # pylint: disable=too-many-instance-attributes
    """Model class for training and predicting with LSTM or ESN models."""

    def __init__(self, config: Config, run_id: str, channel: Channel):
        self.config: Config = config
        self.chan_id: str = channel.id
        self.run_id: str = run_id
        self.y_hat: np.ndarray = np.array([])
        self.model: nn.Module
        self.input_size: int = channel.X_train.shape[2]
        self.device: torch.device = (
            torch.device(f"cuda:{config.cuda_id}")
            if torch.cuda.is_available() and config.cuda_id is not None
            else torch.device("cpu")
        )

        if self.config.model_architecture == "LSTM":
            self.model = LSTMModel(
                input_size=self.input_size,
                hidden_sizes=self.config.layers,
                output_size=self.config.n_predictions,
                num_layers=len(self.config.layers),
                dropout=self.config.dropout,
            )

            self.optimizer: optim.Adam = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            self.valid_loss: float

        self.train_new(channel)

    def train_new(  # pylint: disable=too-many-branches,inconsistent-return-statements
        self, channel: Channel
    ):
        """Train LSTM model according to specifications in config.yaml.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
            esn (bool): flag indicating an echo state network model
        """
        if self.config.model_architecture == "LSTM":
            self.model = self.model.to(self.device)
            criterion: nn.MSELoss = nn.MSELoss()
            if channel.train_with_val:
                best_val_loss: float = float("inf")
            epochs_since_improvement: int = 0
            with tqdm(total=self.config.epochs) as pbar:
                for epoch in range(self.config.epochs):
                    self.model.train()  # Set the model to training mode
                    train_loss: float = 0.0
                    for inputs, targets in channel.train_loader:
                        inputs, targets = inputs.to(self.device), targets.to(
                            self.device
                        )
                        self.optimizer.zero_grad()
                        outputs: torch.Tensor = self.model(inputs)
                        loss: torch.Tensor = criterion(outputs, targets)
                        loss.backward()
                        self.optimizer.step()
                        train_loss += loss.item() * inputs.size(0)

                    # Calculate average loss for the epoch
                    train_loss /= len(channel.train_loader)

                    if channel.train_with_val:
                        # Validate the model
                        self.model.eval()  # Set the model to evaluation mode
                        valid_loss: float = 0.0
                        with torch.no_grad():
                            for inputs, targets in channel.valid_loader:
                                inputs, targets = inputs.to(self.device), targets.to(
                                    self.device
                                )
                                outputs = self.model(inputs)
                                loss = criterion(outputs, targets)
                                valid_loss += loss.item() * inputs.size(0)

                        # Calculate average loss for the validation set
                        valid_loss /= len(channel.valid_loader)

                        pbar.set_description(
                            f"Epoch [{epoch+1}/{self.config.epochs}], "
                            f"Train Loss: {train_loss:.4f}, "
                            f"Valid Loss: {valid_loss:.4f}"
                        )

                        if valid_loss < best_val_loss:
                            best_val_loss = valid_loss
                            epochs_since_improvement = 0
                        else:
                            epochs_since_improvement += 1

                        if epochs_since_improvement >= self.config.patience:
                            logger.info("Early stopping at epoch %s", epoch)
                            break

                    pbar.update(1)

                if channel.train_with_val:
                    self.valid_loss = best_val_loss
        else:
            self.reserviors: List[esn.Reservoir] = []
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
                self.W_readout, best_mse = (  # pylint: disable=invalid-name
                    ridge_regression.fit_and_validate_readout(
                        channel.train_loader,
                        channel.valid_loader,
                        self.config.l2,
                        mse,
                        "min",
                        None,
                        ESN(self.reserviors),
                        str(self.device),
                    )
                )
                return {"MSE": best_mse}

            self.W_readout = ridge_regression.fit_readout(
                channel.train_loader,
                ESN(self.reserviors),
                self.config.l2,
                None,
                str(self.device),
            )

            return

    def aggregate_predictions(self, y_hat_batch: np.ndarray, method: str = "first"):
        """Aggregates predictions for each timestep. When predicting n steps ahead where
        n > 1, will end up with multiple predictions for a timestep.

        Args:
            y_hat_batch (np.ndarray): predictions shape (<batch length>, <n_preds)
            method (str): indicates how to aggregate for a timestep - "first"
                or "mean"
        """

        agg_y_hat_batch: np.ndarray = np.array([])

        for t in range(len(y_hat_batch)):

            start_idx: int = t - self.config.n_predictions
            start_idx = start_idx if start_idx >= 0 else 0

            # predictions pertaining to a specific timestep lie along diagonal
            y_hat_t: np.ndarray = (
                np.flipud(y_hat_batch[start_idx : t + 1]).reshape(1, -1).diagonal()
            )

            if method == "first":
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == "mean":
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))

        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        self.y_hat = np.append(self.y_hat, agg_y_hat_batch)

    def batch_predict(self, path: str, channel: Channel) -> Channel:
        """Used trained LSTM model to predict test data arriving in batches.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Returns:
            channel (obj): Channel class object with y_hat values as attribute
        """

        num_batches: int = int(
            (channel.y_test.shape[0] - self.config.l_s) / self.config.batch_size
        )
        if num_batches < 0:
            raise ValueError(
                f"l_s ({self.config.l_s}) too large for stream length {channel.y_test.shape[0]}."
            )

        # predict each batch
        for (
            X_test_batch,  # pylint: disable=invalid-name
            y_hat_batch,
        ) in channel.test_loader:
            X_test_batch, y_hat_batch = X_test_batch.to(  # pylint: disable=invalid-name
                self.device
            ), y_hat_batch.to(self.device)
            if self.config.model_architecture == "LSTM":
                y_hat_batch = self.model(X_test_batch).detach().cpu().numpy()
            else:
                all_W: List[torch.Tensor] = [  # pylint: disable=invalid-name
                    w.to(self.device) for w in self.W_readout
                ]
                # Processing x
                x: torch.Tensor = ESN(self.reserviors)(X_test_batch)
                size_x: torch.Size = x.size()
                if len(size_x) > 2:
                    x = x.reshape(size_x[0], -1)

                for _, W in enumerate(all_W):  # pylint: disable=invalid-name
                    y_hat_batch = torch.matmul(x.to(W), W.t()).cpu()

            self.aggregate_predictions(y_hat_batch.numpy(), "mean")
        self.y_hat = np.reshape(self.y_hat, (self.y_hat.size,))

        channel.y_hat = self.y_hat
        result_path: str = os.path.join(path, self.run_id, "y_hat")
        os.makedirs(result_path, exist_ok=True)
        np.save(os.path.join(result_path, f"{self.chan_id}.npy"), self.y_hat)

        return channel
