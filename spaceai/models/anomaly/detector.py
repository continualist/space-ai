from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Dict,
    Literal,
    Optional,
    Union,
)

import numpy as np
import torch

if TYPE_CHECKING:
    from spaceai.models.predictors import SequenceModel


class AnomalyDetector:

    def __init__(self):
        self._predictor: Optional[SequenceModel] = None
        self.ignore_first_n_factor: float = 0

    def __call__(self, input: np.ndarray, y_true: np.ndarray, **kwargs) -> np.ndarray:
        """Detect anomalies in the input data.

        Args:
            input (np.ndarray): Input data
            y_true (np.ndarray): True values
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: Detected anomalies
        """
        y_hat = self.predict_values(input)
        return self.detect_anomalies(y_hat, y_true, **kwargs)

    def bind_predictor(
        self,
        predictor: SequenceModel,
        device: Literal["cpu", "cuda"] = "cpu",
    ):
        """Bind a predictor to the framework. If the predictor is a torch model, move it
        to the specified device.

        Args:
            predictor (Callable[..., np.ndarray]): Predictor function
            device (Literal["cpu", "cuda"], optional): Device to move the predictor to.
                Defaults to "cpu".
        """
        self._predictor = predictor
        if isinstance(self._predictor, torch.nn.Module):
            self._predictor.to(device)

    def predict_values(self, input: np.ndarray, **kwargs) -> np.ndarray:
        """Predict values using the bound predictor.

        Args:
            input (np.ndarray): Input data to predict on

        Returns:
            np.ndarray: Predicted values
        """
        if self._predictor is None:
            raise ValueError("Predictor must be bound before calling predict_values.")
        return self._predictor(input).detach().cpu().numpy()

    def detect_anomalies(
        self, y_pred: np.ndarray, y_true: np.ndarray, **kwargs
    ) -> np.ndarray:
        raise NotImplementedError

    def flush_detector(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def evaluate_anomalies(
        self, y_pred: np.ndarray, y_true: np.ndarray, **kwargs
    ) -> Dict[str, Union[int, float]]:
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        pass


__all__ = ["AnomalyDetector"]
