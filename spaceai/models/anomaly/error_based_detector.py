from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
)

import numpy as np
import torch

if TYPE_CHECKING:
    from spaceai.models.predictors import SequenceModel

from .detector import AnomalyDetector


class ErrorBasedDetector(AnomalyDetector):

    def __call__(self, input: np.ndarray, y_true: np.ndarray, **kwargs) -> np.ndarray:
        y_hat = self.predict_values(input)
        return self.detect_anomalies(y_hat, y_true, **kwargs)

    def compute_error(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        reduce: Optional[Literal["mean", "min", "max"]] = None,
    ) -> np.ndarray:
        raise NotImplementedError

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

    def predict_values(
        self, input: np.ndarray, with_error: bool = True, **kwargs
    ) -> np.ndarray:
        if self._predictor is None:
            raise ValueError("Predictor must be bound before calling predict_values.")

        y_pred = self._predictor(input).detach().cpu().numpy()
        if with_error:
            return y_pred, self.compute_error(input, y_pred)
        return y_pred

    def fit(self, *args, **kwargs):
        pass


__all__ = ["ErrorBasedDetector"]
