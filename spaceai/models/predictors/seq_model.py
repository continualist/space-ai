import copy
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class SequenceModel:
    """Base class for generic sequence models.

    This class provides a common interface for training and evaluating sequence models.
    The developer should subclass this class and implement the `build_fn` method to define
    the model architecture and initialization. Optionally, the developer can override the
    `fit` and `evaluate` methods to customize the training and evaluation loops.
    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cpu",
        stateful: bool = False,
        reduce_out: Optional[Literal["first", "mean"]] = None,
        washout: int = 0,
    ):
        self._device: Literal["cpu", "cuda"] = device
        self.model: Optional[torch.nn.Module] = None
        self.stateful: bool = stateful
        self.reduce_out: Optional[Literal["first", "mean"]] = reduce_out
        self.washout: int = washout
        self.state: Optional[List[torch.Tensor]] = None

    def build_fn(self) -> torch.nn.Module:
        """Build the model."""
        raise NotImplementedError(
            "Model build function not defined. When subclassing "
            "SequenceModel, define a build_fn method."
        )

    def build(self):
        """Rebuild the model with new parameters."""
        self.model = self.build_fn()
        self.model.to(self.device)

    def __call__(self, input: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Predicts values using the trained model.

        Args:
            input (Union[torch.Tensor, np.ndarray]): Input data to predict on

        Returns:
            torch.Tensor: Predicted values
        """
        if self.model is None:
            raise ValueError("Model must be built before calling predict.")

        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input).float()

        if not self.stateful:
            pred = self.model(input)
        else:
            pred, self.state = self.model(input, self.state, return_states=True)

        if self.reduce_out is None:
            return pred
        elif self.reduce_out == "mean":
            orig_pred = pred.clone()
            for i in range(1, pred.shape[-1]):
                pred[i:, ..., i] = orig_pred[:-i, ..., i]
            startpred = torch.stack(
                [pred[i - 1, ..., :i].mean(dim=-1) for i in range(1, pred.shape[-1])]
            )
            endpred = pred[pred.shape[-1] - 1 :].mean(dim=-1)
            out = torch.cat([startpred, endpred], dim=0)
            return out
        elif self.reduce_out == "first":
            return pred[..., 0]

        raise ValueError(f"Invalid reduce_out value: {self.reduce_out}")

    def fit(
        self,
        train_loader: DataLoader,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        epochs: int,
        patience_before_stopping: Optional[int] = None,
        min_delta: Optional[float] = None,
        valid_loader: Optional[DataLoader] = None,
        metrics: Optional[
            Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]
        ] = None,
        restore_best: bool = True,
    ):
        """Train the model on the given data.

        Args:
            train_loader (DataLoader): DataLoader containing training data
            criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Loss function
            optimizer (torch.optim.Optimizer): Optimizer
            epochs (int): Number of epochs to train the model
            patience_before_stopping (Optional[int], optional): Number of epochs to wait before stopping training if no improvement. Defaults to None.
            valid_loader (Optional[DataLoader], optional): DataLoader containing validation data. Defaults to None.
            metrics (Optional[Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]], optional): Dictionary of metric functions. Defaults to None.
        """
        if self.model is None:
            raise ValueError("Model must be built before calling fit.")
        min_delta_ = min_delta if min_delta is not None else 0.0
        self.model = self.model.to(self.device)
        best_val_loss = float("inf")
        best_model = copy.deepcopy(self.model.state_dict())
        history: List[Dict[str, float]] = []
        metrics_ = metrics if metrics is not None else {}
        metrics_.update({"loss": lambda x, y: criterion(x, y).item()})
        with tqdm(total=epochs) as pbar:
            for epoch in range(epochs):
                self.model.train()  # Set the model to training mode
                epoch_metrics: Dict[str, Any] = {
                    f"{name}_train": 0.0 for name in metrics_
                }
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    if self.washout is not None:
                        targets = targets[self.washout :]
                        outputs = outputs[-len(targets) :]
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        for name, metric in metrics_.items():
                            if name == "loss":
                                epoch_metrics["loss_train"] += loss.item()
                            else:
                                epoch_metrics[f"{name}_train"] += metric(
                                    outputs, targets
                                ) * inputs.size(1)
                for name in epoch_metrics:
                    epoch_metrics[name] /= len(train_loader.dataset)
                if valid_loader is not None:
                    eval_metrics = self.evaluate(valid_loader, metrics_)
                    epoch_metrics.update(eval_metrics)
                else:
                    epoch_metrics["loss_eval"] = epoch_metrics["loss_train"]
                history.append(epoch_metrics)

                if epoch_metrics["loss_eval"] < best_val_loss - min_delta_:
                    best_val_loss = epoch_metrics["loss_eval"]
                    epochs_since_improvement = 0
                    best_model = copy.deepcopy(self.model.state_dict())
                else:
                    epochs_since_improvement += 1

                pbar.set_description(
                    f"Epoch [{epoch+1}/{epochs}], "
                    + f"Train Loss: {epoch_metrics['loss_train']:.4f}, "
                    + f"Valid Loss: {epoch_metrics['loss_eval']:.4f}"
                )

                pbar.update(1)
                if patience_before_stopping is not None:
                    if epochs_since_improvement >= patience_before_stopping:
                        print("Early stopping at epoch %s", epoch)
                        break
        if restore_best:
            self.model.load_state_dict(best_model)
        return history

    def evaluate(
        self,
        eval_loader: DataLoader,
        metrics: Optional[Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]],
    ) -> Dict[str, float]:
        """Evaluate the model on the given data.

        Args:
            eval_loader (DataLoader): DataLoader containing evaluation data
            metrics (Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]): Dictionary of metric functions

        Returns:
            Dict[str, float]: Dictionary containing metric values
        """
        if self.model is None:
            raise ValueError("Model must be built before calling evaluate.")
        self.model.eval()  # Set the model to evaluation mode
        metrics_ = metrics if metrics is not None else {}
        metrics_values: Dict[str, float] = {f"{name}_eval": 0.0 for name in metrics_}
        with torch.no_grad():
            for inputs, targets in eval_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self(inputs)
                if self.reduce_out is not None:
                    targets = targets[..., -1]
                for name, metric in metrics_.items():
                    metrics_values[f"{name}_eval"] += metric(
                        outputs, targets
                    ) * inputs.size(1)

        for name in metrics_values:
            metrics_values[name] /= len(eval_loader.dataset)

        return metrics_values

    def reset_state(self):
        """Reset the state of the model."""
        self.state = None

    def save(self, path: str):
        """Save the model to the given path."""
        if self.model is None:
            raise ValueError("Model must be built before saving.")
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """Load the model from the given path."""
        if self.model is None:
            raise ValueError("Model must be built before loading.")
        self.model.load_state_dict(torch.load(path))

    @property
    def device(self) -> Literal["cpu", "cuda"]:
        """Return the device the model is on."""
        return self._device

    @device.setter
    def device(self, device: Literal["cpu", "cuda"]):
        """Set the device for the model."""
        self._device = device
        if self.model is not None:
            self.model.to(self._device)
