from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

if TYPE_CHECKING:
    from torch import Size, Tensor
    from torch.nn.modules import Module
    from torch.utils.data import DataLoader

import numpy as np
import torch
from torchrc.models.esn import EchoStateNetwork

from spaceai.models.predictors.seq_model import SequenceModel


class ESN(SequenceModel):

    def __init__(
        self,
        input_size: int,
        layers: List[int],
        output_size: int,
        reduce_out: Optional[Literal["first", "mean"]] = None,
        arch_type: Literal["stacked", "multi"] = "stacked",
        activation: str = "tanh",
        leakage: float = 1.0,
        input_scaling: float = 0.1,
        rho: float = 0.99,
        bias: bool = False,
        kernel_initializer: Union[str, Callable[[Size], Tensor]] = "uniform",
        recurrent_initializer: Union[str, Callable[[Size], Tensor]] = "uniform",
        net_gain_and_bias: bool = False,
        gradient_based: bool = False,
        device: Literal["cpu", "cuda"] = "cpu",
        return_sequences: bool = True,
        stateful: bool = False,
    ):
        """Initialize the ESN model.

        Args:
            input_size (int): Number of features in the input data.
            layers (List[int]): List of hidden layer sizes.
            output_size (int): Number of features in the output data.
            arch_type (Literal["stacked", "multi"], optional): Type of ESN architecture. Defaults to "stacked".
            activation (str, optional): Name of the activation function from `torch`. Defaults to "tanh".
            leakage (float, optional): The value of the leaking parameter `alpha`. Defaults to 1.0.
            input_scaling (float, optional): The value for the desired scaling of the input (must be `<= 1`). Defaults to 0.9.
            rho (float, optional): The desired spectral radius of the recurrent matrix (must be `< 1`). Defaults to 0.99.
            bias (bool, optional): If ``False``, the layer does not use bias weights `b`. Defaults to False.
            kernel_initializer (Union[str, Callable[[Size], Tensor]], optional): The kind of initialization of the input transformation. Defaults to "uniform".
            recurrent_initializer (Union[str, Callable[[Size], Tensor]], optional): The kind of initialization of the recurrent matrix. Defaults to "normal".
            net_gain_and_bias (bool, optional): If ``True``, the network uses additional ``g`` (gain) and ``b`` (bias) parameters. Defaults to False.
            device (Literal["cpu", "cuda"], optional): Device to move the model to. Defaults to "cpu".
        """
        super().__init__(device, stateful=stateful)
        self.model: EchoStateNetwork
        self.input_size: int = input_size
        self.layers: List[int] = layers
        self.output_size: int = output_size
        self.reduce_out: Optional[Literal["first", "mean"]] = reduce_out
        self.arch_type: Literal["stacked", "multi"] = arch_type
        self.activation: str = activation
        self.leakage: float = leakage
        self.input_scaling: float = input_scaling
        self.rho: float = rho
        self.bias: bool = bias
        self.gradient_based: bool = gradient_based
        self.return_sequences: bool = return_sequences
        self.kernel_initializer: Union[str, Callable[[Size], Tensor]] = (
            kernel_initializer
        )
        self.recurrent_initializer: Union[str, Callable[[Size], Tensor]] = (
            recurrent_initializer
        )
        self.net_gain_and_bias: bool = net_gain_and_bias

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
            pred, self.state = self.model(
                input,
                [s[-1] for s in self.state] if self.state is not None else None,
                return_states=True,
            )

        pred = torch.clip(pred, -1, 1)
        if self.reduce_out is None:
            return pred
        elif self.reduce_out == "mean":
            return pred.mean(dim=-1)
        elif self.reduce_out == "first":
            return pred[..., 0]

        raise ValueError(f"Invalid reduce_out value: {self.reduce_out}")

    def fit(  # type: ignore[override]
        self,
        train_loader: DataLoader,
        criterion: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = torch.nn.MSELoss(),
        *args,
        valid_loader: Optional[DataLoader] = None,
        metrics: Optional[
            Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]
        ] = None,
        **kwargs,
    ):
        """Train the ESN model."""
        if self.gradient_based:
            return super().fit(  # type: ignore
                *args,
                train_loader=train_loader,
                criterion=criterion,
                valid_loader=valid_loader,
                metrics=metrics,
                **kwargs,
            )
        else:
            self.model.fit_readout(*args, train_loader=train_loader, **kwargs)

            for x, y in train_loader:
                self.model(x)

            if criterion is None:
                return []

            if metrics is None:
                metrics = {}
            metrics.update({"loss": lambda x, y: criterion(x, y).item()})
            train_metrics = {f"train_{k}": m for k, m in metrics.items()}
            metrics_results = self.evaluate(train_loader, train_metrics)
            if valid_loader is not None:
                eval_metrics = self.evaluate(valid_loader, metrics)
                metrics_results.update(eval_metrics)
            return [metrics_results]

    def build_fn(self) -> Module:
        return EchoStateNetwork(
            input_size=self.input_size,
            layers=self.layers,
            output_size=self.output_size,
            arch_type=self.arch_type,
            activation=self.activation,
            leakage=self.leakage,
            input_scaling=self.input_scaling,
            rho=self.rho,
            bias=self.bias,
            kernel_initializer=self.kernel_initializer,
            recurrent_initializer=self.recurrent_initializer,
            net_gain_and_bias=self.net_gain_and_bias,
        )
