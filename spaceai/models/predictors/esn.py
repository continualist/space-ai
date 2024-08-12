from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Literal,
    Optional,
    Union,
)

if TYPE_CHECKING:
    from torch import Size, Tensor
    from torch.nn.modules import Module
    from torch.utils.data import DataLoader

from torchrc.models.esn import EchoStateNetwork

from .seq_model import SequenceModel


class ESN(SequenceModel):

    def __init__(
        self,
        input_size: int,
        layers: List[int],
        output_size: int,
        arch_type: Literal["stacked", "multi"] = "stacked",
        activation: str = "tanh",
        leakage: float = 1.0,
        input_scaling: float = 0.9,
        rho: float = 0.99,
        bias: bool = False,
        kernel_initializer: Union[str, Callable[[Size], Tensor]] = "uniform",
        recurrent_initializer: Union[str, Callable[[Size], Tensor]] = "normal",
        net_gain_and_bias: bool = False,
        device: Literal["cpu", "cuda"] = "cpu",
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
        super().__init__(device)
        self.model: EchoStateNetwork
        self.input_size: int = input_size
        self.layers: List[int] = layers
        self.output_size: int = output_size
        self.arch_type: Literal["stacked", "multi"] = arch_type
        self.activation: str = activation
        self.leakage: float = leakage
        self.input_scaling: float = input_scaling
        self.rho: float = rho
        self.bias: bool = bias
        self.kernel_initializer: Union[str, Callable[[Size], Tensor]] = (
            kernel_initializer
        )
        self.recurrent_initializer: Union[str, Callable[[Size], Tensor]] = (
            recurrent_initializer
        )
        self.net_gain_and_bias: bool = net_gain_and_bias

    def fit(  # type: ignore[override]
        self,
        train_loader: DataLoader,
        l2_value: Union[float, List[float]] = 1e-6,
        score_fn: Optional[Callable[[Tensor, Tensor], float]] = None,
        mode: Optional[Literal["min", "max"]] = None,
        eval_on: Optional[Union[Literal["train"], DataLoader]] = None,
    ):
        """Train the ESN model.

        Args:
            train_loader (DataLoader): DataLoader containing training data.
            l2_value (Union[float, List[float]], optional): The value of l2 regularization. Defaults to 1e-6.
            score_fn (Optional[Callable[[Tensor, Tensor], float]], optional): Function to calculate the score. Defaults to None.
            mode (Optional[Literal["min", "max"]], optional): The mode to optimize the score. Defaults to None.
            eval_on (Optional[Union[Literal["train"], DataLoader]], optional): DataLoader containing evaluation data. Defaults to None.
        """
        self.model.fit_readout(
            train_loader=train_loader,
            l2_value=l2_value,
            score_fn=score_fn,
            mode=mode,
            eval_on=eval_on,
        )

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
