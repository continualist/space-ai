"""Reservoir module for Echo State Networks."""

from typing import (
    Callable,
    Optional,
    Union,
)

import torch
import torch.nn.functional as F
from torch import (
    Size,
    Tensor,
)
from torch.nn import (
    Module,
    Parameter,
)

from . import initializers


class Reservoir(Module):  # pylint: disable=too-many-instance-attributes
    """A Reservoir of for Echo State Networks.

    Args:
        input_size: the number of expected features in the input `x`
        hidden_size: the number of features in the hidden state `h`
        activation: name of the activation function from `torch` (e.g. `torch.tanh`)
        leakage: the value of the leaking parameter `alpha`
        input_scaling: the value for the desired scaling of the input (must be `<= 1`)
        rho: the desired spectral radius of the recurrent matrix (must be `< 1`)
        bias: if ``False``, the layer does not use bias weights `b`
        mode: execution mode of the reservoir (vanilla or intrinsic plasticity)
        kernel_initializer: the kind of initialization of the input transformation.
            Default: `'uniform'`
        recurrent_initializer: the kind of initialization of the recurrent matrix.
            Default: `'normal'`
        net_gain_and_bias: if ``True``, the network uses additional ``g`` (gain)
            and ``b`` (bias) parameters. Default: ``False``
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        input_size: int,
        hidden_size: int,
        activation: str,
        leakage: float = 1.0,
        input_scaling: float = 0.9,
        rho: float = 0.99,
        bias: bool = False,
        kernel_initializer: Union[str, Callable[[Size], Tensor]] = "uniform",
        recurrent_initializer: Union[str, Callable[[Size], Tensor]] = "normal",
        net_gain_and_bias: bool = False,
    ) -> None:

        super().__init__()
        assert rho < 1 and input_scaling <= 1

        self.input_scaling = Parameter(torch.tensor(input_scaling), requires_grad=False)
        self.rho = Parameter(torch.tensor(rho), requires_grad=False)

        self.W_in = Parameter(  # pylint: disable=invalid-name
            init_params(str(kernel_initializer), scale=input_scaling)(
                [hidden_size, input_size]
            ),
            requires_grad=False,
        )
        self.W_hat = Parameter(  # pylint: disable=invalid-name
            init_params(str(recurrent_initializer), rho=rho)(
                [hidden_size, hidden_size]
            ),
            requires_grad=False,
        )
        self.b = (
            Parameter(
                init_params("uniform", scale=input_scaling)(hidden_size),
                requires_grad=False,
            )
            if bias
            else None
        )
        self.f = getattr(torch, activation)

        self.alpha = Parameter(torch.tensor(leakage), requires_grad=False)

        self.net_gain_and_bias = net_gain_and_bias
        if net_gain_and_bias:
            self.net_a = Parameter(init_params("ones")(hidden_size), requires_grad=True)
            self.net_b = Parameter(
                init_params("zeros")(hidden_size), requires_grad=True
            )

        self._aux_fwd_comp = None

    @torch.no_grad()
    def forward(
        self,
        input: Tensor,  # pylint: disable=redefined-builtin
        initial_state: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Computes the forward pass of the reservoir."""
        if initial_state is None:
            initial_state = torch.zeros(self.hidden_size).to(self.W_hat)
        _fwd_comp = (
            self._state_comp if self._aux_fwd_comp is None else self._aux_fwd_comp
        )

        embeddings = torch.stack(
            [  # pylint: disable=unnecessary-comprehension
                state for state in _fwd_comp(input.to(self.W_hat), initial_state, mask)
            ],
            dim=0,
        )

        return embeddings

    def _state_comp(
        self,
        input: Tensor,  # pylint: disable=redefined-builtin
        initial_state: Tensor,
        mask: Optional[Tensor] = None,
    ):
        timesteps = input.shape[0]
        state = initial_state
        for t in range(timesteps):
            in_signal_t = F.linear(  # pylint: disable=not-callable
                input[t].to(self.W_in), self.W_in, self.b
            ) + F.linear(  # pylint: disable=not-callable
                state, self.W_hat
            )
            if self.net_gain_and_bias:
                in_signal_t = in_signal_t * self.net_a + self.net_b
            h_t = torch.tanh(in_signal_t)
            state = (1 - self.alpha) * state + self.alpha * h_t
            yield state if mask is None else mask * state

    @property
    def input_size(self) -> int:
        """Input dimension."""
        return self.W_in.shape[1]

    @property
    def hidden_size(self) -> int:
        """Reservoir state dimension."""
        return self.W_hat.shape[1]


def init_params(name: str, **options) -> Callable[[Size], Tensor]:
    """Gets a random weight initializer :param name: Name of the random matrix generator
    in `esn.initializers` :param options: Random matrix generator options :return: A
    random weight generator function."""
    init = getattr(initializers, name)
    return lambda size: init(size, **options)
