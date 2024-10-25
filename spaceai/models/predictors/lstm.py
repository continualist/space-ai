from typing import (
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import torch
from torch import nn

from .seq_model import SequenceModel


class LSTM(SequenceModel):

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        reduce_out: Optional[Literal["first", "mean"]] = None,
        dropout: float = 0.3,
        device: Literal["cpu", "cuda"] = "cpu",
        stateful: bool = False,
    ):
        """Initialize the LSTM model.

        Args:
            input_size (int): Number of features in the input data.
            hidden_sizes (List[int]): List of hidden layer sizes.
            output_size (int): Number of features in the output data.
            reduce_out (Optional[Literal["first", "mean"]], optional): Whether to reduce the output. Defaults to None.
            dropout (float): Dropout rate.
        """
        super().__init__(device, stateful=stateful)
        self.input_size: int = input_size
        self.hidden_sizes: List[int] = hidden_sizes
        self.output_size: int = output_size
        self.reduce_out: Optional[Literal["first", "mean"]] = reduce_out
        self.dropout: float = dropout

    def build_fn(self):
        return _LSTM(self.input_size, self.hidden_sizes, self.output_size, self.dropout)

    def __call__(self, input):
        pred = super().__call__(input)
        if self.reduce_out is None:
            return pred
        elif self.reduce_out == "mean":
            return pred.mean(dim=-1)
        elif self.reduce_out == "first":
            return pred[..., 0]

        raise ValueError(f"Invalid reduce_out value: {self.reduce_out}")


class _LSTM(nn.Module):
    """LSTM model for predicting time series data."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float,
    ):
        """Initialize the LSTM model.

        Args:
            input_size (int): Number of features in the input data.
            hidden_sizes (List[int]): List of hidden layer sizes.
            output_size (int): Number of features in the output data.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.input_size: int = input_size
        self.hidden_sizes: List[int] = hidden_sizes
        self.lstm: nn.LSTM = nn.LSTM(
            input_size,
            hidden_sizes[-1],
            num_layers=len(hidden_sizes),
            dropout=dropout,
        )
        self.fc: nn.Linear = nn.Linear(hidden_sizes[-1], output_size)

    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[List[torch.Tensor]] = None,
        return_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input data.
            initial_state (Optional[List[torch.Tensor]]): Initial hidden state.
            return_states (bool): Whether to return hidden states.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]: Output data.
        """
        h, states = (
            self.lstm(x, initial_state) if initial_state is not None else self.lstm(x)
        )
        out = self.fc(h)
        if return_states:
            return out, states
        return out


__all__ = ["LSTM"]
