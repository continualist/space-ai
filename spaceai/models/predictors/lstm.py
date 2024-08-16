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
        dropout: float,
        device: Literal["cpu", "cuda"] = "cpu",
    ):
        """Initialize the LSTM model.

        Args:
            input_size (int): Number of features in the input data.
            hidden_sizes (List[int]): List of hidden layer sizes.
            output_size (int): Number of features in the output data.
            dropout (float): Dropout rate.
        """
        super().__init__(device)
        self.input_size: int = input_size
        self.hidden_sizes: List[int] = hidden_sizes
        self.output_size: int = output_size
        self.dropout: float = dropout

    def build_fn(self):
        return _LSTM(self.input_size, self.hidden_sizes, self.output_size, self.dropout)


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
        h, *_ = self.lstm(x)
        out = self.fc(h)
        return out


__all__ = ["LSTM"]
