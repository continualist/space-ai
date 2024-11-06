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
        washout: int = 0,
    ):
        """Initialize the LSTM model.

        Args:
            input_size (int): Number of features in the input data.
            hidden_sizes (List[int]): List of hidden layer sizes.
            output_size (int): Number of features in the output data.
            reduce_out (Optional[Literal["first", "mean"]], optional): Whether to reduce the output. Defaults to None.
            dropout (float): Dropout rate.
            device (Literal["cpu", "cuda"], optional): Device to use. Defaults to "cpu".
            stateful (bool, optional): Whether to use stateful LSTM. Defaults to False.
            washout (int, optional): Number of time steps to washout. Defaults to 0.
        """
        super().__init__(
            device, stateful=stateful, reduce_out=reduce_out, washout=washout
        )
        self.input_size: int = input_size
        self.hidden_sizes: List[int] = hidden_sizes
        self.output_size: int = output_size
        self.dropout: float = dropout

    def build_fn(self):
        return _LSTM(
            self.input_size,
            self.hidden_sizes,
            self.output_size,
            self.dropout,
            washout=self.washout,
        )


class _LSTM(nn.Module):
    """LSTM model for predicting time series data."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float,
        washout: int = 0,
    ):
        """Initialize the LSTM model.

        Args:
            input_size (int): Number of features in the input data.
            hidden_sizes (List[int]): List of hidden layer sizes.
            output_size (int): Number of features in the output data.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
            washout (int, optional): Number of time steps to washout. Defaults to 0.
        """
        super().__init__()
        self.input_size: int = input_size
        self.hidden_sizes: List[int] = hidden_sizes
        self.lstm_layers = nn.ModuleList(
            [
                _LSTMDropoutLayer(
                    input_size if i == 0 else hidden_sizes[i - 1],
                    hidden_sizes[i],
                    dropout,
                )
                for i in range(len(hidden_sizes))
            ]
        )
        self.fc: nn.Linear = nn.Linear(hidden_sizes[-1], output_size)
        self.washout: int = washout

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
        h = x
        states = []
        for i, lstm in enumerate(self.lstm_layers):
            if return_states:
                h, state = lstm(
                    h, initial_state[i] if initial_state else None, return_states=True
                )
                states.append(state)
            else:
                h = lstm(h, initial_state[i] if initial_state else None)
            if i == len(self.lstm_layers) - 1:
                h = h[self.washout :]
        out = self.fc(h)
        if return_states:
            return out, states
        return out


class _LSTMDropoutLayer(nn.Module):
    """LSTM layer with dropout."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float,
    ):
        """Initialize the LSTM layer.

        Args:
            input_size (int): Number of features in the input data.
            output_size (int): Number of features in the output data.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.input_size: int = input_size
        self.lstm = nn.LSTM(input_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[List[torch.Tensor]] = None,
        return_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass of the LSTM.

        Args:
            x (torch.Tensor): Input data.
            initial_state (Optional[List[torch.Tensor]]): Initial hidden state.
            return_states (bool): Whether to return hidden states.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]: Output data.
        """
        h, state = self.lstm(x, initial_state)
        h = self.dropout(h)
        if return_states:
            return h, state
        return h


__all__ = ["LSTM"]
