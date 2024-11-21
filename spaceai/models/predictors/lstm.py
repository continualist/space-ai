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
        self.reduce_out: Optional[Literal["first", "mean"]] = reduce_out
        self.dropout: float = dropout

    def build_fn(self):
        return _LSTM(
            self.input_size,
            self.hidden_sizes,
            self.output_size,
            self.dropout,
            washout=self.washout,
        )

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
        washout: int = 0,
        unit_forget_bias: bool = True,
    ):
        """Initialize the LSTM model.

        Args:
            input_size (int): Number of features in the input data.
            hidden_sizes (List[int]): List of hidden layer sizes.
            output_size (int): Number of features in the output data.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
            washout (int, optional): Number of time steps to washout. Defaults to 0.
            unit_forget_bias (bool, optional): Whether to use unit forget bias. Defaults to True.
        """
        super().__init__()
        self.input_size: int = input_size
        self.hidden_sizes: List[int] = hidden_sizes
        self.unit_forget_bias: bool = unit_forget_bias
        self.washout: int = washout
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
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.LSTM):
                for name, param in layer.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias_ih" in name:
                        nn.init.zeros_(param.data)
                        if self.unit_forget_bias:
                            hs = param.size(0) // 4
                            param.data[hs : hs * 2].fill_(1)
                    elif "bias_hh" in name:
                        nn.init.zeros_(param.data)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

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
            washout_layer = 0
            if self.training and i == len(self.lstm_layers) - 1:
                washout_layer = self.washout
            h = lstm(
                h,
                initial_state[i] if initial_state else None,
                return_states=return_states,
                washout_layer=washout_layer,
            )
            if return_states:
                h, state = h
                states.append(state)
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
        washout_layer: int = 0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass of the LSTM.

        Args:
            x (torch.Tensor): Input data.
            initial_state (Optional[List[torch.Tensor]]): Initial hidden state.
            return_states (bool): Whether to return hidden states.
            washout_layer (Optional[int]): Number of time steps to washout.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]: Output data.
        """
        h, state = self.lstm(x, initial_state)
        if washout_layer is not None:
            h = h[washout_layer:]
        h = self.dropout(h)
        if return_states:
            return h, state
        return h


__all__ = ["LSTM"]
