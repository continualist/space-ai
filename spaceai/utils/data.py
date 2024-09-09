from typing import (
    Callable,
    Literal,
)

import torch


def seq_collate_fn(
    n_inputs: int = 2, mode: Literal["batch", "time"] = "batch"
) -> Callable:
    """Collate function for sequence data. It stacks sequences of tensors along dim=1.

    Args:
        n_inputs (int): Number of input tensors to stack (includes the target). Check
            the `__getitem__` method of the dataset to see the order of tensors.
        mode (Literal["batch", "time"]): Mode to stack the sequences. If "batch", the
            sequences are stacked along the batch dimension. If "time", the sequences
            are stacked along the time dimension.

    Returns:
        Callable: Collate function for DataLoader.
    """
    if mode == "time":

        def collate_fn(batch):
            """Collate function for sequence data."""
            inputs = [[] for _ in range(n_inputs)]
            for item in batch:
                for i in range(n_inputs):
                    inputs[i].append(item[i])
            inputs = [torch.cat(seq, dim=0).unsqueeze(1) for seq in inputs]
            return inputs

    if mode == "batch":

        def collate_fn(batch):
            """Collate function for sequence data."""
            inputs = [[] for _ in range(n_inputs)]
            for item in batch:
                for i in range(n_inputs):
                    inputs[i].append(item[i])
            inputs = [torch.stack(seq, dim=1) for seq in inputs]
            return inputs

    return collate_fn
