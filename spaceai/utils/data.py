from typing import Callable

import torch


def seq_collate_fn(n_inputs: int = 2) -> Callable:
    """Collate function for sequence data. It stacks sequences of tensors along dim=1.

    Args:
        n_inputs (int): Number of input tensors to stack (includes the target). Check
            the `__getitem__` method of the dataset to see the order of tensors.

    Returns:
        Callable: Collate function for DataLoader.
    """

    def collate_fn(batch):
        """Collate function for sequence data."""
        inputs = [[] for _ in range(n_inputs)]
        for item in batch:
            for i in range(n_inputs):
                inputs[i].append(item[i])
        inputs = [torch.stack(seq, dim=1) for seq in inputs]
        return inputs

    return collate_fn
