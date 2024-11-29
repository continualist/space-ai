import os
import zipfile

import requests  # type: ignore[import-untyped]
from tqdm import tqdm

from typing import (
    Optional,
    Callable,
    Literal,
)

import torch


def download_and_extract_zip(url: str, extract_to: str, cleanup: bool = False):
    """Download a zip file from a URL and extract it to a directory.

    Args:
        url (str): URL of the zip file.
        extract_to (str): Directory to extract the zip file.
        cleanup (bool): If True, the zip file is removed after extraction.
    """

    filename = download_file(url)
    extract_zip(filename, extract_to, cleanup)


def download_file(url: str, to: Optional[str] = None):
    """Download a file from a URL.

    Args:
        url (str): URL of the file.
        to (Optional[str]): Local path to save the file. If None, the file is saved in \
            the current directory.
    """

    if to is None:
        local_filename = url.split("/")[-1]
    else:
        local_filename = to

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        with open(local_filename, "wb") as f, tqdm(
            desc=local_filename,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=block_size):
                f.write(chunk)
                bar.update(len(chunk))
    return local_filename


def extract_zip(filename: str, extract_to: str, cleanup: bool = False):
    """Extract a zip file to a directory.

    Args:
        filename (str): Path to the zip file.
        extract_to (str): Directory to extract the zip file.
        cleanup (bool): If True, the zip file is removed after extraction.
    """

    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    if cleanup:
        os.remove(filename)


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
