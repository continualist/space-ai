"""Module for initializing tensors in a neural network model."""

from typing import Optional

import torch
from torch import (
    Size,
    Tensor,
)

__all__ = ["uniform", "normal", "ring", "orthogonal", "ones", "zeros", "rescale_"]


def uniform(
    size: Size,
    rho: Optional[float] = None,
    sigma: Optional[float] = None,
    scale: Optional[float] = None,
) -> Tensor:
    """Uniform random tensor. Can either be rescaled according to spectral radius `rho`,
    spectral norm `sigma`, or `scale`.

    Args:
        size (Size): shape of the tensor.
        rho (Optional[float], optional): spectral radius value. Defaults to None.
        sigma (Optional[float], optional): standard deviation used for normal
            initialization. Defaults to None.
        scale (Optional[float], optional): scaling value. Defaults to None.

    Returns:
        Tensor: initialized tensor.
    """
    W = torch.empty(size).uniform_(-1, 1)
    rescale_(W, rho, sigma, scale)
    return W.data


def normal(
    size: Size,
    rho: Optional[float] = None,
    sigma: Optional[float] = None,
    scale: Optional[float] = None,
) -> Tensor:
    """Normal random tensor. Can either be rescaled according to spectral radius `rho`,
    spectral norm `sigma`, or `scale`.

    Args:
        size (Size): shape of the tensor.
        rho (Optional[float], optional): spectral radius value. Defaults to None.
        sigma (Optional[float], optional): standard deviation used for normal
            initialization. Defaults to None.
        scale (Optional[float], optional): scaling value. Defaults to None.

    Returns:
        Tensor: initialized tensor.
    """
    W = torch.empty(size).normal_(mean=0, std=1)
    rescale_(W, rho, sigma, scale)
    return W.data


def ring(
    size: Size,
    rho: Optional[float] = None,
    sigma: Optional[float] = None,
    scale: Optional[float] = None,
) -> Tensor:
    """
    Ring matrix. See:
        C. Gallicchio & A. Micheli (2020). Ring Reservoir Neural Networks for Graphs.
        In 2020 International Joint Conference on Neural Networks (IJCNN), IEEE.
        https://doi.org/10.1109/IJCNN48605.2020.9206723

    Args:
        size (Size): shape of the tensor.
        rho (Optional[float], optional): spectral radius value. Defaults to None.
        sigma (Optional[float], optional): standard deviation used for normal
            initialization. Defaults to None.
        scale (Optional[float], optional): scaling value. Defaults to None.

    Returns:
        Tensor: initialized ring tensor.
    """
    assert (len(size) == 2) and (size[0] == size[1])
    assert any(arg is not None for arg in [rho, sigma, scale])
    if scale is None:
        scale = rho if sigma is None else sigma
    W = torch.eye(size[0]).roll(1, 0) * scale
    return W.data


def orthogonal(
    size: Size,
    rho: Optional[float] = None,
    sigma: Optional[float] = None,
    scale: Optional[float] = None,
) -> Tensor:
    """
    Orthogonal matrix. See:
        F. Mezzadri (2007). How to Generate Random Matrices from the Classical Compact
        Groups. Notices of the American Mathematical Society, 54(5), pp. 592-604.
        https://www.ams.org/notices/200705/fea-mezzadri-web.pdf

    Args:
        size (Size): shape of the tensor.
        rho (Optional[float], optional): spectral radius value. Defaults to None.
        sigma (Optional[float], optional): standard deviation used for normal
            initialization. Defaults to None.
        scale (Optional[float], optional): scaling value. Defaults to None.

    Returns:
        Tensor: initialized orthogonal tensor.
    """
    assert any(arg is not None for arg in [rho, sigma, scale])
    if scale is None:
        scale = rho if sigma is None else sigma
    W = torch.empty(size)
    torch.nn.init.orthogonal_(W, scale)
    return W.data


def ones(
    size: Size,
    rho: Optional[float] = None,
    sigma: Optional[float] = None,
    scale: Optional[float] = None,
) -> Tensor:
    """Ones tensor. Can either be rescaled according to spectral radius `rho`, spectral
    norm `sigma`, or `scale`.

    Args:
        size (Size): shape of the tensor.
        rho (Optional[float], optional): spectral radius value. Defaults to None.
        sigma (Optional[float], optional): standard deviation used for normal
            initialization. Defaults to None.
        scale (Optional[float], optional): scaling value. Defaults to None.

    Returns:
        Tensor: initialized ones tensor.
    """
    W = torch.ones(size)
    rescale_(W, rho, sigma, scale)
    return W.data


def zeros(size: Size, **kwargs) -> Tensor:
    """Zeros tensor.

    Args:
        size (Size): size of the tensor

    Returns:
        Tensor: zeros tensor.
    """
    W = torch.zeros(size)
    return W.data


def rescale_(
    W: Tensor,
    rho: Optional[float] = None,
    sigma: Optional[float] = None,
    scale: Optional[float] = None,
) -> Tensor:
    """Rescale a matrix in-place. Can either be rescaled according to spectral radius
    `rho`, spectral norm `sigma`, or `scale`.

    Args:
        size (Size): shape of the tensor.
        rho (Optional[float], optional): spectral radius value. Defaults to None.
        sigma (Optional[float], optional): standard deviation used for normal
            initialization. Defaults to None.
        scale (Optional[float], optional): scaling value. Defaults to None.

    Returns:
        Tensor: initialized tensor.
    """
    if rho is not None:
        return (
            W.div_(torch.linalg.eigvals(W).abs().max())  # pylint: disable=not-callable
            .mul_(rho)
            .float()
        )
    if sigma is not None:
        return (
            W.div_(torch.linalg.matrix_norm(W, ord=2))  # pylint: disable=not-callable
            .mul_(sigma)
            .float()
        )
    if scale is not None:
        return W.mul_(scale).float()
    return W
