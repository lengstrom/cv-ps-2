from typing import Tuple, TypeVar

import torch
ch = torch
from jaxtyping import Float
from torch import Tensor

N = TypeVar("N")


def shift_operator(img_shape: Tuple[int, int], shift_x: int, shift_y: int) -> Tensor:
    """
    Constructs a 2D shift operator for an image with circular boundaries.

    Args:
        img_shape: Tuple[int, int]
            The (height, width) dimensions of the image.
        shift_x: int
            The number of pixels to shift horizontally.
        shift_y: int
            The number of pixels to shift vertically.

    Returns:
        Tensor of shape (h*w, h*w)
            A matrix that, when applied to a flattened image, shifts it by the specified amounts.
    """
    # A = (Py @ A) @ Px
    Py = ch.eye(img_shape[1])
    Py = ch.roll(Py, (-shift_y,), dims=0)

    Px = ch.eye(img_shape[0])
    Px = (ch.roll(Px, (shift_x,), dims=1))

    # vec(Py X Px) = (Px.T kron Py) vec(X)
    # print(Px.shape, Py.shape)
    return ch.kron(Py, Px)

def matrix_from_convolution_kernel(
    kernel: Float[Tensor, "*"], n: int
) -> Float[Tensor, "n n"]:
    """
    Constructs a circulant matrix of size n x n from a 1D convolution kernel with periodic alignment.

    Args:
        kernel: Tensor
            A 1D convolution kernel.
        n: int
            The desired size of the circulant matrix.

    Returns:
        Tensor of shape (n, n)
            The circulant matrix representing the convolution with periodic boundary conditions.
    """
    row = ch.zeros(n)
    row[:len(kernel)] = kernel
    row = row.roll(-(len(kernel)//2))
    rows = [row.roll(i) for i in range(n)]
    return ch.stack(rows, dim=0)

def image_operator_from_sep_kernels(
    img_shape: Tuple[int, int],
    kernel_x: Float[Tensor, "*"],
    kernel_y: Float[Tensor, "*"],
) -> Float[Tensor, "N N"]:
    """
    Constructs a 2D convolution operator for an image by combining separable 1D kernels.

    Args:
        img_shape: Tuple[int, int]
            The (height, width) dimensions of the image.
        kernel_x: Tensor
            The 1D convolution kernel to be applied horizontally.
        kernel_y: Tensor
            The 1D convolution kernel to be applied vertically.

    Returns:
        Tensor of shape (h*w, h*w)
            The 2D convolution operator acting on a flattened image.
    """
    x_op = matrix_from_convolution_kernel(kernel_x, img_shape[1])
    y_op = matrix_from_convolution_kernel(kernel_y, img_shape[0])
    return ch.kron(y_op, x_op)


def eigendecomposition(
    operator: Float[Tensor, "N N"], descending: bool = True
) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N N"]]:
    """
    Computes the eigenvalues and eigenvectors of a self-adjoint (Hermitian) linear operator.

    Args:
        operator: Tensor of shape (N, N)
            A self-adjoint linear operator.
        descending: bool
            If True, sort the eigenvalues and eigenvectors in descending order.

    Returns:
        A tuple (eigenvalues, eigenvectors) where:
            eigenvalues: Tensor of shape (N,)
            eigenvectors: Tensor of shape (N, N)
    """
    values, vecs = ch.linalg.eigh(operator)
    if descending:
        # reverse the order of the eigenvalues and eigenvectors
        values = values.flip(0)
        vecs = vecs.flip(1)

    return values, vecs

def fourier_transform_operator(
    operator: Float[Tensor, "N N"], basis: Float[Tensor, "N N"]
) -> Float[Tensor, "N N"]:
    """
    Computes the representation of a linear operator in the Fourier (eigen) basis.

    Args:
        operator: Tensor of shape (N, N)
            The original linear operator in pixel space.
        basis: Tensor of shape (N, N)
            The Fourier eigenbasis.

    Returns:
        Tensor of shape (N, N)
            The operator represented in the Fourier basis.
    """
    m = basis.T @ operator @ basis
    return m

def fourier_transform(
    img: Float[Tensor, "N"], basis: Float[Tensor, "N N"]
) -> Float[Tensor, "N"]:
    """
    Projects a flattened image onto the Fourier (eigen) basis.

    Args:
        img: Tensor of shape (N,)
            A flattened image.
        basis: Tensor of shape (N, N)
            The Fourier eigenbasis.

    Returns:
        Tensor of shape (N,)
            The image represented in the Fourier domain.
    """
    return basis.T @ img

def inv_fourier_transform(
    fourier_img: Float[Tensor, "N"], basis: Float[Tensor, "N N"]
) -> Float[Tensor, "N"]:
    """
    Reconstructs an image in pixel space from its Fourier coefficients using the provided eigenbasis.

    Args:
        fourier_img: Tensor of shape (N,)
            The image in the Fourier domain.
        basis: Tensor of shape (N, N)
            The Fourier eigenbasis used in the forward transform.

    Returns:
        Tensor of shape (N,)
            The reconstructed image in pixel space.
    """
    return basis @ fourier_img
