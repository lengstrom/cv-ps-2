import torch
import torch.nn.functional as F
from jaxtyping import Float, Complex
from torch import Tensor
ch = torch

from .gauss import _gaussian_filter_1d


def oriented_filter(theta: float, sigma: float, **kwargs) -> Float[Tensor, "N N"]:
    """
    Return an oriented first-order Gaussian filter
    given an angle (in radians) and standard deviation.

    Hint:
    - Use `.gauss._gaussian_filter_1d`!

    Implementation details:
    - **kwargs are passed to `_gaussian_filter_1d`
    """
    # raise NotImplementedError("Homework!")
    # return ch.cos(theta) * 
    from math import cos, sin
    # across x
    G1_0 = _gaussian_filter_1d(sigma, 1, **kwargs)[None, :] * _gaussian_filter_1d(sigma, 0, **kwargs)[:, None]
    G1_90 = _gaussian_filter_1d(sigma, 1, **kwargs)[:, None] * _gaussian_filter_1d(sigma, 0, **kwargs)[None, :]

    costheta = cos(theta)
    sintheta = sin(theta)
    # G1_1 = _gaussian_filter_1d(sigma, 1, **kwargs) * costheta
    # G1_0 = G1_0.unsqueeze(1)
    # G1_1 = G1_1.unsqueeze(0)
    return costheta * G1_0 + sintheta * G1_90

def conv(
    img: Float[Tensor, "B 1 H W"],  # Input image
    kernel: Float[Tensor, "N N"] | Complex[Tensor, "N N"],  # Convolutional kernel
    mode: str = "reflect",  # Padding mode
) -> Float[Tensor, "B 1 H W"]:
    """
    Convolve an image with a 2D kernel (assume N < H and N < W).
    """

    padded_img = F.pad(img, [kernel.shape[-1] // 2] * 4, mode=mode)
    ret = F.conv2d(padded_img, kernel[None, None, ...], padding=0)
    assert ret.shape == img.shape, (ret.shape, img.shape)
    return ret


def steer_the_filter(
    img: Float[Tensor, "B 1 H W"], theta: float, sigma: float, **kwargs
) -> Float[Tensor, "B 1 H W"]:
    """
    Return the image convolved with a steered filter.
    """
    return conv(img, oriented_filter(theta, sigma, **kwargs))


import numpy as np
def steer_the_images(
    img: Float[Tensor, "B 1 H W"], theta: float, sigma: float, **kwargs
) -> Float[Tensor, "B 1 H W"]:
    """
    Return the steered image convolved with a filter.
    """
    img_one = conv(img, oriented_filter(0, sigma, **kwargs))
    img_two = conv(img, oriented_filter(np.pi/2, sigma, **kwargs))
    return np.cos(theta) * img_one + np.sin(theta) * img_two

def complex_filter(sigma, **kwargs):
    from math import cos, sin
    # across x
    G1_0 = _gaussian_filter_1d(sigma, 1, **kwargs)[None, :] * _gaussian_filter_1d(sigma, 0, **kwargs)[:, None]
    G1_90 = _gaussian_filter_1d(sigma, 1, **kwargs)[:, None] * _gaussian_filter_1d(sigma, 0, **kwargs)[None, :]

    # G1_1 = _gaussian_filter_1d(sigma, 1, **kwargs) * costheta
    # G1_0 = G1_0.unsqueeze(1)
    # G1_1 = G1_1.unsqueeze(0)
    return (G1_0 + 0j) + 1j * G1_90

def measure_orientation(
    img: Float[Tensor, "B 1 H W"], sigma: float, **kwargs
) -> Float[Tensor, "B 1 H W"]:
    """
    Design a filter to measure the orientation of edges in an image.

    Hint:
    - Consider the complex filter from the README
    - You will need to design a method for noise suppression
    """
    import numpy as np

    complex_filt = complex_filter(sigma, **kwargs)
    convolved = conv(img + 0j, complex_filt)
    angles = ch.angle(convolved) * ch.abs(convolved) / ch.max(ch.abs(convolved))
    return angles
    
