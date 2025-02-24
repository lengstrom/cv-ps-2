from jaxtyping import Float
from torch import Tensor

from .provided import gaussian_filter


def zeroth_order(img: Float[Tensor, "*H W"], sigma: float) -> Float[Tensor, "*H W"]:
    return gaussian_filter(img, sigma, [0, 0])

def first_order_x(img: Float[Tensor, "*H W"], sigma: float) -> Float[Tensor, "*H W"]:
    return gaussian_filter(img, sigma, [0, 1])

def first_order_y(img: Float[Tensor, "*H W"], sigma: float) -> Float[Tensor, "*H W"]:
    return gaussian_filter(img, sigma, [1, 0])

def first_order_xy(img: Float[Tensor, "*H W"], sigma: float) -> Float[Tensor, "*H W"]:
    return gaussian_filter(img, sigma, [1, 1])

def second_order_xx(img: Float[Tensor, "*H W"], sigma: float) -> Float[Tensor, "*H W"]:
    return gaussian_filter(img, sigma, [0, 2])

def second_order_yy(img: Float[Tensor, "*H W"], sigma: float) -> Float[Tensor, "*H W"]:
    return gaussian_filter(img, sigma, [2, 0])

def log(img: Float[Tensor, "*H W"], sigma: float) -> Float[Tensor, "*H W"]:
    return second_order_xx(img, sigma) + second_order_yy(img, sigma)
