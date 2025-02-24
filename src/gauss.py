from math import factorial, ceil, sqrt

import torch
from jaxtyping import Float
from torch import Tensor
ch = torch
import numpy as np

def _gaussian_filter_1d(
    sigma: float,  # Standard deviation of the Gaussian
    order: int,  # Order of the derivative
    truncate: float = 4.0,  # Truncate the filter at this many standard deviations
    dtype: torch.dtype = torch.float32,  # Data type to run the computation in
    device: torch.device = torch.device("cpu"),  # Device to run the computation on
) -> Float[Tensor, " filter_size"]:
    """
    Return a 1D Gaussian filter of a specified order.

    Implementation details:
    - filter_size = 2r + 1, where r = ceil(truncate * sigma)
    """
    r = ceil(truncate * sigma)
    filter_size = 2 * r + 1
    half_filter_size = filter_size // 2
    # sample_points = ch.linspace(-half_filter_size, half_filter_size, filter_size)
    sample_points = ch.arange(-half_filter_size, half_filter_size + 1, dtype=dtype, device=device)

    exp_part = ch.exp(sample_points**2 / (-2 * sigma**2))
    order0 = exp_part / ((2 * np.pi)**0.5 * sigma)

    order1 = (-sample_points/sigma**2) * order0
    order2 = (sample_points**2/sigma**4 - 1/sigma**2) * order0

    if order == 0:
        ys = order0
    elif order == 1:
        ys = order1
    elif order == 2:
        ys = order2

    assert order <= 2

    # zero center the filter
    if order > 0:
        ys = ys - ch.mean(ys)

        sum_polynomial = ch.sum(ys * sample_points**order)
        desired = factorial(order) / (-1)**order
        ys = (ys / sum_polynomial)  * desired

    return ys.to(dtype).to(device)
