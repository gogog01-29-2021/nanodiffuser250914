"""Schedules module for nano diffusion.

This module implements noise schedules and EDM preconditioning functions.
Refer to Karras et al. (2022) for sigma schedules and EDM preconditioning.
"""

from typing import List


def edm_sigma_schedule(num_steps: int, sigma_min: float = 0.002, sigma_max: float = 80.0) -> List[float]:
    """Return a geometric progression of sigmas for the EDM noise schedule.

    Args:
        num_steps: Number of noise levels.
        sigma_min: Minimum sigma value.
        sigma_max: Maximum sigma value.

    Returns:
        A list of sigmas descending from sigma_max to sigma_min.

    Example:
        >>> edm_sigma_schedule(4)
        [80.0, 2.350, 0.0685, 0.002]
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    import numpy as np
    ratios = np.linspace(0, 1, num_steps)
    sigmas = sigma_max * ((sigma_min / sigma_max) ** ratios)
    return sigmas.tolist()
