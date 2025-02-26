# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:50:13 2024

@author: jvilp
"""

from tqdm import tqdm
import numpy.typing as npt
import numpy as np
from math import pi

def fBM_simul(T: float, N: int, H: float) -> npt.NDArray[np.float32]:
    """Spectral simulation of fBM (Appendix B)

    Args:
        T (float): Time hoizon
        N (int): nb of time step
        H (float): Hurst exponent in [0, 1]

    Returns:
        npt.NDArray[np.float32]: _description_
    """

    if (
        N % 2 != 0
    ):  # if N is not pair we adjust it to ensure 0.5*N is include in integer and that delta stay the same
        T = (T / N) * (N + 1)
        N = N + 1
        shift = 1
    else:
        shift = 0

    dt = T / N
    phi = np.random.uniform(low=0, high=2 * pi, size=N)
    W_increment = [
        compute_Wk(k, N, H, phi)
        for k in tqdm(
            range(0, N), desc="Computing fBm increments...", leave=False, total=N
        )
    ]

    W = dt**H * np.cumsum(W_increment)
    W = np.array(W[: len(W) - shift])
    W[0] = 0
    return W


def compute_Wk(k: int, N: int, H: float, phi:list) -> float:
    """Compute fBM increment (B.2)

    Args:
        k (int): iteration over N
        N (int): nb of time step
        H (float): Hurst exponent range(0,1)

    Returns:
        float: The increment
    """
    return sum(
        map(
            lambda j: np.sqrt(2 / N)
            * (compute_Sf(j / N, N, H) ** 0.5)
            * (
                np.cos(2 * pi * j * k / N) * np.cos(phi[int(j + 0.5 * N)])
                - np.sin(2 * pi * j * k / N) * np.sin(phi[int(j + 0.5 * N)])
            ),
            range(int(-0.5 * N), int(0.5 * N)),
        )
    )  # Wk


def compute_Sf(f: float, N: int, H: float) -> float:
    """Power sprectral density aproximation (B.3)

    Args:
        f (float): frequency
        N (int): nb of time step
        H (float): Hurst exponent range(0,1)

    Returns:
        float: _description_
    """

    return 0.5 * sum(
        map(
            lambda m: (
                abs(m + 1) ** (2 * H) + abs(m - 1) ** (2 * H) - 2 * abs(m) ** (2 * H)
            )
            * np.cos(2 * pi * m * f),
            range(int(-0.5 * N), int(0.5 * N)),
        )
    )  # Sf



### Rendering

import matplotlib.pyplot as plt

T = 1 # time period
N = 100 # nb of point
H = 0.1 # hurst exponent
fBM = fBM_simul(T, N, H)

plt.plot(np.linspace(0,T,N), fBM, label = "fBM")
plt.title(f"Fractionnal Brownian Motion with H={H}")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.show()





