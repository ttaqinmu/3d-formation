from typing import Callable, Tuple

import numba as nb
import numpy as np


def jitter(func: Callable, **kwargs):
    """Jits a function."""
    return nb.njit(func, **kwargs)


def minmax_scale(arr: np.ndarray, min_val: float, max_val: float, feature_range=Tuple[float, float]) -> np.ndarray:
    """
    Scale np.array ke range tertentu (default [0,1]) 
    berdasarkan min_val dan max_val global.

    arr : np.array
    min_val : float
    max_val : float
    feature_range : tuple (a, b) → target range
    
    return : np.array scaled
    """
    a, b = feature_range
    scaled = (arr - min_val) / (max_val - min_val)  # ke [0,1]
    scaled = scaled * (b - a) + a                  # ke [a,b]
    return scaled

def array_zfill(arr: np.ndarray, length: int, fill_value: float = 0.0):
    arr = np.array(arr)
    if len(arr) >= length:
        return arr[:length]  # kalau lebih panjang, trim
    pad_len = length - len(arr)
    return np.pad(arr, (0, pad_len), constant_values=fill_value)
