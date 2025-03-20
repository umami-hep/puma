from __future__ import annotations

import numpy as np


def get_fx_values(resolution=100):
    return np.concatenate((
        np.logspace(-3, -1, resolution // 2),
        np.linspace(0.1, 1.0, resolution // 2),
    ))


def get_efficiency(scores, fx):
    return np.sum(scores > fx) / len(scores)


def get_optimal_fraction_value(
    fraction_scan: np.ndarray,
    fraction_space: np.ndarray,
    rej: bool = False,
):
    """After calculating a fraction scan, find the optimal fraction value.


    Parameters
    ----------
    fraction_scan : np.ndarray
        2D array of efficiency (or rejection) scores for each fraction value.
    fraction_space : np.ndarray
        1D array of fraction values.
    rej : bool, optional
        If True, find the maximum rejection values else find the minimum efficiency values, by
        default False.
    """
    # normalise x- and y-axes
    xs, ys = fraction_scan[:, 0], fraction_scan[:, 1]
    xs = xs / max(xs)
    ys = ys / max(ys)

    # if rej=True get maximum distance to origin
    opt_idx = np.argmax(xs**2 + ys**2) if rej else np.argmin(xs**2 + ys**2)

    return opt_idx, fraction_space[opt_idx]
