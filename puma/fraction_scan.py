from __future__ import annotations

import numpy as np


def get_fx_values(resolution=100):
    return np.concatenate((
        np.logspace(-3, -1, resolution // 2),
        np.linspace(0.1, 1.0, resolution // 2),
    ))


def get_efficiency(scores, fx):
    return np.sum(scores > fx) / len(scores)


def get_optimal_fc(fc_scan: np.ndarray, fc_space: np.ndarray, rej=False):
    """After calculating an fc scan, find the
    optimal value of fc.


    Parameters
    ----------
    fc_scan : np.ndarray
        2D array of efficiency (or rejection) scores
        for each value of fc.
    fc_space : np.ndarray
        1D array of fc values.
    rej : bool, optional
        If True, find the maximum rejection values else
        find the minimum efficiency values, by default False.
    """
    # normalise x- and y-axes
    xs, ys = fc_scan[:, 0], fc_scan[:, 1]
    xs = xs / max(xs)
    ys = ys / max(ys)

    # if rej=True get maximum distance to origin
    opt_idx = np.argmax(xs**2 + ys**2) if rej else np.argmin(xs**2 + ys**2)

    return opt_idx, fc_space[opt_idx]
