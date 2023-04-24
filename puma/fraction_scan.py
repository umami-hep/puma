import numpy as np


def get_fx_values(num=100):
    return np.concatenate(
        (np.logspace(-3, -1, num // 2), np.linspace(0.1, 1.0, num // 2))
    )


def get_efficiency(scores, fx):
    return np.sum(scores > fx) / len(scores)
