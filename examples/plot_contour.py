"""Example for a puma.contour plot"""

import numpy as np

from puma.contour import ContourPlot

rng = np.random.default_rng(seed=42)

N_RANDOM = 50_000

x_values = rng.normal(size=N_RANDOM)
y_values = rng.normal(size=N_RANDOM)

contour_plot = ContourPlot(
    xlabel="$x$ values",
    ylabel="$y$ values",
    bins=30,
    bins2d=15,
    x_range=(-5, 22),
    y_range=(-8, 8),
)

contour_plot.add(
    x_values,
    y_values,
    colour="b",
    label=None,
)
contour_plot.draw()
contour_plot.savefig("contour_plot.pdf")
