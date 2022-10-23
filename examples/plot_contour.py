"""Example for a puma.contour plot"""

import numpy as np

from puma import ContourPlot
from puma.utils import global_config

rng = np.random.default_rng(seed=42)

N_RANDOM = 50_000

x_values_1 = rng.normal(size=N_RANDOM)
y_values_1 = rng.normal(size=N_RANDOM)

x_values_2 = rng.normal(loc=3, scale=3, size=N_RANDOM)
y_values_2 = rng.normal(loc=1.5, scale=2, size=N_RANDOM)

contour_plot = ContourPlot(
    xlabel="$x$ values",
    ylabel="$y$ values",
    bins=30,
    bins2d=15,
    x_range=(-5, 10),
    y_range=(-7.5, 10),
)

contour_plot.add(
    x_values_1,
    y_values_1,
    colour=global_config["flavour_categories"]["bjets"]["colour"],
    label=global_config["flavour_categories"]["bjets"]["legend_label"],
)
contour_plot.add(
    x_values_2,
    y_values_2,
    colour=global_config["flavour_categories"]["cjets"]["colour"],
    label=global_config["flavour_categories"]["cjets"]["legend_label"],
)
contour_plot.draw()
contour_plot.savefig("contour_plot.png")
