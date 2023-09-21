"""Example script that demonstrates Data/MC plots."""
from __future__ import annotations

import numpy as np

from puma import Histogram, HistogramPlot

rng = np.random.default_rng(42)
mc1 = rng.normal(size=10_000)
mc2 = rng.normal(size=20_000)
data = rng.normal(size=30_000)

data_mc_plot = HistogramPlot(
    bins_range=[-2, 2],
    n_ratio_panels=1,
    stacked=True,
    norm=False,
)
data_mc_plot.title = "Test Data/MC Plot"
data_mc_plot.add(Histogram(mc1, label="MC Process 1", colour="b"))
data_mc_plot.add(Histogram(mc2, label="MC Process 2", colour="r"))
data_mc_plot.add(Histogram(data, label="MC Process 2", is_data=True, colour="k"))
data_mc_plot.draw()
data_mc_plot.savefig("data_mc_example.png")
