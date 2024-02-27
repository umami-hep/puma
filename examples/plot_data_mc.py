"""Example script that demonstrates Data/MC plots."""

from __future__ import annotations

import numpy as np

from puma import Histogram, HistogramPlot

# Generate two MC contributions and data
rng = np.random.default_rng(42)
mc1 = rng.normal(size=10_000)
mc2 = rng.normal(size=20_000)
data = rng.normal(size=30_000)

# Set up the real plot
data_mc_plot = HistogramPlot(
    bins_range=[-2, 2],
    n_ratio_panels=1,
    stacked=True,
    norm=False,
)

# Set the plot title
data_mc_plot.title = "Test Data/MC Plot"

# Add the different MC contributions to the plot
data_mc_plot.add(Histogram(mc1, label="MC Process 1"))
data_mc_plot.add(Histogram(mc2, label="MC Process 2"))

# Add the data
data_mc_plot.add(Histogram(data, label="Data", is_data=True, colour="k"))

# Draw the plot
data_mc_plot.draw()

# Add the bin width to the y-axis label
data_mc_plot.add_bin_width_to_ylabel()
data_mc_plot.savefig("data_mc_example.png")
