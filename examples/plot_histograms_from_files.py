"""Example script that demonstrates saving/loading Histograms."""

from __future__ import annotations

import numpy as np

from puma import Histogram, HistogramPlot

rng = np.random.default_rng(42)

vals = rng.normal(size=10_000)

# Create the histogram and store it
histo = Histogram(
    values=vals,
    bins=40,
    bins_range=(-2, 2),
    underoverflow=False,
    label="Gaussian($\\mu=0$, $\\sigma=1$)",
)
histo.save("test_histo.yaml")

# Create the HistogramPlot and add the still-loaded Histogram object
plot = HistogramPlot()
plot.add(histo)
plot.draw()
plot.savefig("fresh_histogram.png")

# Load the Histogram and create a new plot with it
loaded_histo = Histogram.load("test_histo.yaml")
loaded_histo_plot = HistogramPlot()
loaded_histo_plot.add(loaded_histo)
loaded_histo_plot.draw()
loaded_histo_plot.savefig("loaded_histogram.png")
