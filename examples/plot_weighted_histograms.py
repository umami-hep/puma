"""Example script for plotting weighted histograms"""
import numpy as np

from puma import Histogram, HistogramPlot

rng = np.random.default_rng(seed=42)
# we define two gaussian distributions - one located at 0, one at 3
values = np.hstack((rng.normal(size=10_000), rng.normal(loc=3, size=10_000)))
# for the weighted histogram we weight entries of the right peak by a factor of 2
weights = np.hstack((np.ones(10_000), 2 * np.ones(10_000)))

hist_plot = HistogramPlot(n_ratio_panels=1, norm=False)
# add the unweighted histogram
hist_plot.add(
    Histogram(values, label="Without weights"),
    reference=True,
)
# add the weighted histogram
hist_plot.add(
    Histogram(
        values,
        weights=weights,
        label="Weight 2 for right peak",
    )
)
hist_plot.draw()
hist_plot.savefig("histogram_weighted.png")
