"""Example of histogram plot that deviates from puma default plots"""

import numpy as np

from puma import histogram, histogram_plot

# Generate two distributions to plot
n_bkg = int(1e6)
n_sig = int(2e4)
rng = np.random.default_rng(seed=42)
expectation = rng.exponential(size=n_bkg)
measurement = np.concatenate(
    (rng.exponential(size=n_bkg), rng.normal(loc=2, scale=0.2, size=n_sig))
)
expectation_hist = histogram(expectation, label="MC", histtype="stepfilled", alpha=1)
measurement_hist = histogram(measurement, label="dummy data")

# Initialise histogram plot
plot_histo = histogram_plot(
    ylabel="Number of events",
    xlabel="Invariant mass $m$ [a.u.]",
    logy=False,
    # bins=np.linspace(0, 5, 60),  # you can force a binning for the plot here
    bins=50,  # you can also define an integer number for the number of bins
    bins_range=(1.1, 4),  # only considered if bins is an integer
    norm=False,
    atlas_first_tag="Simulation Internal",
    atlas_second_tag="Example plot for plotting python API",
    figsize=(6, 5),
)

# Add histograms and plot
plot_histo.add(expectation_hist, reference=True)
plot_histo.add(measurement_hist)
plot_histo.draw()

plot_histo.savefig("histogram_basic_example.png", transparent=False)
