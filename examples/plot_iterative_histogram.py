"""Example script that demonstrates iterative Histogram filling."""

from __future__ import annotations

import numpy as np

from puma import Histogram, HistogramPlot

rng = np.random.default_rng(42)

# We create now two value sets to mimic a batch-wise loading
vals = rng.normal(size=10_000)
extra_vals = rng.normal(size=10_000)

# We can also define weights for each entry. For simplicity, we set them to one for this example
weights_vals = np.ones_like(vals)
weights_extra_vals = np.ones_like(extra_vals)

# Now loop over our batches
for counter, (batch_values, batch_weights) in enumerate(
    zip([vals, extra_vals], [weights_vals, weights_extra_vals])
):
    # Create the histogram if it's the first batch
    if counter == 0:
        histo = Histogram(
            values=batch_values,
            weights=batch_weights,
            bins=40,
            bins_range=(-2, 2),
            underoverflow=False,
            label="Gaussian($\\mu=0$, $\\sigma=1$)",
        )

    # Update the existing histogram else
    else:
        histo.update(values=batch_values, weights=batch_weights)

# Create the HistogramPlot and add the iterativly filled histogram
plot = HistogramPlot()
plot.add(histo)
plot.draw()
plot.savefig("Iterative_histogram.png")
