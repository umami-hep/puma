from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from puma.matshow import MatshowPlot

# seeded PRNG for reproducibility
prng = np.random.default_rng(seed=0)

# A random matrix
mat = prng.random(size=(4, 3))

# Declaring the plot class
matrix_plotter = MatshowPlot()
matrix_plotter.draw(mat)
# Saving the plot
matrix_plotter.savefig("vanilla_mat.png")

# Some possible customizations
# Matrix's column names
x_ticks = ["a", "b", "c"]
# Matrix's rows names
y_ticks = ["d", "e", "f", "g"]

# Declaring the plot class with custom style
matrix_plotter_custom = MatshowPlot(
    x_ticklabels=x_ticks,
    x_ticks_rotation=45,
    y_ticklabels=y_ticks,
    show_entries=True,
    show_percentage=True,
    text_color_threshold=0.6,
    colormap=plt.cm.PiYG,
    cbar_label="Scalar values as percentages",
    atlas_tag_outside=True,
    fontsize=15,
)
matrix_plotter_custom.draw(mat)
# Saving the plot
matrix_plotter_custom.savefig("mat_custumized.png")
