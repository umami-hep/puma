from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from puma.matshow import MatshowPlot

# A random matrix
mat = np.random.rand(4, 3)

# Declaring the plot class
matrix_plot = MatshowPlot(mat)
# Saving the plot
matrix_plot.savefig("vanilla_mat.png")

# Some possible customizations
# Matrix's column names
x_ticks = ["a", "b", "c"]
# Matrix's rows names
y_ticks = ["d", "e", "f", "g"]

# Declaring the plot class with custom style
matrix_plot_custom = MatshowPlot(
    mat,
    x_ticklabels=x_ticks,
    x_ticks_rotation=45,
    y_ticklabels=y_ticks,
    show_entries=True,
    show_percentage=True,
    text_color_threshold=0.6,
    colormap=plt.cm.PiYG,
    cbar_label="Scalar values as percentages",
    atlas_offset=0.65,
)

# Saving the plot
matrix_plot_custom.savefig("mat_custumized.png")
