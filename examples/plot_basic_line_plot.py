"""Example for a basic line plot with puma."""

from __future__ import annotations

import numpy as np

from puma import Line2D, Line2DPlot

# This is just some dummy data to make the plot look reasonable
epochs = np.arange(0, 100)
training_loss = np.exp(-epochs) + np.random.normal(0, 0.01, size=len(epochs)) + 0.8
validation_loss = np.exp(-epochs) + np.random.normal(0, 0.03, size=len(epochs)) + 0.85

# Initialise the plot
line_plot = Line2DPlot(
    xlabel="Epoch",
    ylabel="Loss",
    atlas_second_tag="This is an example of a basic line plot",
    bin_array_path="plot_line2d_example.pkl",
)

# Add and draw the lines
line_plot.add(Line2D(epochs, training_loss, label="Training loss"), key="train")
line_plot.add(Line2D(epochs, validation_loss, label="Validation loss"), key="val")

line_plot.draw()
line_plot.savefig("line_plot_example.png", transparent=False)

# Plot from pickle file
line_plot_from_file = Line2DPlot(
    xlabel="Epoch",
    ylabel="Loss",
    atlas_second_tag="This is an example of a basic line plot from the pickle file",
    bin_array_path="plot_line2d_example.pkl",
)

# Add and draw the lines
line_plot_from_file.add(key="train")
line_plot_from_file.add(key="val")

line_plot_from_file.draw()
line_plot_from_file.savefig("line_plot_example_from_file.png", transparent=False)
