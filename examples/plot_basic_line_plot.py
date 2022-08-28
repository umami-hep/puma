"""Example for a basic line plot with puma"""
import numpy as np
from puma import Line2DPlot, Line2D

# This is just some dummy data to make the plot look reasonable
epochs = np.arange(0, 100)
training_loss = np.exp(-epochs) + np.random.normal(0, 0.01, size=len(epochs)) + 0.8
validation_loss = np.exp(-epochs) + np.random.normal(0, 0.03, size=len(epochs)) + 0.85

# Initialise the plot
line_plot = Line2DPlot(
    xlabel="Epoch",
    ylabel="Loss",
    atlas_second_tag="This is an example of a basic line plot",
)

# Add and draw the lines
line_plot.add(Line2D(epochs, training_loss, label="Training loss"))
line_plot.add(Line2D(epochs, validation_loss, label="Validation loss"))
line_plot.draw()
line_plot.savefig("line_plot_example.png", transparent=False)
