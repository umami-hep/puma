"""Example of fraction scan plot"""

import numpy as np

from puma import FractionScan, FractionScanPlot

# Init example x- and y values of the fraction scan. Theses
# are the x- and y values for the rejections for different fractions.
# You can plot 2 rejections against each other. These values can either
# be arrays, lists, ints or floats but the dtype must be the same and also
# the number of values inside!
x_values = np.linspace(30, 50, 1000)
y_values = np.linspace(30, 50, 1000)

# If you want to mark a specific point with a marker, you just need to
# define the x- and y values again for the marker (This must be floats)
MARKER_X = 40
MARKER_Y = 40

# You can give several plotting options to the plot itself
kwargs = {"xlabel": "Rejection 1", "ylabel": "Rejection 2"}

# Now init a fraction scan plot
frac_plot = FractionScanPlot(**kwargs)

# Add our x- and y values as a new line
# The colour and linestyle are optional here
frac_plot.add(
    FractionScan(
        x_values=x_values,
        y_values=y_values,
        label="Tagger 1",
        colour="r",
        linestyle="-",
    )
)

# Add a marker for the just added fraction scan. If you don't
# set the colour here, the colour of the last added element will be used.
# marker, markersize and markeredgewidth are optional. The here
# given values are the default values.
# The is_marker bool tells the plot that this is a marker and not a line
frac_plot.add(
    FractionScan(
        x_values=MARKER_X,
        y_values=MARKER_Y,
        colour="r",
        marker="x",
        label="Marker label",
        markersize=15,
        markeredgewidth=2,
    ),
    is_marker=True,
)

# Draw and save the plot
frac_plot.draw()
frac_plot.savefig("FractionScanPlot_test.png", transparent=False)
