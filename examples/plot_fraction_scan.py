"""Example of fraction scan plot"""

import numpy as np

from puma import Line2D, Line2DPlot
from puma.metrics import calc_eff
from puma.utils import get_dummy_2_taggers, logger

# The line below generates dummy data which is similar to a NN output
df = get_dummy_2_taggers(size=100_000)

logger.info("caclulate tagger discriminants")

# defining boolean arrays to select the different flavour classes
is_light = df["HadronConeExclTruthLabelID"] == 0
is_c = df["HadronConeExclTruthLabelID"] == 4
is_b = df["HadronConeExclTruthLabelID"] == 5


fc_values = np.linspace(0.0, 1.0, 101)
SIG_EFF = 0.77

dips_scores = df[["dips_pu", "dips_pc", "dips_pb"]].values


def calc_effs(fc_value: float):
    """Tagger efficiency for fixed working point

    Parameters
    ----------
    fc_value : float
        Value for the charm fraction used in discriminant calculation.

    Returns
    -------
    tuple
        Tuple of shape (, 3) containing (fc_value, ujets_eff, cjets_eff)
    """
    arr = dips_scores
    disc = arr[:, 2] / (fc_value * arr[:, 1] + (1 - fc_value) * arr[:, 0])
    ujets_eff = calc_eff(disc[is_b], disc[is_light], SIG_EFF)
    cjets_eff = calc_eff(disc[is_b], disc[is_c], SIG_EFF)

    return [fc_value, ujets_eff, cjets_eff]


eff_results = np.array(list(map(calc_effs, fc_values)))


# Init example x- and y values of the fraction scan. Theses
# are the x- and y values for the rejections for different fractions.
# You can plot 2 rejections against each other. These values can either
# be arrays, lists, ints or floats but the dtype must be the same and also
# the number of values inside!
x_values = eff_results[:, 2]
y_values = eff_results[:, 1]

# If you want to mark a specific point with a marker, you just need to
# define the x- and y values again for the marker (This must be floats)
MARKER_X = eff_results[30, 2]
MARKER_Y = eff_results[30, 1]

# You can give several plotting options to the plot itself

# Now init a fraction scan plot
frac_plot = Line2DPlot()

# Add our x- and y values as a new line
# The colour and linestyle are optional here
frac_plot.add(
    Line2D(
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
    Line2D(
        x_values=MARKER_X,
        y_values=MARKER_Y,
        colour="r",
        marker="x",
        label=rf"$f_c={eff_results[30, 0]}$",
        markersize=15,
        markeredgewidth=2,
    ),
    is_marker=True,
)

# Adding labels
frac_plot.ylabel = "Light-flavour jets efficiency"
frac_plot.xlabel = "$c$-jets efficiency"

# Draw and save the plot
frac_plot.draw()
frac_plot.savefig("FractionScanPlot_test.png")
