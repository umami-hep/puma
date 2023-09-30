"""Example plotting script for the puma.PiePlot class."""
from __future__ import annotations

from ftag import Flavours

from puma.pie import PiePlot

HadrTruthLabel_fracs = [200_000, 34_000, 150_000, 5_000]
HadrTruthLabel_labels = ["Light-flavour jets", "$c$-jets", "$b$-jets", "$\\tau$-jets"]

# Basic example with default values only
example_plot_1 = PiePlot(
    wedge_sizes=HadrTruthLabel_fracs,
    labels=HadrTruthLabel_labels,
    figsize=(5.5, 3.5),
    draw_legend=False,
)
example_plot_1.savefig("pie_example_1.png")


# Another example with some styling

# Get the flavour colours from the global config
colours = [
    Flavours["ujets"].colour,
    Flavours["cjets"].colour,
    Flavours["bjets"].colour,
    Flavours["taujets"].colour,
]

example_plot_2 = PiePlot(
    wedge_sizes=HadrTruthLabel_fracs,
    labels=HadrTruthLabel_labels,
    draw_legend=False,
    colours=colours,
    # have a look at the possible kwargs for matplotlib.pyplot.pie here:
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html
    mpl_pie_kwargs={
        "explode": [0, 0.2, 0, 0.1],
        "shadow": False,
        "startangle": 90,
        "textprops": {"fontsize": 10},
        "radius": 1,
        "wedgeprops": {"width": 0.4, "edgecolor": "w"},
        "pctdistance": 0.4,
    },
    # kwargs passed to puma.PlotObject
    atlas_second_tag="Dummy flavour fractions",
    figsize=(5.5, 3.5),
    y_scale=1.3,
)
example_plot_2.savefig("pie_example_2.png")
