"""Produce histogram of discriminant from tagger output and labels."""

from __future__ import annotations

import numpy as np
from ftag import Flavours
from ftag.utils import get_discriminant

from puma import Histogram, HistogramPlot
from puma.utils import get_dummy_2_taggers, get_good_linestyles

# The line below generates dummy data which is similar to a NN output
df = get_dummy_2_taggers()

# Calculate discriminant scores for DIPS and RNNIP, and add them to the dataframe
disc_dips = get_discriminant(
    jets=df,
    tagger="dips",
    signal=Flavours["bjets"],
    flavours=Flavours.by_category("single-btag"),
    fraction_values={
        "fc": 0.018,
        "fu": 0.982,
        "ftau": 0,
    },
)
disc_rnnip = get_discriminant(
    jets=df,
    tagger="rnnip",
    signal=Flavours["bjets"],
    flavours=Flavours.by_category("single-btag"),
    fraction_values={
        "fc": 0.018,
        "fu": 0.982,
        "ftau": 0,
    },
)

# defining boolean arrays to select the different flavour classes
is_light = df["HadronConeExclTruthLabelID"] == 0
is_c = df["HadronConeExclTruthLabelID"] == 4
is_b = df["HadronConeExclTruthLabelID"] == 5

taggers = ["dips", "rnnip"]
discs = {"dips": disc_dips, "rnnip": disc_rnnip}
linestyles = get_good_linestyles()[:2]

# Initialise histogram plot
plot_histo = HistogramPlot(
    n_ratio_panels=1,
    ylabel="Normalised number of jets",
    ylabel_ratio=["Ratio to DIPS"],
    xlabel="$b$-jet discriminant",
    logy=False,
    leg_ncol=1,
    figsize=(5.5, 4.5),
    y_scale=1.5,
    ymax_ratio=[1.5],
    ymin_ratio=[0.5],
    atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets \ndummy sample, $f_{c}=0.018$",
)

# Add the histograms
for tagger, linestyle in zip(taggers, linestyles):
    plot_histo.add(
        Histogram(
            values=discs[tagger][is_light],
            # Only specify the label for the case of the "DIPS" light-jets, since we
            # want to hide the legend entry for "RNNIP" light-jets as it has the same
            # linecolour. Instead, we specify a "linestyle legend" further down in the
            # script
            bins=np.linspace(-10, 10, 50),
            label="Light-flavour jets" if tagger == "dips" else None,
            colour=Flavours["ujets"].colour,
            ratio_group="ujets",
            linestyle=linestyle,
        ),
        reference=tagger == "dips",
    )
    plot_histo.add(
        Histogram(
            values=discs[tagger][is_c],
            bins=np.linspace(-10, 10, 50),
            label="$c$-jets" if tagger == "dips" else None,
            colour=Flavours["cjets"].colour,
            ratio_group="cjets",
            linestyle=linestyle,
        ),
        reference=tagger == "dips",
    )
    plot_histo.add(
        Histogram(
            values=discs[tagger][is_b],
            bins=np.linspace(-10, 10, 50),
            label="$b$-jets" if tagger == "dips" else None,
            colour=Flavours["bjets"].colour,
            ratio_group="bjets",
            linestyle=linestyle,
        ),
        reference=tagger == "dips",
    )

plot_histo.draw()
# The lines below create a legend for the linestyles (i.e. solid lines -> DIPS, dashed
# lines -> RNNIP here). The "bbox_to_anchor" argument specifies where to place the
# linestyle legend
plot_histo.make_linestyle_legend(
    linestyles=linestyles, labels=["DIPS", "RNNIP"], bbox_to_anchor=(0.55, 1)
)
plot_histo.savefig("histogram_discriminant.png", transparent=False)
