"""Produce histogram of discriminant from tagger output and labels."""

import numpy as np

from puma import Histogram, HistogramPlot
from puma.utils import get_dummy_2_taggers, global_config

# The line below generates dummy data which is similar to a NN output
df = get_dummy_2_taggers()

# Calculate discriminant scores for DIPS and RNNIP, and add them to the dataframe
FRAC_C = 0.018
df["disc_dips"] = np.log(
    df["dips_pb"] / (FRAC_C * df["dips_pc"] + (1 - FRAC_C) * df["dips_pu"])
)
df["disc_rnnip"] = np.log(
    df["rnnip_pb"] / (FRAC_C * df["rnnip_pc"] + (1 - FRAC_C) * df["rnnip_pu"])
)

# defining boolean arrays to select the different flavour classes
is_light = df["HadronConeExclTruthLabelID"] == 0
is_c = df["HadronConeExclTruthLabelID"] == 4
is_b = df["HadronConeExclTruthLabelID"] == 5

flav_cat = global_config["flavour_categories"]
taggers = ["dips", "rnnip"]
linestyles = ["solid", (0, (1, 1))]

# Initialise histogram plot
plot_histo = HistogramPlot(
    n_ratio_panels=1,
    ylabel="Normalised number of jets",
    ylabel_ratio_1="Ratio to DIPS",
    xlabel="$b$-jet discriminant",
    logy=False,
    leg_ncol=1,
    figsize=(5.5, 4.5),
    bins=np.linspace(-10, 10, 30),
    y_scale=1.5,
    ymax_ratio_1=1.5,
    ymin_ratio_1=0.5,
    atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets \ndummy sample, $f_{c}=0.018$",
)

# Add the histograms
for tagger, linestyle in zip(taggers, linestyles):

    plot_histo.add(
        Histogram(
            df[is_light][f"disc_{tagger}"],
            label="Light-flavour jets" if tagger == "dips" else None,
            colour=flav_cat["ujets"]["colour"],
            ratio_group="ujets",
            linestyle=linestyle,
        ),
        reference=tagger == "dips",
    )
    plot_histo.add(
        Histogram(
            df[is_c][f"disc_{tagger}"],
            label="$c$-jets" if tagger == "dips" else None,
            colour=flav_cat["cjets"]["colour"],
            ratio_group="cjets",
            linestyle=linestyle,
        ),
        reference=tagger == "dips",
    )
    plot_histo.add(
        Histogram(
            df[is_b][f"disc_{tagger}"],
            label="$b$-jets" if tagger == "dips" else None,
            colour=flav_cat["bjets"]["colour"],
            ratio_group="bjets",
            linestyle=linestyle,
        ),
        reference=tagger == "dips",
    )

plot_histo.draw()
plot_histo.make_linestyle_legend(
    linestyles=["solid", "dashed"], labels=["DIPS", "RNNIP"], bbox_to_anchor=(0.55, 1)
)
plot_histo.savefig("histogram_discriminant.png", transparent=False)
