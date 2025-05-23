"""Produce roc curves from tagger output and labels."""

from __future__ import annotations

import numpy as np
from ftag import Flavours
from ftag.utils import calculate_rejection, get_discriminant

from puma import Roc, RocPlot
from puma.utils import get_dummy_2_taggers, logger

# The line below generates dummy data which is similar to a NN output
df = get_dummy_2_taggers(add_pt=True)

logger.info("caclulate tagger discriminants")
discs_dips = get_discriminant(
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
discs_rnnip = get_discriminant(
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

# defining target efficiency
sig_eff = np.linspace(0.49, 1, 20)

# defining boolean arrays to select the different flavour classes
is_light = df["HadronConeExclTruthLabelID"] == 0
is_c = df["HadronConeExclTruthLabelID"] == 4
is_b = df["HadronConeExclTruthLabelID"] == 5

n_jets_light = sum(is_light)
n_jets_c = sum(is_c)

logger.info("Calculate rejection")
rnnip_ujets_rej = calculate_rejection(discs_rnnip[is_b], discs_rnnip[is_light], sig_eff)
rnnip_cjets_rej = calculate_rejection(discs_rnnip[is_b], discs_rnnip[is_c], sig_eff)
dips_ujets_rej = calculate_rejection(discs_dips[is_b], discs_dips[is_light], sig_eff)
dips_cjets_rej = calculate_rejection(discs_dips[is_b], discs_dips[is_c], sig_eff)

# Define now the actual ROC curve objects
rnnip_ujets_roc = Roc(
    sig_eff=sig_eff,
    bkg_rej=rnnip_ujets_rej,
    n_test=n_jets_light,
    rej_class="ujets",
    signal_class="bjets",
    label="RNNIP",
)
dips_ujets_roc = Roc(
    sig_eff=sig_eff,
    bkg_rej=dips_ujets_rej,
    n_test=n_jets_light,
    rej_class="ujets",
    signal_class="bjets",
    label="DIPS r22",
)
rnnip_cjets_roc = Roc(
    sig_eff=sig_eff,
    bkg_rej=rnnip_cjets_rej,
    n_test=n_jets_c,
    rej_class="cjets",
    signal_class="bjets",
    label="RNNIP",
)
dips_cjets_roc = Roc(
    sig_eff=sig_eff,
    bkg_rej=dips_cjets_rej,
    n_test=n_jets_c,
    rej_class="cjets",
    signal_class="bjets",
    label="DIPS r22",
)

# ROC curve objects can also be stored as yaml or json files
rnnip_ujets_roc.save("rnnip_ujets_roc.yaml")
dips_ujets_roc.save("dips_ujets_roc.yaml")
rnnip_cjets_roc.save("rnnip_cjets_roc.yaml")
dips_cjets_roc.save("dips_cjets_roc.yaml")

# here the plotting of the roc starts
logger.info("Plotting ROC curves.")
plot_roc = RocPlot(
    n_ratio_panels=2,
    ylabel="Background rejection",
    xlabel="$b$-jet efficiency",
    atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets \ndummy sample, $f_{c}=0.018$",
    figsize=(6.5, 6),
    y_scale=1.4,
)

# Add the ROC curve objects to the plot
plot_roc.add_roc(roc_curve=rnnip_ujets_roc, reference=True)
plot_roc.add_roc(roc_curve=dips_ujets_roc)
plot_roc.add_roc(roc_curve=rnnip_cjets_roc, reference=True)
plot_roc.add_roc(roc_curve=dips_cjets_roc)

# setting which flavour rejection ratio is drawn in which ratio panel
plot_roc.set_ratio_class(1, "ujets")
plot_roc.set_ratio_class(2, "cjets")

plot_roc.draw()
plot_roc.savefig("roc.png", transparent=False)

# If you want to load now the ROC curves from file, you can do so
loaded_rnnip_ujets_roc = Roc.load("rnnip_ujets_roc.yaml")
loaded_dips_ujets_roc = Roc.load("dips_ujets_roc.yaml")
loaded_rnnip_cjets_roc = Roc.load("rnnip_cjets_roc.yaml", colour="red")
loaded_dips_cjets_roc = Roc.load("dips_cjets_roc.yaml", colour="red")


# Now init a new plot
loaded_plot_roc = RocPlot(
    n_ratio_panels=2,
    ylabel="Background rejection",
    xlabel="$b$-jet efficiency",
    atlas_second_tag="$\\sqrt{s}=13$ TeV, Loaded ROC Curves",
    figsize=(6.5, 6),
    y_scale=1.4,
)

loaded_plot_roc.add_roc(roc_curve=rnnip_ujets_roc, reference=True)
loaded_plot_roc.add_roc(roc_curve=dips_ujets_roc)
loaded_plot_roc.add_roc(roc_curve=rnnip_cjets_roc, reference=True)
loaded_plot_roc.add_roc(roc_curve=dips_cjets_roc)

# setting which flavour rejection ratio is drawn in which ratio panel
loaded_plot_roc.set_ratio_class(1, "ujets")
loaded_plot_roc.set_ratio_class(2, "cjets")

loaded_plot_roc.draw()
loaded_plot_roc.savefig("roc_loaded.png", transparent=False)
