"""Produce roc curves from tagger output and labels."""

from __future__ import annotations

from puma.hlplots import Results, Tagger
from puma.utils import get_dummy_2_taggers, logger

# The line below generates dummy data which is similar to a NN output
file = get_dummy_2_taggers(add_pt=True, return_file=True)

# define jet selections
cuts = [("n_truth_promptLepton", "==", 0)]

# define the taggers
dips = Tagger(
    name="dips",
    output_flavours=["ujets", "cjets", "bjets"],
    label="dummy DIPS ($f_{c}=0.005$)",
    fxs={"fc": 0.005},
    colour="#AA3377",
)
rnnip = Tagger(
    name="rnnip",
    output_flavours=["ujets", "cjets", "bjets"],
    label="dummy RNNIP ($f_{c}=0.07$)",
    fxs={"fc": 0.07},
    colour="#4477AA",
    reference=True,
)

# create the Results object
# for c-tagging use signal="cjets"
# for Xbb/cc-tagging use signal="hbb"/"hcc"
results = Results(signal="bjets", sample="dummy")

# load taggers from the file object
logger.info("Loading taggers.")
results.load_taggers_from_file(
    [dips, rnnip],
    file.filename,
    cuts=cuts,
    num_jets=len(file["jets"]),
)

results.atlas_second_tag = (
    "$\\sqrt{s}=13$ TeV, dummy jets \n$t\\bar{t}$, $20$ GeV $< p_{T} <250$ GeV"
)

# tagger probability distributions
results.plot_probs(logy=True, bins=40)

# tagger discriminant distributions
logger.info("Plotting tagger discriminant plots.")
results.plot_discs(logy=False, wp_vlines=[60, 85])
results.plot_discs(logy=True, wp_vlines=[60, 85], suffix="log")

# ROC curves
logger.info("Plotting ROC curves.")
results.plot_rocs()

# eff/rej vs. variable plots
logger.info("Plotting efficiency/rejection vs pT curves.")
results.atlas_second_tag = "$\\sqrt{s}=13$ TeV, dummy jets \n$t\\bar{t}$"

# or alternatively also pass the argument `working_point` to the plot_var_perf function.
# specifying the `disc_cut` per tagger is also possible.
results.plot_var_perf(
    working_point=0.7,
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    flat_per_bin=False,
)

results.atlas_second_tag = "$\\sqrt{s}=13$ TeV, dummy jets \n$t\\bar{t}$"
results.plot_var_perf(
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    flat_per_bin=True,
    working_point=0.7,
    h_line=0.7,
    disc_cut=None,
)
# flat rej vs. variable plots, a third tag is added relating to the fixed
#  rejection per bin
results.atlas_second_tag = "$\\sqrt{s}=13$ TeV, dummy jets \n$t\\bar{t}$"
results.plot_flat_rej_var_perf(
    fixed_rejections={"cjets": 2.2, "ujets": 1.2},
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
)

# fraction scan plots
logger.info("Plotting fraction scans.")
results.atlas_second_tag = "$\\sqrt{s}=13$ TeV, dummy jets \n$t\\bar{t}$\n70% WP"
results.plot_fraction_scans(backgrounds_to_plot=["cjets", "ujets"], efficiency=0.7, rej=False)
