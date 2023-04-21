"""Produce roc curves from tagger output and labels."""

import numpy as np

from puma.hlplots import Results, Tagger
from puma.utils import get_dummy_2_taggers, logger

# The line below generates dummy data which is similar to a NN output
file = get_dummy_2_taggers(add_pt=True, return_file=True)

# define jet selections
cuts = [("n_truth_promptLepton", "==", 0)]

# define the taggers
dips = Tagger(
    name="dips",
    label="dummy DIPS ($f_{c}=0.005$)",
    f_c=0.005,
    f_b=0.04,
    colour="#AA3377",
)
rnnip = Tagger(
    name="rnnip",
    label="dummy RNNIP ($f_{c}=0.07$)",
    f_c=0.07,
    f_b=0.04,
    colour="#4477AA",
    reference=True,
)

# create the Results object
# for c-tagging use signal="cjets"
# for Xbb/cc-tagging use signal="hbb"/"hcc"
results = Results(signal="bjets")

# load taggers from the file object
logger.info("Loading taggers.")
results.add_taggers_from_file(
    [dips, rnnip],
    file.filename,
    cuts=cuts,
    num_jets=len(file["jets"]),
)

results.sig_eff = np.linspace(0.6, 0.95, 20)
results.atlas_second_tag = (
    "$\\sqrt{s}=13$ TeV, dummy jets \n$t\\bar{t}$, $20$ GeV $< p_{T} <250$ GeV"
)

# tagger discriminant plots
logger.info("Plotting tagger discriminant plots.")
results.plot_discs("hlplots_disc_b.png")

# ROC curves
logger.info("Plotting ROC curves.")
results.plot_rocs("hlplots_roc_b.png")


logger.info("Plotting efficiency/rejection vs pT curves.")
# eff/rej vs. variable plots
results.atlas_second_tag = "$\\sqrt{s}=13$ TeV, dummy jets \n$t\\bar{t}$\n70% WP"
# you can either specify a WP per tagger
# dips.working_point = 0.7
# rnnip.working_point = 0.7
# or alternatively also pass the argument `working_point` to the plot_var_perf function.
# to specify the `disc_cut` per tagger is also possible.
results.plot_var_perf(
    plot_name="hlplots_dummy_tagger_pt",
    working_point=0.7,
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    fixed_eff_bin=False,
)

results.atlas_second_tag = (
    "$\\sqrt{s}=13$ TeV, dummy jets \n$t\\bar{t}$\n70% WP per bin"
)
results.plot_var_perf(
    plot_name="hlplots_dummy_tagger_pt_fixed_per_bin",
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    fixed_eff_bin=True,
    working_point=0.7,
    h_line=0.7,
    disc_cut=None,
)
