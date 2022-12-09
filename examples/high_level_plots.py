"""Produce roc curves from tagger output and labels."""
# from pathlib import Path

# import h5py
import numpy as np

from puma.hlplots import Results, Tagger
from puma.utils import get_dummy_2_taggers, logger

# The line below generates dummy data which is similar to a NN output
df = get_dummy_2_taggers(add_pt=True)
class_ids = [0, 4, 5]
# Remove all jets which are not trained on
df.query(f"HadronConeExclTruthLabelID in {class_ids}", inplace=True)
df.query("pt < 250e3", inplace=True)

logger.info("Start plotting")

tagger_args = {
    "perf_var": df["pt"] / 1e3,
    "is_light": df["HadronConeExclTruthLabelID"] == 0,
    "is_c": df["HadronConeExclTruthLabelID"] == 4,
    "is_b": df["HadronConeExclTruthLabelID"] == 5,
}


dips = Tagger("dips", template=tagger_args)
dips.label = "dummy DIPS ($f_{c}=0.005$)"
dips.f_c = 0.005
dips.f_b = 0.04
dips.colour = "#AA3377"
dips.extract_tagger_scores(df)

rnnip = Tagger("rnnip", template=tagger_args)
rnnip.label = "dummy RNNIP ($f_{c}=0.07$)"
rnnip.f_c = 0.07
rnnip.f_b = 0.04
rnnip.colour = "#4477AA"
rnnip.reference = True
rnnip.extract_tagger_scores(df)


results = Results()
results.add(dips)
results.add(rnnip)


results.sig_eff = np.linspace(0.6, 0.95, 20)
results.atlas_second_tag = (
    "$\\sqrt{s}=13$ TeV, dummy jets \n$t\\bar{t}$, $20<p_{T}<250$ GeV"
)

# tagger discriminant plots
logger.info("Plotting tagger discriminant plots.")
results.plot_discs("dummy_disc.png")


logger.info("Plotting ROC curves.")
# ROC curves as a function of the b-jet efficiency
results.plot_rocs("roc_test.png")
# ROC curves as a function of the c-jet efficiency
results.plot_rocs("roc_test_c.png", signal_class="cjets")


logger.info("Plotting efficiency/rejection vs pT curves.")
# eff/rej vs. variable plots
results.atlas_second_tag = "$\\sqrt{s}=13$ TeV, dummy jets \n$t\\bar{t}$\n70% WP"
# you can either specify a WP per tagger
# dips.working_point = 0.7
# rnnip.working_point = 0.7
# or alternatively also pass the argument `working_point` to the plot_var_perf function.
# to specify the `disc_cut` per tagger is also possible.
results.plot_var_perf(
    plot_name="dummy_tagger",
    working_point=0.7,
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    fixed_eff_bin=False,
)

results.atlas_second_tag = (
    "$\\sqrt{s}=13$ TeV, dummy jets \n$t\\bar{t}$\n70% WP per bin"
)
results.plot_var_perf(
    plot_name="dummy_tagger_fixed_per_bin",
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    fixed_eff_bin=True,
    working_point=0.7,
    disc_cut=None,
)

logger.info("Plotting discriminants.")
results.atlas_second_tag = "$\\sqrt{s}=13$ TeV, dummy jets \n$t\\bar{t}$"
# b-jet discriminant
results.plot_discs(plot_name="dummy_tagger_disc")
# c-jet discriminant
results.plot_discs(plot_name="dummy_tagger_disc", signal_class="cjets")
