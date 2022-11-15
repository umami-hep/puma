"""Produce roc curves from tagger output and labels."""
# from pathlib import Path

# import h5py
import numpy as np

from puma.hlplots.results import Results
from puma.hlplots.tagger import Tagger
from puma.utils import get_dummy_2_taggers, logger

# import pandas as pd
# from plot_settings import base_dir, colours, file_dir, load_df


# The line below generates dummy data which is similar to a NN output
df = get_dummy_2_taggers()


# n_jets = 1_000_000
# file_name = "inclusive_testing_ttbar_PFlow.h5"
logger.info("Start plotting")

tagger_args = {
    "working_point": 0.7,
    "sig_eff": np.linspace(0.6, 0.95, 20),
    "is_light": df["HadronConeExclTruthLabelID"] == 0,
    "is_c": df["HadronConeExclTruthLabelID"] == 4,
    "is_b": df["HadronConeExclTruthLabelID"] == 5,
}

dips = Tagger("dips", template=tagger_args)
dips.label = "dummy DIPS ($f_{c}=0.05$)"
dips.f_c = 0.05
dips.colour = "#AA3377"
rnnip = Tagger("rnnip")
rnnip.label = "dummy RNNIP ($f_{c}=0.05$)"
rnnip.f_c = 0.05
rnnip.colour = "#EE6677"
rnnip.reference = True


results = Results()
results.add(dips)
results.add(rnnip)

results.sig_eff = np.linspace(0.6, 0.95, 20)
results.atlas_second_tag = (
    "$\\sqrt{s}=13$ TeV, dummy jets \n$t\\bar{t}$, $20<p_{T}<250$ GeV"
)

# df = load_df(file_dir / file_name, n_jets)
# # Get the class ids for removing
# class_ids = [0, 4, 5]
# # Remove all jets which are not trained on
# df.query(f"HadronConeExclTruthLabelID in {class_ids}", inplace=True)

# ## Load the network's exported values
# for model in results.model_names:
#     if f"{model}_pb" in df.columns:
#         continue
#     decorate_df(df, base_dir / model / file_name, model)


# df.query("pt_btagJes < 250", inplace=True)
# logger.info("caclulate tagger discriminants")

# results.plot_rocs(df, "plots/roc-trans.png")

# # results.atlas_second_tag = (
# #     "$\\sqrt{s}=13$ TeV, PFlow jets \n$t\\bar{t}$, $20<p_{T}<250$ GeV\n" "70% WP"
# # )
# # dl1d.disc_cut = 3.493
# # gn1.disc_cut = 3.642
# # trans.disc_cut = 2.794
# # trans_allaux.disc_cut = 2.8072676
# # # trans.working_point = 0.7

# # results.plot_pt_perf(
# #     df,
# #     plot_name="plots/transformer",
# #     bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
# #     fixed_eff_bin=False,
# # )

# # results.atlas_second_tag = (
# #     "$\\sqrt{s}=13$ TeV, PFlow jets \n$t\\bar{t}$, $20<p_{T}<250$ GeV\n"
# #     "70% WP per bin"
# # )

# # results.plot_pt_perf(
# #     df,
# #     plot_name="plots/transformer_fixed_per_bin",
# #     bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
# #     fixed_eff_bin=True,
# #     working_point=0.7,
# #     disc_cut=None,
# # )
# # print(
# #     "70% wp cut tranfsormer:",
# #     np.percentile(trans_allaux.discs[results.is_b], 100.0 * (1.0 - 0.7)),
# # )

# # gn1.reference = True
# # results.plot_discs(df, "plots/trans-disc.png", ["DL1dv01", "Trans_allaux"])
