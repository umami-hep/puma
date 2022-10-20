"""Produce roc curves from tagger output and labels."""
# from pathlib import Path

# import h5py
import numpy as np

# import pandas as pd
from plot_settings import base_dir, colours, file_dir, load_df
from plot_utils import Results, Tagger, decorate_df

# from umami.helper_tools import get_class_label_ids
from puma.utils import logger

n_jets = 1_000_000
file_name = "inclusive_testing_ttbar_PFlow.h5"


# pT plots for Spice120 vs AllAux vs GN1

dl1d = Tagger("DL1dv01")
dl1d.label = "DL1dv01 ($f_{c}=0.018$)"
dl1d.f_c = 0.018
dl1d.colour = colours[dl1d.model_name]
# dl1d.reference = True
gn1 = Tagger("GN120220509")
gn1.label = "GN1 ($f_{c}=0.05$)"
gn1.f_c = 0.05
gn1.reference = True
gn1.colour = colours[gn1.model_name]
# trans = Tagger("FlavTrans_v1")
# trans.label = "Transformer ($f_{c}=0.018$)"
# trans.f_c = 0.018
# trans_allaux = Tagger("AllAux")
# trans_allaux.label = "Transformer all aux ($f_{c}=0.018$)"
# trans_allaux.f_c = 0.018
trans_AllAux = Tagger("AllAux")
trans_AllAux.label = "Spice ($f_{c}=0.012$)"
trans_AllAux.f_c = 0.02
trans_Spice120 = Tagger("Spice120")
trans_Spice120.label = "Spice(120M) ($f_{c}=0.02$)"
trans_Spice120.f_c = 0.02
trans_Spice120.colour = colours[trans_Spice120.model_name]


results = Results()
# results.add(dl1d)
results.add(gn1)
results.add(trans_AllAux)
results.add(trans_Spice120)

results.sig_eff = np.linspace(0.6, 0.95, 20)
results.atlas_second_tag = (
    "$\\sqrt{s}=13$ TeV, PFlow jets \n$t\\bar{t}$, $20<p_{T}<250$ GeV"
)

df = load_df(file_dir / file_name, n_jets)
# Get the class ids for removing
class_ids = [0, 4, 5]
# Remove all jets which are not trained on
df.query(f"HadronConeExclTruthLabelID in {class_ids}", inplace=True)

## Load the network's exported values
for model in results.model_names:
    if f"{model}_pb" in df.columns:
        continue
    decorate_df(df, base_dir / model / file_name, model)


df.query("pt_btagJes < 250", inplace=True)
logger.info("caclulate tagger discriminants")

results.plot_rocs(df, "plots/roc-trans.png")

# results.atlas_second_tag = (
#     "$\\sqrt{s}=13$ TeV, PFlow jets \n$t\\bar{t}$, $20<p_{T}<250$ GeV\n" "70% WP"
# )
# dl1d.disc_cut = 3.493
# gn1.disc_cut = 3.642
# trans.disc_cut = 2.794
# trans_allaux.disc_cut = 2.8072676
# # trans.working_point = 0.7

# results.plot_pt_perf(
#     df,
#     plot_name="plots/transformer",
#     bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
#     fixed_eff_bin=False,
# )

# results.atlas_second_tag = (
#     "$\\sqrt{s}=13$ TeV, PFlow jets \n$t\\bar{t}$, $20<p_{T}<250$ GeV\n"
#     "70% WP per bin"
# )

# results.plot_pt_perf(
#     df,
#     plot_name="plots/transformer_fixed_per_bin",
#     bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
#     fixed_eff_bin=True,
#     working_point=0.7,
#     disc_cut=None,
# )
# print(
#     "70% wp cut tranfsormer:",
#     np.percentile(trans_allaux.discs[results.is_b], 100.0 * (1.0 - 0.7)),
# )

# gn1.reference = True
# results.plot_discs(df, "plots/trans-disc.png", ["DL1dv01", "Trans_allaux"])
