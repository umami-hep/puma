"""Produce aux task plots from tagger output and labels."""
from __future__ import annotations

from puma.hlplots import AuxResults, Tagger
from puma.utils import get_dummy_tagger_aux, logger

# The line below generates dummy data which is similar to a NN output
file = get_dummy_tagger_aux()

# define jet selections
cuts = [("n_truth_promptLepton", "==", 0)]

# define the tagger
GN2 = Tagger(
    name="GN2",
    label="dummy GN2",
    colour="#4477AA",
    reference=True,
)

# create the AuxResults object
# for c-tagging use signal="cjets"
# for Xbb/cc-tagging use signal="hbb"/"hcc"
aux_results = AuxResults(signal="bjets", sample="dummy")

# load tagger from the file object
logger.info("Loading taggers.")
aux_results.add_taggers_from_file(
    [GN2],
    file.filename,
    cuts=cuts,
    num_jets=len(file["jets"]),
)

aux_results.atlas_second_tag = (
    "$\\sqrt{s}=13$ TeV, dummy jets \n$t\\bar{t}$, $20$ GeV $< p_{T} <250$ GeV"
)

# vertexing efficiency and fake rate
logger.info("Plotting vertexing efficiency and fake rate.")
aux_results.plot_var_vtx_eff()
aux_results.plot_var_vtx_fr()

# track to vertex association efficiency and fake rate
logger.info("Plotting track to vertex association efficiency and fake rate.")
aux_results.plot_var_vtx_trk_eff()
aux_results.plot_var_vtx_trk_fr()
