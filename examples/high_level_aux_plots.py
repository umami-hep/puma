"""Produce aux task plots from tagger output and labels."""

from __future__ import annotations

from puma.hlplots import AuxResults, Tagger
from puma.utils import get_dummy_tagger_aux, logger

# The line below generates dummy data which is similar to a NN output
fname, file = get_dummy_tagger_aux()

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
aux_results = AuxResults(sample="dummy")

# load tagger from the file object
logger.info("Loading taggers.")
aux_results.load_taggers_from_file(
    [GN2],
    fname,
    cuts=cuts,
    num_jets=len(file["jets"]),
)

aux_results.atlas_second_tag = (
    "$\\sqrt{s}=13$ TeV, dummy jets \n$t\\bar{t}$, $20$ GeV $< p_{T} <250$ GeV"
)

# vertexing performance for b-jets
logger.info("Plotting vertexing performance.")
aux_results.plot_var_vtx_perf(flavour="bjets")
