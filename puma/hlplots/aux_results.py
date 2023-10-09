"""Auxiliary task results module for high level API."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ftag import Cuts
from ftag.hdf5 import H5Reader

from puma.hlplots.results import Results
from puma.hlplots.tagger import Tagger
from puma.utils.vertexing import calculate_vertex_metrics


@dataclass
class AuxResults(Results):
    """Store information about several taggers and plot auxiliary task results."""

    def add_taggers_from_file(  # pylint: disable=R0913
        self,
        taggers: list[Tagger],
        file_path: Path | str,
        key="jets",
        label_var="HadronConeExclTruthLabelID",
        aux_key="tracks",
        vtx_label_var="truthVertexIndex",
        vtx_reco_var="VertexIndex",
        cuts: Cuts | list | None = None,
        num_jets: int | None = None,
        perf_var: str | None = None,
    ):
        """Load one or more taggers from a common file.

        Parameters
        ----------
        taggers : list[Tagger]
            List of taggers to add
        file_path : str | Path
            Path to file
        key : str, optional
            Key in file, by default 'jets'
        label_var : str
            Label variable to use, by default 'HadronConeExclTruthLabelID'
        aux_key : str, optional
            Key for auxiliary information, by default 'tracks'
        vtx_label_var : str, optional
            Vertex label variable, by default 'truthVertexIndex'
        vtx_reco_var : str, optional
            Vertex reconstruction variable, by default 'VertexIndex'
        cuts : Cuts | list, optional
            Cuts to apply, by default None
        num_jets : int, optional
            Number of jets to load from the file, by default all jets
        perf_var : np.ndarray, optional
            Override the performance variable to use, by default None
        """
        # set tagger output nodes
        for tagger in taggers:
            tagger.output_flavours = self.flavours

        # get a list of all variables to be loaded from the file
        if not isinstance(cuts, Cuts):
            cuts = Cuts.empty() if cuts is None else Cuts.from_list(cuts)
        var_list = sum([tagger.variables for tagger in taggers], [label_var])
        var_list += cuts.variables
        var_list += sum([t.cuts.variables for t in taggers if t.cuts is not None], [])
        var_list = list(set(var_list + [self.perf_var]))

        # load data
        reader = H5Reader(file_path, precision="full")
        data = reader.load({key: var_list}, num_jets)[key]
        aux_reader = H5Reader(file_path, precision="full", jets_name="tracks")
        aux_data = aux_reader.load({aux_key: [vtx_label_var, vtx_reco_var]}, num_jets)[
            aux_key
        ]

        # apply common cuts
        if cuts:
            idx, data = cuts(data)
            aux_data = aux_data[idx]
            if perf_var is not None:
                perf_var = perf_var[idx]

        # for each tagger
        for tagger in taggers:
            sel_data = data
            sel_aux_data = aux_data
            sel_perf_var = perf_var

            # apply tagger specific cuts
            if tagger.cuts:
                idx, sel_data = tagger.cuts(data)
                sel_aux_data = aux_data[idx]
                if perf_var is not None:
                    sel_perf_var = perf_var[idx]

            # calculate vertexing metrics
            tagger.aux_metrics = calculate_vertex_metrics(
                sel_aux_data[vtx_reco_var],
                sel_aux_data[vtx_label_var],
            )

            # attach data to tagger objects
            tagger.extract_tagger_scores(sel_data, source_type="structured_array")
            tagger.labels = np.array(sel_data[label_var], dtype=[(label_var, "i4")])
            if perf_var is None:
                tagger.perf_var = sel_data[self.perf_var]
                if any(x in self.perf_var for x in ["pt", "mass"]):
                    tagger.perf_var = tagger.perf_var * 0.001
            else:
                tagger.perf_var = sel_perf_var

            # add tagger to results
            self.add(tagger)

    def plot_var_eff(
        self,
        suffix: str | None = None,
        xlabel: str = r"$p_{T}$ [GeV]",
        x_var: str = "pt",
        h_line: float | None = None,
        working_point: float | None = None,
        disc_cut: float | None = None,
        **kwargs,
    ):
        pass
    
    def plot_var_purity(
        self,
        suffix: str | None = None,
        xlabel: str = r"$p_{T}$ [GeV]",
        x_var: str = "pt",
        h_line: float | None = None,
        working_point: float | None = None,
        disc_cut: float | None = None,
        **kwargs,
    ):
        pass

    def plot_size_eff(
        self,
        suffix: str | None = None,
        xlabel: str = r"$p_{T}$ [GeV]",
        x_var: str = "pt",
        h_line: float | None = None,
        working_point: float | None = None,
        disc_cut: float | None = None,
        **kwargs,
    ):
        pass

    def plot_size_purity(
        self,
        suffix: str | None = None,
        xlabel: str = r"$p_{T}$ [GeV]",
        x_var: str = "pt",
        h_line: float | None = None,
        working_point: float | None = None,
        disc_cut: float | None = None,
        **kwargs,
    ):
        pass
