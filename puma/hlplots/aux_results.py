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
from puma.var_vs_aux import VarVsAux, VarVsAuxPlot


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

    def plot_var_vtx_eff(
        self,
        suffix: str | None = None,
        xlabel: str = r"$p_{T}$ [GeV]",
        x_var: str = "pt",
        **kwargs,
    ):
        # define the curves
        plot_vtx_eff = VarVsAuxPlot(
            mode="efficiency",
            ylabel="Vertexing efficiency",
            xlabel=xlabel,
            logy=False,
            atlas_first_tag=self.atlas_first_tag,
            atlas_second_tag=self.signal.label,
            y_scale=1.4,
        )

        for tagger in self.taggers.values():
            is_signal = tagger.is_flav(self.signal)

            plot_vtx_eff.add(
                VarVsAux(
                    x_var=tagger.perf_var[is_signal],
                    n_match=tagger.aux_metrics["n_match"][is_signal],
                    n_true=tagger.aux_metrics["n_ref"][is_signal],
                    n_reco=tagger.aux_metrics["n_test"][is_signal],
                    label=tagger.label,
                    colour=tagger.colour,
                    **kwargs,
                ),
                reference=tagger.reference,
            )

        plot_vtx_eff.draw()

        plot_details = f"vtx_eff_vs_{x_var}"
        plot_vtx_eff.savefig(self.get_filename(plot_details, suffix))

    def plot_var_vtx_fr(
        self,
        suffix: str | None = None,
        xlabel: str = r"$p_{T}$ [GeV]",
        x_var: str = "pt",
        **kwargs,
    ):
        # define the curves
        plot_vtx_fr = VarVsAuxPlot(
            mode="fake_rate",
            ylabel="Vertexing fake rate",
            xlabel=xlabel,
            logy=False,
            atlas_first_tag=self.atlas_first_tag,
            atlas_second_tag=self.signal.label,
            y_scale=1.4,
        )

        for tagger in self.taggers.values():
            is_signal = tagger.is_flav(self.signal)

            plot_vtx_fr.add(
                VarVsAux(
                    x_var=tagger.perf_var[is_signal],
                    n_match=tagger.aux_metrics["n_match"][is_signal],
                    n_true=tagger.aux_metrics["n_ref"][is_signal],
                    n_reco=tagger.aux_metrics["n_test"][is_signal],
                    label=tagger.label,
                    colour=tagger.colour,
                    **kwargs,
                ),
                reference=tagger.reference,
            )

        plot_vtx_fr.draw()

        plot_details = f"vtx_fr_vs_{x_var}"
        plot_vtx_fr.savefig(self.get_filename(plot_details, suffix))

    def plot_var_vtx_trk_eff(
        self,
        suffix: str | None = None,
        xlabel: str = r"$p_{T}$ [GeV]",
        x_var: str = "pt",
        **kwargs,
    ):
        # define the curves
        plot_vtx_eff = VarVsAuxPlot(
            mode="efficiency",
            ylabel="Track-vertex association efficiency",
            xlabel=xlabel,
            logy=False,
            atlas_first_tag=self.atlas_first_tag,
            atlas_second_tag=self.signal.label,
            y_scale=1.4,
        )

        for tagger in self.taggers.values():
            is_signal = tagger.is_flav(self.signal)
            include_sum = tagger.aux_metrics["track_overlap"][is_signal] >= 0

            plot_vtx_eff.add(
                VarVsAux(
                    x_var=tagger.perf_var[is_signal],
                    n_match=np.sum(
                        tagger.aux_metrics["track_overlap"][is_signal],
                        axis=1,
                        where=include_sum,
                    ),
                    n_true=np.sum(
                        tagger.aux_metrics["ref_vertex_size"][is_signal],
                        axis=1,
                        where=include_sum,
                    ),
                    n_reco=np.sum(
                        tagger.aux_metrics["test_vertex_size"][is_signal],
                        axis=1,
                        where=include_sum,
                    ),
                    label=tagger.label,
                    colour=tagger.colour,
                    **kwargs,
                ),
                reference=tagger.reference,
            )

        plot_vtx_eff.draw()

        plot_details = f"vtx_trk_eff_vs_{x_var}"
        plot_vtx_eff.savefig(self.get_filename(plot_details, suffix))

    def plot_var_vtx_trk_fr(
        self,
        suffix: str | None = None,
        xlabel: str = r"$p_{T}$ [GeV]",
        x_var: str = "pt",
        **kwargs,
    ):
        # define the curves
        plot_vtx_eff = VarVsAuxPlot(
            mode="fake_rate",
            ylabel="Track-vertex association fake rate",
            xlabel=xlabel,
            logy=False,
            atlas_first_tag=self.atlas_first_tag,
            atlas_second_tag=self.signal.label,
            y_scale=1.4,
        )

        for tagger in self.taggers.values():
            is_signal = tagger.is_flav(self.signal)
            include_sum = tagger.aux_metrics["track_overlap"][is_signal] >= 0

            plot_vtx_eff.add(
                VarVsAux(
                    x_var=tagger.perf_var[is_signal],
                    n_match=np.sum(
                        tagger.aux_metrics["track_overlap"][is_signal],
                        axis=1,
                        where=include_sum,
                    ),
                    n_true=np.sum(
                        tagger.aux_metrics["ref_vertex_size"][is_signal],
                        axis=1,
                        where=include_sum,
                    ),
                    n_reco=np.sum(
                        tagger.aux_metrics["test_vertex_size"][is_signal],
                        axis=1,
                        where=include_sum,
                    ),
                    label=tagger.label,
                    colour=tagger.colour,
                    **kwargs,
                ),
                reference=tagger.reference,
            )

        plot_vtx_eff.draw()

        plot_details = f"vtx_trk_fr_vs_{x_var}"
        plot_vtx_eff.savefig(self.get_filename(plot_details, suffix))
