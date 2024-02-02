"""Auxiliary task results module for high level API."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ftag import Cuts
from ftag.hdf5 import H5Reader

from puma.hlplots.results import Results
from puma.hlplots.tagger import Tagger
from puma.utils.vertexing import calculate_vertex_metrics, clean_indices
from puma.var_vs_aux import VarVsAux, VarVsAuxPlot


def get_aux_labels():
    """Get the truth labels for all aux tasks."""
    return {
        "vertexing": "ftagTruthVertexIndex",
        "track_origin": "ftagTruthOriginLabel",
    }


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
        var_list = [label_var]
        var_list += cuts.variables
        var_list += sum([t.cuts.variables for t in taggers if t.cuts is not None], [])
        var_list = list(set(var_list + [self.perf_var]))

        aux_labels = get_aux_labels()

        aux_var_list = sum([list(t.aux_variables.values()) for t in taggers], [])
        aux_var_list += sum([list(aux_labels.values()) for t in taggers], [])
        aux_var_list = list(set(aux_var_list))

        # load data
        reader = H5Reader(file_path, precision="full")
        data = reader.load({key: var_list}, num_jets)[key]
        aux_reader = H5Reader(file_path, precision="full", jets_name="tracks")
        aux_data = aux_reader.load({aux_key: aux_var_list}, num_jets)[aux_key]

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

            # attach data to tagger objects
            tagger.labels = np.array(sel_data[label_var], dtype=[(label_var, "i4")])
            for task in tagger.aux_tasks:
                tagger.aux_scores[task] = sel_aux_data[tagger.aux_variables[task]]
            for task in aux_labels.keys():
                tagger.aux_labels[task] = sel_aux_data[aux_labels[task]]
            if perf_var is None:
                tagger.perf_var = sel_data[self.perf_var]
                if any(x in self.perf_var for x in ["pt", "mass"]):
                    tagger.perf_var = tagger.perf_var * 0.001
            else:
                tagger.perf_var = sel_perf_var

            # add tagger to results
            self.add(tagger)

    def plot_var_vtx_perf(
        self,
        suffix: str | None = None,
        xlabel: str = r"$p_{T}$ [GeV]",
        xvar: str = "pt",
        incl_vertexing: bool = False,
        **kwargs,
    ):
        # define the curves
        plot_vtx_eff = VarVsAuxPlot(
            mode="efficiency",
            ylabel="Vertexing efficiency",
            xlabel=xlabel,
            logy=False,
            atlas_first_tag=self.atlas_first_tag,
            atlas_second_tag=self.atlas_second_tag,
            y_scale=1.4,
        )
        plot_vtx_purity = VarVsAuxPlot(
            mode="purity",
            ylabel="Vertexing purity",
            xlabel=xlabel,
            logy=False,
            atlas_first_tag=self.atlas_first_tag,
            atlas_second_tag=self.atlas_second_tag,
            y_scale=1.4,
        )
        plot_vtx_trk_eff = VarVsAuxPlot(
            mode="efficiency",
            ylabel="Track-vertex association efficiency",
            xlabel=xlabel,
            logy=False,
            atlas_first_tag=self.atlas_first_tag,
            atlas_second_tag=self.atlas_second_tag,
            y_scale=1.4,
        )
        plot_vtx_trk_purity = VarVsAuxPlot(
            mode="purity",
            ylabel="Track-vertex association purity",
            xlabel=xlabel,
            logy=False,
            atlas_first_tag=self.atlas_first_tag,
            atlas_second_tag=self.atlas_second_tag,
            y_scale=1.4,
        )

        for tagger in self.taggers.values():
            if "vertexing" not in tagger.aux_tasks:
                continue

            # clean truth vertex indices - remove indices from true PV, pileup, fake
            truth_indices = clean_indices(
                tagger.aux_labels["vertexing"],
                tagger.aux_labels["track_origin"] < 3,
                mode="remove",
            )

            # merge truth vertices from HF for inclusive performance
            if incl_vertexing:
                truth_indices = clean_indices(
                    truth_indices,
                    np.isin(tagger.aux_labels["track_origin"], [3,4,5]),
                    mode="merge",
                )

            # clean reco vertex indices - remove indices from reco PV, pileup, fake
            if "track_origin" in tagger.aux_tasks:
                reco_indices = clean_indices(
                    tagger.aux_scores["vertexing"],
                    tagger.aux_scores["track_origin"] < 3,
                    mode="remove",
                )

                # merge reco vertices - from HF if track origin is available
                if incl_vertexing:
                    reco_indices = clean_indices(
                        reco_indices,
                        np.isin(tagger.aux_scores["track_origin"], [3,4,5]),
                        mode="merge",
                    )
            else:
                reco_indices = tagger.aux_scores["vertexing"]

                # merge reco vertices - all if track origin isn't available
                if incl_vertexing:
                    reco_indices = clean_indices(
                        reco_indices,
                        reco_indices > 0,
                        mode="merge",
                    )

            # calculate vertexing metrics
            vtx_metrics = calculate_vertex_metrics(reco_indices, truth_indices)

            is_signal = tagger.is_flav(self.signal)
            include_sum = vtx_metrics["track_overlap"][is_signal] >= 0

            vtx_perf = VarVsAux(
                x_var=tagger.perf_var[is_signal],
                n_match=vtx_metrics["n_match"][is_signal],
                n_true=vtx_metrics["n_ref"][is_signal],
                n_reco=vtx_metrics["n_test"][is_signal],
                label=tagger.label,
                colour=tagger.colour,
                **kwargs,
            )
            vtx_trk_perf = VarVsAux(
                x_var=tagger.perf_var[is_signal],
                n_match=np.sum(
                    vtx_metrics["track_overlap"][is_signal],
                    axis=1,
                    where=include_sum,
                ),
                n_true=np.sum(
                    vtx_metrics["ref_vertex_size"][is_signal],
                    axis=1,
                    where=include_sum,
                ),
                n_reco=np.sum(
                    vtx_metrics["test_vertex_size"][is_signal],
                    axis=1,
                    where=include_sum,
                ),
                label=tagger.label,
                colour=tagger.colour,
                **kwargs,
            )

            plot_vtx_eff.add(vtx_perf, reference=tagger.reference)
            plot_vtx_purity.add(vtx_perf, reference=tagger.reference)
            plot_vtx_trk_eff.add(vtx_trk_perf, reference=tagger.reference)
            plot_vtx_trk_purity.add(vtx_trk_perf, reference=tagger.reference)

        plot_vtx_eff.draw()
        plot_vtx_eff.savefig(self.get_filename(f"vtx_eff_vs_{xvar}", suffix))

        plot_vtx_purity.draw()
        plot_vtx_purity.savefig(self.get_filename(f"vtx_purity_vs_{xvar}", suffix))

        plot_vtx_trk_eff.draw()
        plot_vtx_trk_eff.savefig(self.get_filename(f"vtx_trk_eff_vs_{xvar}", suffix))

        plot_vtx_trk_purity.draw()
        plot_vtx_trk_purity.savefig(
            self.get_filename(f"vtx_trk_purity_vs_{xvar}", suffix)
        )
