"""Auxiliary task results module for high level API."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ftag import Cuts, Flavour, Flavours
from ftag.hdf5 import H5Reader

from puma.hlplots.tagger import Tagger
from puma.utils import logger
from puma.utils.aux import get_aux_labels
from puma.utils.vertexing import calculate_vertex_metrics
from puma.var_vs_aux import VarVsAux, VarVsAuxPlot


@dataclass
class AuxResults:
    """Store information about several taggers and plot auxiliary task results."""

    sample: str
    atlas_first_tag: str = "Simulation Internal"
    atlas_second_tag: str = None
    taggers: dict = field(default_factory=dict)
    perf_vars: str | tuple | list = "pt"
    output_dir: str | Path = "."
    extension: str = "png"

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        if isinstance(self.perf_vars, str):
            self.perf_vars = [self.perf_vars]

    def add(self, tagger):
        """Add tagger to class.

        Parameters
        ----------
        tagger : puma.hlplots.Tagger
            Instance of the puma.hlplots.Tagger class, containing tagger information.

        Raises
        ------
        KeyError
            if model name duplicated
        """
        if str(tagger) in self.taggers:
            raise KeyError(f"{tagger} was already added.")
        self.taggers[str(tagger)] = tagger

    def add_taggers_from_file(  # pylint: disable=R0913
        self,
        taggers: list[Tagger],
        file_path: Path | str,
        key="jets",
        label_var="HadronConeExclTruthLabelID",
        aux_key="tracks",
        cuts: Cuts | list | None = None,
        num_jets: int | None = None,
        perf_vars: dict | None = None,
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
        perf_vars : dict, optional
            Override the performance variable to use, by default None
        """
        # get a list of all variables to be loaded from the file
        if not isinstance(cuts, Cuts):
            cuts = Cuts.empty() if cuts is None else Cuts.from_list(cuts)
        var_list = [label_var]
        var_list += cuts.variables
        var_list += sum([t.cuts.variables for t in taggers if t.cuts is not None], [])
        var_list = list(set(var_list + self.perf_vars))

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
            if perf_vars is not None:
                for perf_var_array in perf_vars.values():
                    perf_var_array = perf_var_array[idx]

        # for each tagger
        for tagger in taggers:
            sel_data = data
            sel_aux_data = aux_data
            sel_perf_vars = perf_vars

            # apply tagger specific cuts
            if tagger.cuts:
                idx, sel_data = tagger.cuts(data)
                sel_aux_data = aux_data[idx]
                if perf_vars is not None:
                    for perf_var_array in sel_perf_vars.values():
                        perf_var_array = perf_var_array[idx]

            # attach data to tagger objects
            tagger.labels = np.array(sel_data[label_var], dtype=[(label_var, "i4")])
            for task in tagger.aux_tasks:
                tagger.aux_scores[task] = sel_aux_data[tagger.aux_variables[task]]
            for task in aux_labels:
                tagger.aux_labels[task] = sel_aux_data[aux_labels[task]]
            if perf_vars is None:
                tagger.perf_vars = dict()
                for perf_var in self.perf_vars:
                    if any(x in perf_var for x in ["pt", "mass"]):
                        tagger.perf_vars[perf_var] = sel_data[perf_var] * 0.001
                    else:
                        tagger.perf_vars[perf_var] = sel_data[perf_var]
            else:
                tagger.perf_vars = sel_perf_vars

            # add tagger to results
            self.add(tagger)

    def __getitem__(self, tagger_name: str):
        """Retrieve Tagger object.

        Parameters
        ----------
        tagger_name : str
            Name of model

        Returns
        -------
        Tagger
            Instance of the puma.hlplots.Tagger class, containing tagger information.
        """
        return self.taggers[tagger_name]

    def get_filename(self, plot_name: str, suffix: str | None = None):
        """Get output name.

        Parameters
        ----------
        plot_name : str
            plot name
        suffix : str, optional
            suffix to add to output name, by default None

        Returns
        -------
        str
            output name
        """
        base = f"{self.sample}_{plot_name}"
        if suffix is not None:
            base += f"_{suffix}"
        return Path(self.output_dir / base).with_suffix(f".{self.extension}")

    def plot_var_vtx_perf(
        self,
        flavour: Flavour | str = None,
        suffix: str | None = None,
        xlabel: str = r"$p_{T}$ [GeV]",
        perf_var: str = "pt",
        incl_vertexing: bool = False,
        **kwargs,
    ):
        if isinstance(flavour, str):
            flavour = Flavours[flavour]

        vtx_string = "Inclusive vertexing" if incl_vertexing else "Exclusive vertexing"

        if self.atlas_second_tag:
            second_tag = self.atlas_second_tag + ", " + vtx_string
        else:
            second_tag = vtx_string

        # define the curves
        plot_vtx_eff = VarVsAuxPlot(
            mode="efficiency",
            ylabel=r"$n_{vtx}^{match}/n_{vtx}^{true}$",
            xlabel=xlabel,
            logy=False,
            atlas_first_tag=self.atlas_first_tag,
            atlas_second_tag=second_tag,
            y_scale=1.4,
        )
        plot_vtx_purity = VarVsAuxPlot(
            mode="purity",
            ylabel=r"$n_{vtx}^{match}/n_{vtx}^{reco}$",
            xlabel=xlabel,
            logy=False,
            atlas_first_tag=self.atlas_first_tag,
            atlas_second_tag=second_tag,
            y_scale=1.4,
        )
        plot_vtx_nreco = VarVsAuxPlot(
            mode="total_reco",
            ylabel=r"$n_{vtx}^{reco}$",
            xlabel=xlabel,
            logy=False,
            atlas_first_tag=self.atlas_first_tag,
            atlas_second_tag=second_tag,
            y_scale=1.4,
        )
        plot_vtx_trk_eff = VarVsAuxPlot(
            mode="efficiency",
            ylabel=r"$n_{trk}^{match}/n_{trk}^{true}$",
            xlabel=xlabel,
            logy=False,
            atlas_first_tag=self.atlas_first_tag,
            atlas_second_tag=second_tag,
            y_scale=1.4,
        )
        plot_vtx_trk_purity = VarVsAuxPlot(
            mode="purity",
            ylabel=r"$n_{trk}^{match}/n_{trk}^{reco}$",
            xlabel=xlabel,
            logy=False,
            atlas_first_tag=self.atlas_first_tag,
            atlas_second_tag=second_tag,
            y_scale=1.4,
        )

        for tagger in self.taggers.values():
            if "vertexing" not in tagger.aux_tasks:
                logger.warning(
                    f"{tagger.name} does not have vertexing aux task defined. Skipping."
                )
            assert (
                perf_var in tagger.perf_vars
            ), f"{perf_var} not in tagger {tagger.name} data!"

            # get cleaned vertex indices
            truth_indices, reco_indices = tagger.vertex_indices(
                incl_vertexing=incl_vertexing
            )

            # calculate vertexing metrics
            vtx_metrics = calculate_vertex_metrics(reco_indices, truth_indices)

            # filter out jets of chosen flavour - if flavour is not set, use all
            if flavour:
                is_flavour = tagger.is_flav(flavour)
                prefix = f"{flavour}_"
            else:
                is_flavour = np.ones_like(tagger.labels, dtype=bool)
                prefix = "alljets_"

            include_sum = vtx_metrics["track_overlap"][is_flavour] >= 0

            vtx_perf = VarVsAux(
                x_var=tagger.perf_vars[perf_var][is_flavour],
                n_match=vtx_metrics["n_match"][is_flavour],
                n_true=vtx_metrics["n_ref"][is_flavour],
                n_reco=vtx_metrics["n_test"][is_flavour],
                label=tagger.label,
                colour=tagger.colour,
                **kwargs,
            )
            vtx_trk_perf = VarVsAux(
                x_var=tagger.perf_vars[perf_var][is_flavour],
                n_match=np.sum(
                    vtx_metrics["track_overlap"][is_flavour],
                    axis=1,
                    where=include_sum,
                ),
                n_true=np.sum(
                    vtx_metrics["ref_vertex_size"][is_flavour],
                    axis=1,
                    where=include_sum,
                ),
                n_reco=np.sum(
                    vtx_metrics["test_vertex_size"][is_flavour],
                    axis=1,
                    where=include_sum,
                ),
                label=tagger.label,
                colour=tagger.colour,
                **kwargs,
            )

            plot_vtx_eff.add(vtx_perf, reference=tagger.reference)
            plot_vtx_purity.add(vtx_perf, reference=tagger.reference)
            plot_vtx_nreco.add(vtx_perf, reference=tagger.reference)
            plot_vtx_trk_eff.add(vtx_trk_perf, reference=tagger.reference)
            plot_vtx_trk_purity.add(vtx_trk_perf, reference=tagger.reference)

        if not plot_vtx_eff:
            raise ValueError("No taggers with vertexing aux task added.")

        plot_vtx_eff.draw()
        plot_vtx_eff.savefig(
            self.get_filename(prefix + f"vtx_eff_vs_{perf_var}", suffix)
        )

        plot_vtx_purity.draw()
        plot_vtx_purity.savefig(
            self.get_filename(prefix + f"vtx_purity_vs_{perf_var}", suffix)
        )

        plot_vtx_nreco.draw()
        plot_vtx_nreco.savefig(
            self.get_filename(prefix + f"vtx_nreco_vs_{perf_var}", suffix)
        )

        plot_vtx_trk_eff.draw()
        plot_vtx_trk_eff.savefig(
            self.get_filename(prefix + f"vtx_trk_eff_vs_{perf_var}", suffix)
        )

        plot_vtx_trk_purity.draw()
        plot_vtx_trk_purity.savefig(
            self.get_filename(prefix + f"vtx_trk_purity_vs_{perf_var}", suffix)
        )
