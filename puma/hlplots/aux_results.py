"""Auxiliary task results module for high level API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ftag import Cuts, Flavours, Label
from ftag.hdf5 import H5Reader

from puma import Histogram, HistogramPlot
from puma.hlplots.tagger import Tagger
from puma.matshow import MatshowPlot
from puma.utils import get_good_linestyles, logger
from puma.utils.aux import get_aux_labels, get_trackOrigin_classNames
from puma.utils.confusion_matrix import confusion_matrix
from puma.utils.mass import calculate_vertex_mass
from puma.utils.precision_recall_scores import precision_recall_scores_per_class
from puma.utils.vertexing import calculate_vertex_metrics
from puma.var_vs_vtx import VarVsVtx, VarVsVtxPlot


@dataclass
class AuxResults:
    """Store information about several taggers and plot auxiliary task results."""

    sample: str
    atlas_first_tag: str = "Simulation Internal"
    atlas_second_tag: str = None
    atlas_third_tag: str = None
    taggers: dict = field(default_factory=dict)
    perf_vars: str | tuple | list = "pt"
    aux_perf_vars: str | tuple | list = None
    output_dir: str | Path = "."
    extension: str = "pdf"
    global_cuts: Cuts | list | None = None
    num_jets: int | None = None
    remove_nan: bool = False

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(self.perf_vars, str):
            self.perf_vars = [self.perf_vars]
        if self.atlas_second_tag is not None and self.atlas_third_tag is not None:
            self.atlas_second_tag = f"{self.atlas_second_tag}\n{self.atlas_third_tag}"

        self.plot_funcs = {
            "vertexing": self.plot_var_vtx_perf,
            "track_origin": self.plot_track_origin_confmat,
        }

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

    def load_taggers_from_file(  # pylint: disable=R0913
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

        def check_nan(data: np.ndarray) -> np.ndarray:
            """
            Filter out NaN values from loaded data.

            Parameters
            ----------
            data : ndarray
                Data to filter
            """
            mask = np.ones(len(data), dtype=bool)
            for name in data.dtype.names:
                mask = np.logical_and(mask, ~np.isnan(data[name]))
            if np.sum(~mask) > 0:
                if self.remove_nan:
                    logger.warning(
                        f"{np.sum(~mask)} NaN values found in loaded data. Removing" " them."
                    )
                    return data[mask]
                raise ValueError(f"{np.sum(~mask)} NaN values found in loaded data.")
            return data

        # set tagger output nodes
        for tagger in taggers:
            if tagger not in self.taggers.values():
                self.add(tagger)

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
        if self.aux_perf_vars is not None:
            aux_var_list += self.aux_perf_vars
        aux_var_list = list(set(aux_var_list))

        # load data
        reader = H5Reader(file_path, precision="full", shuffle=False)
        data = reader.load({key: var_list}, num_jets)[key]
        aux_reader = H5Reader(file_path, precision="full", jets_name=aux_key, shuffle=False)
        aux_data = aux_reader.load({aux_key: aux_var_list}, num_jets)[aux_key]

        # check for nan values
        data = check_nan(data)
        # apply common cuts
        if cuts:
            idx, data = cuts(data)
            aux_data = aux_data[idx]
            if perf_vars is not None:
                for name, array in perf_vars.items():
                    perf_vars[name] = array[idx]

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
                    for name, array in sel_perf_vars.items():
                        sel_perf_vars[name] = array[idx]

            # attach data to tagger objects
            tagger.labels = np.array(sel_data[label_var], dtype=[(label_var, "i4")])
            for task in tagger.aux_tasks:
                tagger.aux_scores[task] = sel_aux_data[tagger.aux_variables[task]]
            for task in aux_labels:
                tagger.aux_labels[task] = sel_aux_data[aux_labels[task]]
            if perf_vars is None:
                tagger.perf_vars = {}
                for perf_var in self.perf_vars:
                    if any(x in perf_var for x in ["pt", "mass"]):
                        tagger.perf_vars[perf_var] = sel_data[perf_var] * 0.001
                    else:
                        tagger.perf_vars[perf_var] = sel_data[perf_var]
            else:
                tagger.perf_vars = sel_perf_vars
            tagger.aux_perf_vars = {}
            if self.aux_perf_vars is not None:
                for aux_perf_var in self.aux_perf_vars:
                    if any(x in aux_perf_var for x in ["pt"]):
                        tagger.aux_perf_vars[aux_perf_var] = sel_aux_data[aux_perf_var] * 0.001
                    elif "deta" in aux_perf_var and "eta" in self.perf_vars:
                        tagger.aux_perf_vars["eta"] = sel_aux_data["deta"] + sel_data[
                            "eta"
                        ].reshape(-1, 1)
                    else:
                        tagger.aux_perf_vars[aux_perf_var] = sel_aux_data[aux_perf_var]

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
        vtx_flavours: list[Label] | list[str] | None = None,
        no_vtx_flavours: list[Label] | list[str] | None = None,
        suffix: str | None = None,
        xlabel: str = r"$p_{T}$ [GeV]",
        perf_var: str = "pt",
        incl_vertexing: bool = False,
        vertex_match_requirement: dict | None = None,
        **kwargs,
    ):
        # Get good linestyles for plotting
        line_styles = get_good_linestyles()

        if vtx_flavours is None and no_vtx_flavours is None:
            raise ValueError(
                "Need to specify either vtx_flavours or no_vtx_flavours (or both) to make plots."
            )
        if vtx_flavours is None:
            vtx_flavours = []
        elif no_vtx_flavours is None:
            no_vtx_flavours = []

        if incl_vertexing:
            suffix = "incl" if not suffix else f"{suffix}_incl"

        if vertex_match_requirement is None:
            vertex_match_requirement = {"eff_req": 0.65, "purity_req": 0.5}
            # if you want to make the track level recall and purity plots,
            # don't require any quality requirements
            # vertex_match_requirement = {"eff_req": 0, "purity_req": 0}
        eff_req = round(vertex_match_requirement["eff_req"] * 100, 1)
        purity_req = round(vertex_match_requirement["purity_req"] * 100, 1)
        vtx_match_str = rf"Recall $\geq {eff_req}\%$, Purity $\geq {purity_req}\%$"
        vtx_string = "\nInclusive vertexing" if incl_vertexing else "\nExclusive vertexing"
        atlas_second_tag = self.atlas_second_tag if self.atlas_second_tag else ""
        atlas_second_tag += vtx_string

        # calculate vertexing information for each tagger
        vtx_metrics = {}
        for tagger in self.taggers.values():
            if "vertexing" not in tagger.aux_tasks:
                logger.warning(
                    f"{tagger.label} does not have vertexing aux task defined. Skipping."
                )
            assert perf_var in tagger.perf_vars, f"{perf_var} not in tagger {tagger.label} data!"

            # get cleaned vertex indices and calculate vertexing metrics
            truth_indices, reco_indices = tagger.vertex_indices(incl_vertexing=incl_vertexing)
            max_vertices = 20 if not incl_vertexing else 1
            vtx_metrics[tagger.label] = calculate_vertex_metrics(
                reco_indices, truth_indices, max_vertices=max_vertices, **vertex_match_requirement
            )

        if not vtx_metrics:
            raise ValueError("No taggers with vertexing aux task added.")

        # make plots for flavours where vertices are expected
        for flavour in vtx_flavours:
            if isinstance(flavour, str):
                flav = Flavours[flavour]

            # $n_{vtx}^{match}/n_{vtx}^{true}$
            plot_vtx_eff = VarVsVtxPlot(
                mode="efficiency",
                ylabel="Efficiency",
                xlabel=xlabel,
                logy=False,
                atlas_first_tag=self.atlas_first_tag,
                atlas_second_tag=atlas_second_tag + f", {flav.label.lower()}\n{vtx_match_str}",
                y_scale=1.4,
            )
            # $n_{vtx}^{match}/n_{vtx}^{reco}$
            plot_vtx_purity = VarVsVtxPlot(
                mode="purity",
                ylabel="Purity",
                xlabel=xlabel,
                logy=False,
                atlas_first_tag=self.atlas_first_tag,
                atlas_second_tag=atlas_second_tag + f", {flav.label.lower()}\n{vtx_match_str}",
                y_scale=1.4,
            )
            # $n_{trk}^{match}/n_{trk}^{true}$
            plot_vtx_trk_eff = VarVsVtxPlot(
                mode="efficiency",
                ylabel="Track assignment efficiency",
                xlabel=xlabel,
                logy=False,
                atlas_first_tag=self.atlas_first_tag,
                atlas_second_tag=atlas_second_tag + f", {flav.label.lower()}",
                y_scale=1.4,
            )
            # $n_{trk}^{match}/n_{trk}^{reco}$
            plot_vtx_trk_purity = VarVsVtxPlot(
                mode="purity",
                ylabel="Track assignment purity",
                xlabel=xlabel,
                logy=False,
                atlas_first_tag=self.atlas_first_tag,
                atlas_second_tag=atlas_second_tag + f", {flav.label.lower()}",
                y_scale=1.4,
            )

            for counter, tagger in enumerate(self.taggers.values()):
                if tagger.label not in vtx_metrics:
                    continue
                is_flavour = tagger.is_flav(flav)
                include_sum = vtx_metrics[tagger.label]["track_overlap"][is_flavour] >= 0

                vtx_perf = VarVsVtx(
                    x_var=tagger.perf_vars[perf_var][is_flavour],
                    n_match=vtx_metrics[tagger.label]["n_match"][is_flavour],
                    n_true=vtx_metrics[tagger.label]["n_ref"][is_flavour],
                    n_reco=vtx_metrics[tagger.label]["n_test"][is_flavour],
                    label=tagger.label,
                    colour=tagger.colour,
                    linestyle=line_styles[counter],
                    **kwargs,
                )
                vtx_trk_perf = VarVsVtx(
                    x_var=tagger.perf_vars[perf_var][is_flavour],
                    n_match=np.sum(
                        vtx_metrics[tagger.label]["track_overlap"][is_flavour],
                        axis=1,
                        where=include_sum,
                    ),
                    n_true=np.sum(
                        vtx_metrics[tagger.label]["ref_vertex_size"][is_flavour],
                        axis=1,
                        where=include_sum,
                    ),
                    n_reco=np.sum(
                        vtx_metrics[tagger.label]["test_vertex_size"][is_flavour],
                        axis=1,
                        where=include_sum,
                    ),
                    label=tagger.label,
                    colour=tagger.colour,
                    linestyle=line_styles[counter],
                    **kwargs,
                )

                plot_vtx_eff.add(vtx_perf, reference=tagger.reference)
                plot_vtx_purity.add(vtx_perf, reference=tagger.reference)
                plot_vtx_trk_eff.add(vtx_trk_perf, reference=tagger.reference)
                plot_vtx_trk_purity.add(vtx_trk_perf, reference=tagger.reference)

            plot_vtx_eff.draw()
            plot_vtx_eff.savefig(self.get_filename(f"{flav}_vtx_eff_vs_{perf_var}", suffix))

            plot_vtx_purity.draw()
            plot_vtx_purity.savefig(self.get_filename(f"{flav}_vtx_purity_vs_{perf_var}", suffix))

            plot_vtx_trk_eff.draw()
            plot_vtx_trk_eff.savefig(self.get_filename(f"{flav}_vtx_trk_eff_vs_{perf_var}", suffix))

            plot_vtx_trk_purity.draw()
            plot_vtx_trk_purity.savefig(
                self.get_filename(f"{flav}_vtx_trk_purity_vs_{perf_var}", suffix)
            )

        # make plots for flavours where vertices are not expected
        for flavour in no_vtx_flavours:
            if isinstance(flavour, str):
                flav = Flavours[flavour]

            plot_vtx_fakes = VarVsVtxPlot(
                mode="fakes",
                ylabel="Vertex rate",
                xlabel=xlabel,
                logy=False,
                atlas_first_tag=self.atlas_first_tag,
                atlas_second_tag=atlas_second_tag + f", {flav.label.lower()}",
                y_scale=1.4,
            )

            for counter, tagger in enumerate(self.taggers.values()):
                if tagger.label not in vtx_metrics:
                    continue
                is_flavour = tagger.is_flav(flav)

                vtx_perf = VarVsVtx(
                    x_var=tagger.perf_vars[perf_var][is_flavour],
                    n_match=vtx_metrics[tagger.label]["n_match"][is_flavour],
                    n_true=vtx_metrics[tagger.label]["n_ref"][is_flavour],
                    n_reco=vtx_metrics[tagger.label]["n_test"][is_flavour],
                    label=tagger.label,
                    colour=tagger.colour,
                    linestyle=line_styles[counter],
                    **kwargs,
                )

                plot_vtx_fakes.add(vtx_perf, reference=tagger.reference)

            plot_vtx_fakes.draw()
            plot_vtx_fakes.savefig(self.get_filename(f"{flav}_vtx_fakes_vs_{perf_var}", suffix))

    def plot_vertex_mass(
        self,
        vtx_flavours: list[Label] | list[str],
        incl_vertexing: bool = True,
        mass_range: tuple = (0, 5),
        **kwargs,
    ):
        """Plot vertex mass for each tagger.

        Parameters
        ----------
        vtx_flavours : list[Label] | list[str]
            List of jet flavours to make SV mass plots for
        incl_vertexing : bool, optional
            Whether to use inclusive or exclusive vertexing, by default inclusive
        mass_range: tuple, optional
            Range of the mass histogram, by default (0, 5)
        kwargs : dict
            Keyword arguments for `puma.HistogramPlot`
        """
        if incl_vertexing:
            vertexing_text = "Inclusive"
            suffix = "incl"
        else:
            vertexing_text = "Exclusive"
            suffix = "excl"

        atlas_second_tag = "" if self.atlas_second_tag is None else self.atlas_second_tag

        for flavour in vtx_flavours:
            if isinstance(flavour, str):
                flav = Flavours[flavour]

            mass_plot = HistogramPlot(
                bins_range=mass_range,
                xlabel="$m_{SV}$ [GeV]",
                ylabel="Normalised number of vertices",
                atlas_first_tag=self.atlas_first_tag,
                atlas_second_tag=atlas_second_tag
                + f"\n{vertexing_text} vertexing, {flav.label.lower()}",
                y_scale=1.7,
                n_ratio_panels=1,
                ymin_ratio=[0.8],
                ymax_ratio=[1.4],
                **kwargs,
            )

            if incl_vertexing:
                mass_diff_plot = HistogramPlot(
                    bins=np.linspace(-mass_range[1] / 2, mass_range[1] / 2, 12),
                    xlabel=r"$\Delta m_{SV}$ [GeV] (reco - truth)",
                    ylabel="Normalised number of vertices",
                    atlas_first_tag=self.atlas_first_tag,
                    atlas_second_tag=atlas_second_tag
                    + f"\n{vertexing_text} vertexing, {flav.label.lower()}",
                    y_scale=1.4,
                    **kwargs,
                )

            for i, tagger in enumerate(self.taggers.values()):
                if "vertexing" not in tagger.aux_tasks:
                    logger.warning(
                        f"{tagger.label} does not have vertexing aux task defined. Skipping."
                    )

                assert {"pt", "eta", "dphi"}.issubset(
                    set(tagger.aux_perf_vars.keys())
                ), "Track pt, eta or dphi not in tagger.aux_perf_vars (required to calculate \
                    vertex masses). Track eta is automatically calculated if deta and jet eta \
                    are supplied."

                # get cleaned vertex indices and calculate vertexing metrics
                truth_indices, reco_indices = tagger.vertex_indices(incl_vertexing=incl_vertexing)
                is_flavour = tagger.is_flav(flav)

                if i == 0:
                    masses = calculate_vertex_mass(
                        tagger.aux_perf_vars["pt"][is_flavour],
                        tagger.aux_perf_vars["eta"][is_flavour],
                        tagger.aux_perf_vars["dphi"][is_flavour],
                        truth_indices[is_flavour],
                        particle_mass=0.13957,  # pion mass in GeV
                    )

                    if incl_vertexing:
                        truth_masses = np.max(masses, axis=1)
                    else:
                        truth_masses = np.unique(masses, axis=1).flatten()

                    mass_plot.add(
                        Histogram(
                            truth_masses[truth_masses > 0.14],
                            label="MC truth",
                            colour="#000000",
                        ),
                        reference=True,
                    )

                masses = calculate_vertex_mass(
                    tagger.aux_perf_vars["pt"][is_flavour],
                    tagger.aux_perf_vars["eta"][is_flavour],
                    tagger.aux_perf_vars["dphi"][is_flavour],
                    reco_indices[is_flavour],
                    particle_mass=0.13957,  # pion mass in GeV
                )

                if incl_vertexing:
                    sv_masses = np.max(masses, axis=1)
                    mass_diffs = sv_masses - truth_masses
                    mass_diffs = mass_diffs[np.logical_and(truth_masses > 0.14, sv_masses > 0.14)]
                    mass_diff_plot.add(
                        Histogram(mass_diffs, label=tagger.label, colour=tagger.colour)
                    )
                else:
                    sv_masses = np.concatenate([np.unique(imass) for imass in masses])

                sv_masses = sv_masses[sv_masses > 0.14]  # remove single and zero track vertices
                mass_plot.add(Histogram(sv_masses, label=tagger.label, colour=tagger.colour))

            mass_plot.draw()
            mass_plot.savefig(self.get_filename(f"{flav}_sv_mass_{suffix}"))

            if incl_vertexing:
                mass_diff_plot.draw()
                mass_diff_plot.savefig(self.get_filename(f"{flav}_sv_mass_diff"))

    def plot_track_origin_confmat(
        self,
        normalize: str | None = "rownorm",
        minimal_plot: bool = True,
        **kwargs,
    ):
        """Plot Track Origin Aux Task confusion matrix.

        Parameters
        ----------
        normalize : str | None, optional
            Normalization of the confusion matrix. Can be:
            None: Give raw counts;
            "rownorm": Normalize across the prediction class, i.e. such that the rows add to one;
            "colnorm": Normalize across the target class, i.e. such that the columns add to one;
            "all" : Normalize across all examples, i.e. such that all matrix entries add to one.
            Defaults to "rownorm".
        minimal_plot : bool, optional
            Whether to plot the CM with minimal or full info (title, cbar), by default True
        **kwargs : kwargs
            Keyword arguments for `puma.MatshowPlot` and `puma.PlotObject`
        """
        for tagger in self.taggers.values():
            # Reading tagger's target and predicted labels
            # and flattening them so that they have shape (Ntracks,)
            target = tagger.aux_labels["track_origin"].reshape(-1)
            predictions = tagger.aux_scores["track_origin"].reshape(-1)

            padding_removal = target >= 0

            target = target[padding_removal]
            predictions = predictions[padding_removal]

            # Computing the confusion matrix
            cm = confusion_matrix(target, predictions, normalize=normalize)
            precision, recall = precision_recall_scores_per_class(target, predictions)

            class_names = get_trackOrigin_classNames()
            class_names_with_perf = []

            if minimal_plot:
                for i, c in enumerate(class_names):
                    # class_names_with_perf.append(f"{c}\nRecall = {recall[i]:.3f}")
                    class_names_with_perf.append(
                        f"{c}\nPrecision = {precision[i]:.3f}\nRecall = {recall[i]:.3f}"
                    )
                # Plotting the confusion matrix
                plot_cm = MatshowPlot(
                    x_ticklabels=class_names,
                    y_ticklabels=class_names_with_perf,
                    xlabel="Predicted Classes",
                    ylabel="Target Classes",
                    atlas_first_tag=self.atlas_first_tag,
                    atlas_second_tag=self.atlas_second_tag,
                    show_cbar=False,
                    atlas_tag_outside=True,
                    **kwargs,
                )
            else:
                for i, c in enumerate(class_names):
                    class_names_with_perf.append(
                        f"{c}\nPrecision = {precision[i]:.3f}\nRecall = {recall[i]:.3f}"
                    )
                # Plotting the confusion matrix
                plot_cm = MatshowPlot(
                    x_ticklabels=class_names,
                    y_ticklabels=class_names_with_perf,
                    title="Track Origin Auxiliary Task\nConfusion Matrix",
                    xlabel="Predicted Classes",
                    ylabel="Target Classes",
                    atlas_first_tag=self.atlas_first_tag,
                    atlas_second_tag=self.atlas_second_tag,
                    atlas_tag_outside=True,
                    show_cbar=False,
                    **kwargs,
                )
            plot_cm.draw(cm)
            base = tagger.name + "_trackOrigin_cm"
            plot_cm.savefig(self.get_filename(base))
            print("saved file with name", self.get_filename(base))
