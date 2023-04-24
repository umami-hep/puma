"""Results module for high level API."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ftag import Cuts, Flavour, Flavours
from ftag.hdf5 import H5Reader

from puma import Histogram, HistogramPlot, Roc, RocPlot, VarVsEff, VarVsEffPlot
from puma.metrics import calc_rej
from puma.utils import get_good_linestyles


@dataclass
class Results:
    """Store information about several taggers and plot results."""

    signal: Flavour | str = Flavours.bjets
    backgrounds: list = field(init=False)
    atlas_first_tag: str = "Simulation Internal"
    atlas_second_tag: str = None
    taggers: dict = field(default_factory=dict)
    sig_eff: float = None
    perf_var: str = "pt"

    def __post_init__(self):
        if isinstance(self.signal, str):
            self.signal = Flavours[self.signal]
        if self.signal == Flavours.bjets:
            self.backgrounds = [Flavours.cjets, Flavours.ujets]
        elif self.signal == Flavours.cjets:
            self.backgrounds = [Flavours.bjets, Flavours.ujets]
        elif self.signal == Flavours.hbb:
            self.backgrounds = [Flavours.hcc, Flavours.top, Flavours.qcd]
        elif self.signal == Flavours.hcc:
            self.backgrounds = [Flavours.hbb, Flavours.top, Flavours.qcd]
        else:
            raise ValueError(f"Unsupported signal class {self.signal}.")

    @property
    def flavours(self):
        """Return a list of all flavours.

        Returns
        -------
        list
            List of all flavours
        """
        return self.backgrounds + [self.signal]

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
        if tagger.output_nodes is None:
            tagger.output_nodes = self.flavours
        self.taggers[str(tagger)] = tagger

    def add_taggers_from_file(  # pylint: disable=R0913
        self,
        taggers,
        file_path,
        key="jets",
        label_var="HadronConeExclTruthLabelID",
        cuts=None,
        num_jets=None,
        perf_var=None,
    ):
        """Add taggers from file.

        # TODO: proper cuts class implementation

        Parameters
        ----------
        taggers : list
            List of taggers to add
        file_path : str
            Path to file
        key : str, optional
            Key in file, by default 'jets'
        label_var : str
            Label variable to use
        cuts : Cuts | list
            List of cuts to apply
        num_jets : int, optional
            Number of jets to load from the file, by default all jets
        perf_var : np.ndarray, optional
            Override the performance variable to use, by default None
        """
        # set tagger output nodes
        for tagger in taggers:
            if tagger.output_nodes is None:
                tagger.output_nodes = self.flavours

        # get a list of all variables to be loaded from the file
        if not isinstance(cuts, Cuts):
            cuts = Cuts.empty() if cuts is None else Cuts.from_list(cuts)
        var_list = sum([tagger.variables for tagger in taggers], [label_var])
        var_list += cuts.variables
        var_list = list(set(var_list + [self.perf_var]))

        # load data
        reader = H5Reader(file_path)
        data = reader.load({key: var_list}, num_jets)[key]

        # apply cuts
        idx, data = cuts(data)
        if perf_var is not None:
            perf_var = perf_var[idx]

        # attach data to tagger objects
        for tagger in taggers:
            tagger.extract_tagger_scores(data, source_type="structured_array")
            tagger.labels = np.array(data[label_var], dtype=[(label_var, "i4")])
            if perf_var is None:
                tagger.perf_var = data[self.perf_var]
                if any(x in self.perf_var for x in ["pt", "mass"]):
                    tagger.perf_var = tagger.perf_var * 0.001
            else:
                tagger.perf_var = perf_var
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

    def plot_discs(
        self,
        plot_name: str,
        exclude_tagger: list = None,
        xlabel: str = None,
        **kwargs,
    ):
        """Plot tagger discriminants.

        Parameters
        ----------
        plot_name : _type_
            Name of the plot.
        exclude_tagger : list, optional
            List of taggers to be excluded from this plot, by default None
        xlabel : str, optional
            x-axis label, by default "$D_{b}$"
        **kwargs : kwargs
            key word arguments for `puma.HistogramPlot`
        """
        if xlabel is None:
            xlabel = rf"$D_{{{self.signal.name.rstrip('jets')}}}$"

        line_styles = get_good_linestyles()

        tagger_output_plot = HistogramPlot(
            n_ratio_panels=0,
            xlabel=xlabel,
            ylabel="Normalised number of jets",
            figsize=(7.0, 4.5),
            atlas_first_tag=self.atlas_first_tag,
            atlas_second_tag=self.atlas_second_tag,
            **kwargs,
        )
        tag_i = 0
        tag_labels = []
        for tagger in self.taggers.values():
            if exclude_tagger is not None and tagger.name in exclude_tagger:
                continue
            discs = tagger.get_disc(self.signal)
            for flav in self.flavours:
                tagger_output_plot.add(
                    Histogram(
                        discs[tagger.is_flav(flav)],
                        ratio_group=flav,
                        label=flav.label if tag_i == 0 else None,
                        colour=flav.colour,
                        linestyle=line_styles[tag_i],
                    ),
                    reference=tagger.reference,
                )
            tag_i += 1
            tag_labels.append(tagger.label if tagger.label else tagger.name)
        tagger_output_plot.draw()
        tagger_output_plot.make_linestyle_legend(
            linestyles=line_styles[:tag_i],
            labels=tag_labels,
            bbox_to_anchor=(0.55, 1),
        )
        tagger_output_plot.savefig(plot_name)

    def plot_rocs(
        self,
        plot_name: str,
        args_roc_plot: dict = None,
    ):
        """Plots rocs.

        Parameters
        ----------
        plot_name : puma.RocPlot
            roc plot object
        args_roc_plot: dict, optional
            key word arguments being passed to `RocPlot`
        """
        roc_plot_args = {
            "n_ratio_panels": len(self.backgrounds),
            "ylabel": "Background rejection",
            "xlabel": self.signal.eff_str,
            "atlas_first_tag": self.atlas_first_tag,
            "atlas_second_tag": self.atlas_second_tag,
            "y_scale": 1.3,
        }
        # TODO: update in python 3.9
        if args_roc_plot is not None:
            roc_plot_args.update(args_roc_plot)
        plot_roc = RocPlot(**roc_plot_args)

        for tagger in self.taggers.values():
            discs = tagger.get_disc(self.signal)
            for background in self.backgrounds:
                rej = calc_rej(
                    discs[tagger.is_flav(self.signal)],
                    discs[tagger.is_flav(background)],
                    self.sig_eff,
                )
                plot_roc.add_roc(
                    Roc(
                        self.sig_eff,
                        rej,
                        n_test=tagger.n_jets(background),
                        rej_class=background,
                        signal_class=self.signal,
                        label=tagger.label,
                        colour=tagger.colour,
                    ),
                    reference=tagger.reference,
                )

        # setting which flavour rejection ratio is drawn in which ratio panel
        for i, background in enumerate(self.backgrounds):
            plot_roc.set_ratio_class(i + 1, background)

        plot_roc.draw()
        plot_roc.savefig(plot_name)

    def plot_var_perf(  # pylint: disable=too-many-locals
        self,
        plot_name: str,
        xlabel: str = r"$p_{T}$ [GeV]",
        h_line: float = None,
        ext: str = "png",
        **kwargs,
    ):
        """Variable vs efficiency/rejection plot.

        You can choose between different modes: "sig_eff", "bkg_eff", "sig_rej",
        "bkg_rej"

        Parameters
        ----------
        plot_name : str
            plot name base
        xlabel : regexp, optional
            _description_, by default "$p_{T}$ [GeV]"
        h_line : float, optional
            draws a horizonatal line in the signal efficiency plot
        ext : str, optional
            changes the extension of the save plot
        **kwargs : kwargs
            key word arguments for `puma.VarVsEff`
        """
        # define the curves
        plot_sig_eff = VarVsEffPlot(
            mode="sig_eff",
            ylabel=self.signal.eff_str,
            xlabel=xlabel,
            logy=False,
            atlas_first_tag=self.atlas_first_tag,
            atlas_second_tag=self.atlas_second_tag,
            n_ratio_panels=1,
            y_scale=1.4,
        )
        plot_bkg = []
        for background in self.backgrounds:
            plot_bkg.append(
                VarVsEffPlot(
                    mode="bkg_rej",
                    ylabel=background.rej_str,
                    xlabel=xlabel,
                    logy=False,
                    atlas_first_tag=self.atlas_first_tag,
                    atlas_second_tag=self.atlas_second_tag,
                    n_ratio_panels=1,
                    y_scale=1.4,
                )
            )

        disc_cut_in_kwargs = "disc_cut" in kwargs
        working_point_in_kwargs = "working_point" in kwargs
        for tagger in self.taggers.values():
            if not disc_cut_in_kwargs:
                kwargs["disc_cut"] = tagger.disc_cut
            if not working_point_in_kwargs:
                kwargs["working_point"] = tagger.working_point

            discs = tagger.get_disc(self.signal)
            is_signal = tagger.is_flav(self.signal)
            plot_sig_eff.add(
                VarVsEff(
                    x_var_sig=tagger.perf_var[is_signal],
                    disc_sig=discs[is_signal],
                    label=tagger.label,
                    colour=tagger.colour,
                    **kwargs,
                ),
                reference=tagger.reference,
            )
            for i, background in enumerate(self.backgrounds):
                is_bkg = tagger.is_flav(background)
                plot_bkg[i].add(
                    VarVsEff(
                        x_var_sig=tagger.perf_var[is_signal],
                        disc_sig=discs[is_signal],
                        x_var_bkg=tagger.perf_var[is_bkg],
                        disc_bkg=discs[is_bkg],
                        label=tagger.label,
                        colour=tagger.colour,
                        **kwargs,
                    ),
                    reference=tagger.reference,
                )

        plot_sig_eff.draw()
        if h_line:
            plot_sig_eff.draw_hline(h_line)
        plot_sig_eff.savefig(f"{plot_name}_{self.signal}_eff.{ext}")
        for i, background in enumerate(self.backgrounds):
            plot_bkg[i].draw()
            plot_bkg[i].savefig(f"{plot_name}_{background}_rej.{ext}")
