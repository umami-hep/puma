"""Results module for high level API."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ftag import Cuts, Flavour, Flavours
from ftag.hdf5 import H5Reader

import puma.fraction_scan as fraction_scan
from puma import (
    Histogram,
    HistogramPlot,
    Line2D,
    Line2DPlot,
    Roc,
    RocPlot,
    VarVsEff,
    VarVsEffPlot,
)
from puma.metrics import calc_eff, calc_rej
from puma.utils import get_good_linestyles


@dataclass
class Results:
    """Store information about several taggers and plot results."""

    signal: Flavour | str
    sample: str
    backgrounds: list = field(init=False)
    atlas_first_tag: str = "Simulation Internal"
    atlas_second_tag: str = None
    taggers: dict = field(default_factory=dict)
    sig_eff: float = None
    perf_var: str = "pt"
    output_dir: str | Path = "."
    extension: str = "png"

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

        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

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
        reader = H5Reader(file_path, precision="full")
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

    def get_filename(self, plot_name: str, suffix: str = None):
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
        base = f"{self.sample}_{self.signal}_{plot_name}"
        if suffix is not None:
            base += f"_{suffix}"
        return Path(self.output_dir / base).with_suffix(f".{self.extension}")

    def plot_probs(
        self,
        suffix: str = None,
        **kwargs,
    ):
        """Plot probability distributions.

        Parameters
        ----------
        suffix : str, optional
            Suffix to add to output file name, by default None
        **kwargs : kwargs
            key word arguments for `puma.HistogramPlot`
        """
        line_styles = get_good_linestyles()
        flavours = self.backgrounds + [self.signal]

        # group by output probability
        for flav_prob in flavours:
            histo = HistogramPlot(
                n_ratio_panels=1,
                xlabel=flav_prob.px,
                ylabel="Normalised number of jets",
                figsize=(7.0, 4.5),
                atlas_first_tag=self.atlas_first_tag,
                atlas_second_tag=self.atlas_second_tag,
                **kwargs,
            )

            tagger_labels = []
            for i, tagger in enumerate(self.taggers.values()):
                tagger_labels.append(tagger.label if tagger.label else tagger.name)
                for flav_class in flavours:
                    histo.add(
                        Histogram(
                            tagger.probs(flav_prob, flav_class),
                            ratio_group=flav_class,
                            label=flav_class.label if i == 0 else None,
                            colour=flav_class.colour,
                            linestyle=line_styles[i],
                        ),
                        reference=tagger.reference,
                    )

            histo.draw()
            histo.make_linestyle_legend(
                linestyles=line_styles,
                labels=tagger_labels,
                bbox_to_anchor=(0.55, 1),
            )
            histo.savefig(self.get_filename(f"probs_{flav_prob.px}", suffix))

        # group by flavour
        for flav_class in flavours:
            histo = HistogramPlot(
                n_ratio_panels=1,
                xlabel=flav_class.label,
                ylabel="Normalised number of jets",
                figsize=(7.0, 4.5),
                atlas_first_tag=self.atlas_first_tag,
                atlas_second_tag=self.atlas_second_tag,
                **kwargs,
            )

            tagger_labels = []
            for i, tagger in enumerate(self.taggers.values()):
                tagger_labels.append(tagger.label if tagger.label else tagger.name)
                for flav_prob in flavours:
                    histo.add(
                        Histogram(
                            tagger.probs(flav_prob, flav_class),
                            ratio_group=flav_prob,
                            label=flav_prob.px if i == 0 else None,
                            colour=flav_prob.colour,
                            linestyle=line_styles[i],
                        ),
                        reference=tagger.reference,
                    )

            histo.draw()
            histo.make_linestyle_legend(
                linestyles=line_styles,
                labels=tagger_labels,
                bbox_to_anchor=(0.55, 1),
            )
            histo.savefig(self.get_filename(f"probs_{flav_class}", suffix))

    def plot_discs(
        self,
        suffix: str = None,
        exclude_tagger: list = None,
        xlabel: str = None,
        wp_vlines: list = None,
        **kwargs,
    ):
        """Plot discriminant distributions.

        Parameters
        ----------
        suffix : str, optional
            Suffix to add to output file name, by default None
        exclude_tagger : list, optional
            List of taggers to be excluded from this plot, by default None
        xlabel : str, optional
            x-axis label, by default "$D_{b}$"
        wp_vlines : list, optional
            List of WPs to draw vertical lines at, by default None
        **kwargs : kwargs
            key word arguments for `puma.HistogramPlot`
        """
        if xlabel is None:
            xlabel = rf"$D_{{{self.signal.name.rstrip('jets')}}}$"
        if wp_vlines is None:
            wp_vlines = []

        line_styles = get_good_linestyles()

        histo = HistogramPlot(
            n_ratio_panels=0,
            xlabel=xlabel,
            ylabel="Normalised number of jets",
            figsize=(7.0, 4.5),
            atlas_first_tag=self.atlas_first_tag,
            atlas_second_tag=self.atlas_second_tag,
            **kwargs,
        )

        tagger_labels = []
        for i, tagger in enumerate(self.taggers.values()):
            if exclude_tagger is not None and tagger.name in exclude_tagger:
                continue
            discs = tagger.discriminant(self.signal)

            # get working point
            for wp in wp_vlines:
                cut = np.percentile(discs[tagger.is_flav(self.signal)], 100 - wp)
                label = None if i > 0 else f"{wp}%"
                histo.draw_vlines([cut], labels=[label], linestyle=line_styles[i])

            for flav in self.flavours:
                histo.add(
                    Histogram(
                        discs[tagger.is_flav(flav)],
                        ratio_group=flav,
                        label=flav.label if i == 0 else None,
                        colour=flav.colour,
                        linestyle=line_styles[i],
                    ),
                    reference=tagger.reference,
                )
            tagger_labels.append(tagger.label if tagger.label else tagger.name)
        histo.draw()
        histo.make_linestyle_legend(
            linestyles=line_styles,
            labels=tagger_labels,
            bbox_to_anchor=(0.55, 1),
        )
        histo.savefig(self.get_filename("disc", suffix))

    def plot_rocs(
        self,
        suffix: str = None,
        args_roc_plot: dict = None,
    ):
        """Plots rocs.

        Parameters
        ----------
        suffix : str, optional
            suffix to add to output file name, by default None
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
            discs = tagger.discriminant(self.signal)
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
        plot_name = self.get_filename("roc", suffix)
        plot_roc.savefig(plot_name)

    def plot_var_perf(  # pylint: disable=too-many-locals
        self,
        suffix: str = None,
        xlabel: str = r"$p_{T}$ [GeV]",
        h_line: float = None,
        **kwargs,
    ):
        """Variable vs efficiency/rejection plot.

        You can choose between different modes: "sig_eff", "bkg_eff", "sig_rej",
        "bkg_rej"

        Parameters
        ----------
        suffix : str, optional
            suffix to add to output file name, by default None
        xlabel : regexp, optional
            _description_, by default "$p_{T}$ [GeV]"
        h_line : float, optional
            draws a horizonatal line in the signal efficiency plot
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

            discs = tagger.discriminant(self.signal)
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

        plot_base = "profile_flat" if kwargs.get("fixed_eff_bin") else "profile_fixed"
        plot_suffix = f"{self.signal}_eff_{suffix}" if suffix else f"{self.signal}_eff"
        plot_sig_eff.savefig(self.get_filename(plot_base, plot_suffix))
        for i, background in enumerate(self.backgrounds):
            plot_bkg[i].draw()
            plot_suffix = (
                f"{background}_rej_{suffix}" if suffix else f"{background}_rej"
            )
            plot_bkg[i].savefig(self.get_filename(plot_base, plot_suffix))

    def plot_fraction_scans(
        self, suffix: str = None, efficiency: float = 0.7, rej: bool = False
    ):
        """Produce fraction scan (fc/fb) iso-efficiency plots.

        Parameters
        ----------
        suffix : str, optional
            suffix to add to output file name, by default None
        efficiency : float, optional
            signal efficiency, by default 0.7
        rej : bool, optional
            if True, plot rejection instead of efficiency, by default False
        """
        if self.signal not in (Flavours.bjets, Flavours.cjets):
            raise ValueError("Signal flavour must be bjets or cjets")
        if len(self.backgrounds) != 2:
            raise ValueError("Only two background flavours are supported")

        fxs = fraction_scan.get_fx_values()
        plot = Line2DPlot(atlas_second_tag=self.atlas_second_tag)
        eff_or_rej = calc_eff if not rej else calc_rej

        for tagger in self.taggers.values():
            xs = []
            ys = []
            for fx in fxs:
                sig_idx = tagger.is_flav(self.signal)
                disc = tagger.discriminant(self.signal, fx=fx)
                bkg_idx = tagger.is_flav(self.backgrounds[0])
                xs.append(eff_or_rej(disc[sig_idx], disc[bkg_idx], efficiency))
                bkg_idx = tagger.is_flav(self.backgrounds[1])
                ys.append(eff_or_rej(disc[sig_idx], disc[bkg_idx], efficiency))

            # add curve for this tagger
            tagger_fx = tagger.f_c if self.signal == Flavours.bjets else tagger.f_b
            plot.add(
                Line2D(
                    x_values=xs,
                    y_values=ys,
                    label=f"{tagger.label} ($f_x={tagger_fx}$)",
                    colour=tagger.colour,
                )
            )

            # Add a marker for the just added fraction scan
            # The is_marker bool tells the plot that this is a marker and not a line
            fx_idx = np.argmin(np.abs(fxs - tagger_fx))
            plot.add(
                Line2D(
                    x_values=xs[fx_idx],
                    y_values=ys[fx_idx],
                    marker="x",
                    markersize=15,
                    markeredgewidth=2,
                ),
                is_marker=True,
            )

            # Adding labels
            if not rej:
                plot.xlabel = self.backgrounds[0].eff_str
                plot.ylabel = self.backgrounds[1].eff_str
            else:
                plot.xlabel = self.backgrounds[0].rej_str
                plot.ylabel = self.backgrounds[1].rej_str

            # Draw and save the plot
            plot.draw()
            plot.savefig(self.get_filename("fraction_scan", suffix))
