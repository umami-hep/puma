"""Results module for high level API."""
import operator
from dataclasses import dataclass, field
from typing import Literal

import h5py

from puma import Histogram, HistogramPlot, Roc, RocPlot, VarVsEff, VarVsEffPlot
from puma.metrics import calc_rej
from puma.utils import get_good_linestyles, global_config

OPERATORS = {
    "==": operator.__eq__,
    ">=": operator.__ge__,
    "<=": operator.__le__,
    ">": operator.__gt__,
    "<": operator.__lt__,
}


@dataclass
class Results:
    """Store information about several taggers and plot results."""

    signal: Literal["bjets", "cjets", "Hcc"] = "bjets"
    backgrounds: list = field(init=False)
    atlas_second_tag: str = None
    taggers: dict = field(default_factory=dict)
    sig_eff: float = None
    perf_var: str = "pt"

    def __post_init__(self):
        if self.signal == "bjets":
            self.backgrounds = ["ujets", "cjets"]
        elif self.signal == "cjets":
            self.backgrounds = ["ujets", "bjets"]
        elif self.signal == "Hbb":
            self.backgrounds = ["Hcc", "top", "dijets"]
        elif self.signal == "Hcc":
            self.backgrounds = ["Hbb", "top", "dijets"]
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

    def flav_string(self, flav):
        """Return the label string for a given flavour.

        Parameters
        ----------
        flav : str
            Flavour to get string for

        Returns
        -------
        str
            Flavour string
        """
        string = global_config["flavour_categories"][flav]["legend_label"]
        string = string.replace("jets", "jet")
        if flav == self.signal:
            string = f"{string} efficiency"
        else:
            string = f"{string} rejection"
        return string

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
        cuts : lits
            List of cuts to apply
        num_jets : int, optional
            Number of jets to load from the file, by default all jets
        perf_var : str, optional
           Override the performance variable to use, by default None
        """
        if cuts is None:
            cuts = []

        # get a list of all variables to be loaded from the file
        var_list = sum([tagger.variables for tagger in taggers], [label_var])

        var_list += [cut[0] for cut in cuts]
        var_list = list(set(var_list))

        # load data
        with h5py.File(file_path) as file:
            data = file[key].fields(var_list)[:num_jets]

        # apply cuts
        for var, cut_op, value in cuts:
            perf_var = perf_var[OPERATORS[cut_op](data[var], value)]
            data = data[OPERATORS[cut_op](data[var], value)]

        # add taggers from loaded data
        for tagger in taggers:
            tagger.extract_tagger_scores(data, source_type="structured_array")

            if label_var == "HadronConeExclTruthLabelID":
                tagger.is_flav["bjets"] = data[label_var] == 5
                tagger.is_flav["ujets"] = data[label_var] == 0
                tagger.is_flav["cjets"] = data[label_var] == 4
            elif label_var == "R10TruthLabel_R22v1":
                tagger.is_flav["Hbb"] = data[label_var] == 11
                tagger.is_flav["Hcc"] = data[label_var] == 12
                tagger.is_flav["top"] = data[label_var] == 1
                tagger.is_flav["dijets"] = data[label_var] == 10
            if perf_var is None:
                tagger.perf_var = data[self.perf_var] * 0.001
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

    def plot_rocs(
        self,
        plot_name: str,
        args_roc_plot: dict = None,
    ):
        """Plots rocs

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
            "xlabel": self.flav_string(self.signal),
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
                    discs[tagger.is_flav[self.signal]],
                    discs[tagger.is_flav[background]],
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
        **kwargs : kwargs
            key word arguments for `puma.VarVsEff`
        """
        # define the curves
        plot_sig_eff = VarVsEffPlot(
            mode="sig_eff",
            ylabel=self.flav_string(self.signal),
            xlabel=xlabel,
            logy=False,
            atlas_second_tag=self.atlas_second_tag,
            n_ratio_panels=1,
            y_scale=1.4,
        )
        plot_bkg = []
        for background in self.backgrounds:

            plot_bkg.append(
                VarVsEffPlot(
                    mode="bkg_rej",
                    ylabel=self.flav_string(background),
                    xlabel=xlabel,
                    logy=False,
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
            is_signal = tagger.is_flav[self.signal]
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
                is_bkg = tagger.is_flav[background]
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
        plot_sig_eff.savefig(f"{plot_name}_{self.signal}_eff.png")
        for i, background in enumerate(self.backgrounds):
            plot_bkg[i].draw()
            plot_bkg[i].savefig(f"{plot_name}_{background}_rej.png")

    def plot_discs(
        self,
        plot_name: str,
        exclude_tagger: list = None,
        xlabel: str = None,
        **kwargs,
    ):
        """Plots discriminant


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
            xlabel = rf"$D_{{{self.signal.rstrip('jets')}}}$"

        flav_cat = global_config["flavour_categories"]
        line_styles = get_good_linestyles()

        tagger_output_plot = HistogramPlot(
            n_ratio_panels=0,
            xlabel=xlabel,
            ylabel="Normalised number of jets",
            figsize=(7.0, 4.5),
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
                        discs[tagger.is_flav[flav]],
                        ratio_group=flav,
                        label=flav_cat[flav]["legend_label"] if tag_i == 0 else None,
                        colour=flav_cat[flav]["colour"],
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
