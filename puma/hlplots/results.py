"""Results module for high level API."""
import operator
from dataclasses import dataclass, field

import h5py

from puma import Histogram, HistogramPlot, Roc, RocPlot, VarVsEff, VarVsEffPlot
from puma.metrics import calc_rej
from puma.utils import get_good_linestyles, global_config, logger

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

    atlas_second_tag: str = None
    taggers: dict = field(default_factory=dict)
    sig_eff: float = None

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
        if tagger.name in self.taggers:
            raise KeyError(f"{tagger.name} was already added.")
        self.taggers[tagger.name] = tagger

    def add_taggers_from_file(  # pylint: disable=R0913
        self,
        taggers,
        file_path,
        key="jets",
        label_var="HadronConeExclTruthLabelID",
        cuts=None,
        num_jets=None,
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
        """

        # get a list of all variables to be loaded from the file
        var_list = sum([tagger.variables for tagger in taggers], [label_var])
        var_list += [cut[0] for cut in cuts]
        var_list = list(set(var_list))

        # load data
        with h5py.File(file_path) as file:
            data = file[key].fields(var_list)[:num_jets]

        # apply cuts
        if cuts is None:
            cuts = []
        for var, cut_op, value in cuts:
            data = data[OPERATORS[cut_op](data[var], value)]

        # add taggers from loaded data
        for tagger in taggers:
            tagger.extract_tagger_scores(data, source_type="structured_array")
            tagger.is_b = data[label_var] == 5
            tagger.is_light = data[label_var] == 4
            tagger.is_c = data[label_var] == 0
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
        signal_class: str = "bjets",
        args_roc_plot: dict = None,
    ):
        """Plots rocs

        Parameters
        ----------
        plot_name : puma.RocPlot
            roc plot object
        signal_class : str, optional
            signal class to plot Roc with, wither `bjets` or `cjets`, by default `bjets`
        args_roc_plot: dict, optional
            key word arguments being passed to `RocPlot`

        Raises
        ------
        ValueError
            if specified signal class is invalid
        """
        if signal_class not in ["bjets", "cjets"]:
            raise ValueError(
                "So far only `bjets` and `cjets` are supported as signal class."
            )
        is_b_sig = signal_class == "bjets"
        roc_plot_args = {
            "n_ratio_panels": 2,
            "ylabel": "Background rejection",
            "xlabel": "$b$-jet efficiency" if is_b_sig else "$c$-jet efficiency",
            "atlas_second_tag": self.atlas_second_tag,
            "y_scale": 1.3,
        }
        # TODO: update in python 3.9
        if args_roc_plot is not None:
            roc_plot_args.update(args_roc_plot)
        plot_roc = RocPlot(**roc_plot_args)

        for tagger in self.taggers.values():
            discs = tagger.calc_disc_b() if is_b_sig else tagger.calc_disc_c()
            signal_selection = tagger.is_b if is_b_sig else tagger.is_c
            bkg_selection = tagger.is_c if is_b_sig else tagger.is_b
            light_rej = calc_rej(
                discs[signal_selection],
                discs[tagger.is_light],
                self.sig_eff,
            )
            c_or_b_rejection = calc_rej(
                discs[signal_selection],
                discs[bkg_selection],
                self.sig_eff,
            )
            plot_roc.add_roc(
                Roc(
                    self.sig_eff,
                    light_rej,
                    n_test=tagger.n_jets_light,
                    rej_class="ujets",
                    signal_class="bjets",
                    label=tagger.label,
                    colour=tagger.colour,
                ),
                reference=tagger.reference,
            )
            plot_roc.add_roc(
                Roc(
                    self.sig_eff,
                    c_or_b_rejection,
                    n_test=tagger.n_jets_c,
                    rej_class="cjets",
                    signal_class="bjets",
                    label=tagger.label,
                    colour=tagger.colour,
                ),
                reference=tagger.reference,
            )

        # setting which flavour rejection ratio is drawn in which ratio panel
        plot_roc.set_ratio_class(1, "ujets")
        plot_roc.set_ratio_class(2, "cjets")

        plot_roc.draw()
        plot_roc.savefig(plot_name)

    def plot_var_perf(  # pylint: disable=too-many-locals
        self,
        plot_name: str,
        xlabel: str = r"$p_{T}$ [GeV]",
        signal_class: str = "bjets",
        h_line: float = None,
        **kwargs,
    ):
        """Variable vs efficiency/rejection plot.

        Parameters
        ----------
        plot_name : str
            plot name base
        xlabel : regexp, optional
            _description_, by default "$p_{T}$ [GeV]"
        signal_class : str, optional
            takes either `bjets` or `cjets` as signal class, by default "bjets"
        h_line : float, optional
            draws a horizonatal line in the signal efficiency plot
        **kwargs : kwargs
            key word arguments for `puma.VarVsEff`

        Raises
        ------
        ValueError
            if specified signal class is invalid
        """
        if signal_class not in ["bjets", "cjets"]:
            raise ValueError(
                "So far only `bjets` and `cjets` are supported as signal class."
            )
        is_b_sig = signal_class == "bjets"
        # define the curves
        plot_light_rej = VarVsEffPlot(
            mode="bkg_rej",
            ylabel="Light-jets rejection",
            xlabel=xlabel,
            logy=False,
            atlas_second_tag=self.atlas_second_tag,
            n_ratio_panels=1,
        )
        plot_c_rej = VarVsEffPlot(
            mode="bkg_rej",
            ylabel=r"$c$-jets rejection" if is_b_sig else r"$b$-jets rejection",
            xlabel=xlabel,
            logy=False,
            atlas_second_tag=self.atlas_second_tag,
            n_ratio_panels=1,
        )
        plot_b_eff = VarVsEffPlot(
            mode="sig_eff",
            ylabel="$b$-jets efficiency" if is_b_sig else "$c$-jets efficiency",
            xlabel=xlabel,
            logy=False,
            atlas_second_tag=self.atlas_second_tag,
            n_ratio_panels=1,
        )
        disc_cut_in_kwargs = "disc_cut" in kwargs
        working_point_in_kwargs = "working_point" in kwargs
        for tagger in self.taggers.values():
            if not disc_cut_in_kwargs:
                kwargs["disc_cut"] = tagger.disc_cut
            if not working_point_in_kwargs:
                kwargs["working_point"] = tagger.working_point

            discs = tagger.calc_disc_b() if is_b_sig else tagger.calc_disc_c()
            # Switch signal and background if signal_class is not b_jets and using
            # c-jets as signal_class
            is_signal = tagger.is_b if is_b_sig else tagger.is_c
            is_bkg = tagger.is_c if is_b_sig else tagger.is_b
            plot_light_rej.add(
                VarVsEff(
                    x_var_sig=tagger.perf_var[is_signal],
                    disc_sig=discs[is_signal],
                    x_var_bkg=tagger.perf_var[tagger.is_light],
                    disc_bkg=discs[tagger.is_light],
                    label=tagger.label,
                    colour=tagger.colour,
                    **kwargs,
                ),
                reference=tagger.reference,
            )
            plot_c_rej.add(
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
            plot_b_eff.add(
                VarVsEff(
                    x_var_sig=tagger.perf_var[is_signal],
                    disc_sig=discs[is_signal],
                    x_var_bkg=tagger.perf_var[tagger.is_light],
                    disc_bkg=discs[tagger.is_light],
                    label=tagger.label,
                    colour=tagger.colour,
                    **kwargs,
                ),
                reference=tagger.reference,
            )

        logger.info(
            "Plotting bkg rejection for inclusive efficiency as a function of pt."
        )
        # You can choose between different modes: "sig_eff", "bkg_eff", "sig_rej",
        # "bkg_rej"

        plot_light_rej.draw()
        plot_light_rej.savefig(f"{plot_name}_pt_light_rej.png")

        plot_c_rej.draw()
        plot_c_rej.savefig(
            f"{plot_name}_pt_c_rej.png" if is_b_sig else f"{plot_name}_pt_b_rej.png"
        )

        plot_b_eff.draw()
        # Drawing a hline indicating inclusive efficiency
        if h_line:
            plot_b_eff.draw_hline(h_line)
        plot_b_eff.savefig(
            f"{plot_name}_pt_b_eff.png" if is_b_sig else f"{plot_name}_pt_c_eff.png"
        )

    def plot_discs(
        self,
        plot_name: str,
        exclude_tagger: list = None,
        xlabel: str = None,
        signal_class: str = "bjets",
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
        signal_class : str, optional
            Signal class which can be either "bjets" or "cjets", by default "bjets"
        **kwargs : kwargs
            key word arguments for `puma.HistogramPlot`

        Raises
        ------
        ValueError
            if specified signal class is invalid
        """
        if signal_class not in ["bjets", "cjets"]:
            raise ValueError(
                "So far only `bjets` and `cjets` are supported as signal class."
            )
        is_b_sig = signal_class == "bjets"
        if is_b_sig and xlabel is None:
            xlabel = r"$D_{b}$"
        elif not is_b_sig and xlabel is None:
            xlabel = r"$D_{c}$"

        flav_cat = global_config["flavour_categories"]
        line_styles = get_good_linestyles()

        tagger_output_plot = HistogramPlot(
            bins=50,
            n_ratio_panels=0,
            xlabel=xlabel,
            ylabel="Normalised number of jets",
            figsize=(7.0, 4.5),
            logy=True,
            atlas_second_tag=self.atlas_second_tag,
            **kwargs,
        )
        tag_i = 0
        tag_labels = []
        for tagger in self.taggers.values():
            if exclude_tagger is not None and tagger.name in exclude_tagger:
                continue
            discs = tagger.calc_disc_b() if is_b_sig else tagger.calc_disc_c()
            tagger_output_plot.add(
                Histogram(
                    discs[tagger.is_light],
                    ratio_group="ujets",
                    label="Light-jets" if tag_i == 0 else None,
                    colour=flav_cat["ujets"]["colour"],
                    linestyle=line_styles[tag_i],
                ),
                reference=tagger.reference,
            )
            tagger_output_plot.add(
                Histogram(
                    discs[tagger.is_c],
                    ratio_group="cjets",
                    label="$c$-jets" if tag_i == 0 else None,
                    colour=flav_cat["cjets"]["colour"],
                    linestyle=line_styles[tag_i],
                ),
                reference=tagger.reference,
            )
            tagger_output_plot.add(
                Histogram(
                    discs[tagger.is_b],
                    ratio_group="bjets",
                    label="$b$-jets" if tag_i == 0 else None,
                    colour=flav_cat["bjets"]["colour"],
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
