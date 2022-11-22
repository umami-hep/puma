"""Results module for high level API."""
from puma import Histogram, HistogramPlot, Roc, RocPlot, VarVsEff, VarVsEffPlot
from puma.metrics import calc_rej
from puma.utils import get_good_linestyles, global_config, logger


class ResultsBase:
    """Base class handling plotting of tagger results."""

    def __init__(self) -> None:
        self.taggers = []
        self.model_names = []
        # defining target efficiency
        self.sig_eff = None

    def add(self, tagger):
        """Add tagger to class.

        Parameters
        ----------
        tagger : Tagger
            class containing tagger info

        Raises
        ------
        KeyError
            if model name duplicated
        """
        if tagger.model_name in self.model_names:
            raise KeyError(
                f"You are adding the model {tagger.model_name} but it was already "
                "added."
            )
        self.model_names.append(tagger.model_name)
        self.taggers.append(tagger)

    def get(self, model_name):
        """Retrieve tagger info.

        Parameters
        ----------
        model_name : str
            name of model

        Returns
        -------
        Tagger
            Tagger class with info about tagger
        """
        return self.taggers[self.model_names.index(model_name)]


class Results(ResultsBase):
    """Stores all results of the different taggers."""

    def __init__(self) -> None:
        super().__init__()
        self.atlas_second_tag = None

    def plot_rocs(self, plot_name, signal_class: str = "bjets", args_roc_plot: dict = None):
        """Plots rocs

        Parameters
        ----------
        plot_name : puma.RocPlot
            roc plot object
        signal_class : str, optional
            signal class to plot Roc with, wither `bjets` or `cjets`, by default `bjets`
        args_roc_plot: dict, optional
            key word arguments being passed to `RocPlot`
        """
        roc_plot_args = {
            "n_ratio_panels": 2,
            "ylabel": "Background rejection",
            "xlabel": "$b$-jet efficiency"
            if signal_class == "bjets"
            else "$c$-jet efficiency",
            "atlas_second_tag": self.atlas_second_tag,
            "figsize": (6.5, 6),
            "y_scale": 1.4,
        }
        # TODO: update in python 3.9
        if args_roc_plot is not None:
            roc_plot_args.update(args_roc_plot)
        plot_roc = RocPlot(**roc_plot_args)

        for tagger in self.taggers:
            discs = (
                tagger.calc_disc_b()
                if signal_class == "bjets"
                else tagger.calc_disc_c()
            )
            signal_selection = tagger.is_b if signal_class == "bjets" else tagger.is_c
            # print("discs", discs)
            # print(signal_selection)
            bkg_selection = tagger.is_c if signal_class == "bjets" else tagger.is_b
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
        plot_roc.set_ratio_class(1, "ujets", label="Light-jet ratio")
        plot_roc.set_ratio_class(
            2,
            "cjets",
            label="$c$-jet ratio" if signal_class == "bjets" else "$b$-jet ratio",
        )
        plot_roc.set_leg_rej_labels("ujets", "Light-jet rejection")
        plot_roc.set_leg_rej_labels(
            "cjets",
            "$c$-jet rejection" if signal_class == "bjets" else "$b$-jet rejection",
        )

        plot_roc.draw()
        plot_roc.savefig(plot_name)

    def plot_var_perf(
        self, plot_name: str, xlabel=r"$p_{T}$ [GeV]", signal_class="bjets", **kwargs
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
        **kwargs : kwargs
            key word arguments for `puma.VarVsEff`
        """
        # define the curves
        plot_light_rej = VarVsEffPlot(
            mode="bkg_rej",
            ylabel="Light-flavour jets rejection",
            xlabel=xlabel,
            logy=False,
            atlas_second_tag=self.atlas_second_tag,
            n_ratio_panels=1,
        )
        plot_c_rej = VarVsEffPlot(
            mode="bkg_rej",
            ylabel=r"$c$-jets rejection",
            xlabel=xlabel,
            logy=False,
            atlas_second_tag=self.atlas_second_tag,
            n_ratio_panels=1,
        )
        plot_b_eff = VarVsEffPlot(
            mode="sig_eff",
            ylabel="$b$-jets efficiency",
            xlabel=xlabel,
            logy=False,
            atlas_second_tag=self.atlas_second_tag,
            n_ratio_panels=1,
        )
        disc_cut_in_kwargs = "disc_cut" in kwargs
        working_point_in_kwargs = "working_point" in kwargs
        for tagger in self.taggers:
            if not disc_cut_in_kwargs:
                kwargs["disc_cut"] = tagger.disc_cut
            if not working_point_in_kwargs:
                kwargs["working_point"] = tagger.working_point

            discs = (
                tagger.calc_disc_b()
                if signal_class == "bjets"
                else tagger.calc_disc_c()
            )
            # TODO: add switch everywhere for c-jets as signal_class as well
            plot_light_rej.add(
                VarVsEff(
                    x_var_sig=tagger.perf_var[tagger.is_b],
                    disc_sig=discs[tagger.is_b],
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
                    x_var_sig=tagger.perf_var[tagger.is_b],
                    disc_sig=discs[tagger.is_b],
                    x_var_bkg=tagger.perf_var[tagger.is_c],
                    disc_bkg=discs[tagger.is_c],
                    label=tagger.label,
                    colour=tagger.colour,
                    **kwargs,
                ),
                reference=tagger.reference,
            )
            plot_b_eff.add(
                VarVsEff(
                    x_var_sig=tagger.perf_var[tagger.is_b],
                    disc_sig=discs[tagger.is_b],
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
        plot_c_rej.savefig(f"{plot_name}_pt_c_rej.png")

        plot_b_eff.draw()
        # Drawing a hline indicating inclusive efficiency
        plot_b_eff.draw_hline(0.7)
        plot_b_eff.savefig(f"{plot_name}_pt_b_eff.png")

    def plot_discs(
        self,
        plot_name: str,
        exclude_tagger: list = None,
        xlabel: str = r"$D_{b}$",
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
        """
        flav_cat = global_config["flavour_categories"]
        line_styles = get_good_linestyles()

        tagger_output_plot = HistogramPlot(
            n_ratio_panels=1,
            xlabel=xlabel,
            ylabel="Normalised number of jets",
            figsize=(6.5, 5.5),
            atlas_second_tag=self.atlas_second_tag,
            **kwargs,
        )
        tag_i = 0
        tag_labels = []
        for tagger in self.taggers:
            if exclude_tagger is not None and tagger.model_name in exclude_tagger:
                continue
            discs = (
                tagger.calc_disc_b()
                if signal_class == "bjets"
                else tagger.calc_disc_c()
            )
            tagger_output_plot.add(
                Histogram(
                    discs[tagger.is_light],
                    ratio_group="ujets",
                    label="Light-flavour jets" if tag_i == 0 else None,
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
            tag_labels.append(tagger.label)
        tagger_output_plot.draw()
        tagger_output_plot.make_linestyle_legend(
            linestyles=line_styles[:tag_i],
            labels=tag_labels,
            bbox_to_anchor=(0.55, 1),
        )
        tagger_output_plot.savefig(plot_name)
