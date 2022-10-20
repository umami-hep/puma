# import h5py

from puma import Histogram, HistogramPlot, Roc, RocPlot, VarVsEff, VarVsEffPlot
from puma.utils import global_config, logger


class Results:
    """Stores all results of the different taggers."""

    def __init__(self) -> None:
        self.taggers = []
        self.model_names = []
        # defining target efficiency
        self.sig_eff = None
        self.atlas_second_tag = None
        self.plot_name = None
        self.n_jets_light = None
        self.n_jets_c = None
        self.is_light = None
        self.is_c = None
        self.is_b = None

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

    def prepare_discs(self, df):
        # defining boolean arrays to select the different flavour classes
        self.is_light = df["HadronConeExclTruthLabelID"] == 0
        self.is_c = df["HadronConeExclTruthLabelID"] == 4
        self.is_b = df["HadronConeExclTruthLabelID"] == 5

        self.n_jets_light = sum(self.is_light)
        self.n_jets_c = sum(self.is_c)
        for tagger in self.taggers:
            tagger.calc_discs(df)
            tagger.calc_rej(
                self.sig_eff, is_b=self.is_b, is_light=self.is_light, is_c=self.is_c
            )

    def plot_rocs(self, df, plot_name):
        """Plots rocs

        Parameters
        ----------
        plot_roc : puma.RocPlot
            roc plot object
        """
        self.prepare_discs(df)
        plot_roc = RocPlot(
            n_ratio_panels=2,
            ylabel="Background rejection",
            xlabel="$b$-jet efficiency",
            atlas_second_tag=self.atlas_second_tag,
            figsize=(6.5, 6),
            y_scale=1.4,
        )

        for tagger in self.taggers:
            plot_roc.add_roc(
                Roc(
                    self.sig_eff,
                    tagger.ujets_rej,
                    n_test=self.n_jets_light,
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
                    tagger.cjets_rej,
                    n_test=self.n_jets_c,
                    rej_class="cjets",
                    signal_class="bjets",
                    label=tagger.label,
                    colour=tagger.colour,
                ),
                reference=tagger.reference,
            )

        # setting which flavour rejection ratio is drawn in which ratio panel
        plot_roc.set_ratio_class(1, "ujets", label="Light-jet ratio")
        plot_roc.set_ratio_class(2, "cjets", label="$c$-jet ratio")
        plot_roc.set_leg_rej_labels("ujets", "Light-jet rejection")
        plot_roc.set_leg_rej_labels("cjets", "$c$-jet rejection")

        plot_roc.draw()
        plot_roc.savefig(plot_name)

    def plot_pt_perf(self, df, plot_name, **kwargs):
        self.prepare_discs(df)
        # define the curves
        plot_light_rej = VarVsEffPlot(
            mode="bkg_rej",
            ylabel="Light-flavour jets rejection",
            xlabel=r"$p_{T}$ [GeV]",
            logy=False,
            atlas_second_tag=self.atlas_second_tag,
            n_ratio_panels=1,
        )
        plot_c_rej = VarVsEffPlot(
            mode="bkg_rej",
            ylabel=r"$c$-jets rejection",
            xlabel=r"$p_{T}$ [GeV]",
            logy=False,
            atlas_second_tag=self.atlas_second_tag,
            n_ratio_panels=1,
        )
        plot_b_eff = VarVsEffPlot(
            mode="sig_eff",
            ylabel="$b$-jets efficiency",
            xlabel=r"$p_{T}$ [GeV]",
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
            plot_light_rej.add(
                VarVsEff(
                    x_var_sig=df["pt_btagJes"][self.is_b],
                    disc_sig=tagger.discs[self.is_b],
                    x_var_bkg=df["pt_btagJes"][self.is_light],
                    disc_bkg=tagger.discs[self.is_light],
                    label=tagger.label,
                    colour=tagger.colour,
                    **kwargs,
                ),
                reference=tagger.reference,
            )
            plot_c_rej.add(
                VarVsEff(
                    x_var_sig=df["pt_btagJes"][self.is_b],
                    disc_sig=tagger.discs[self.is_b],
                    x_var_bkg=df["pt_btagJes"][self.is_c],
                    disc_bkg=tagger.discs[self.is_c],
                    label=tagger.label,
                    colour=tagger.colour,
                    **kwargs,
                ),
                reference=tagger.reference,
            )
            plot_b_eff.add(
                VarVsEff(
                    x_var_sig=df["pt_btagJes"][self.is_b],
                    disc_sig=tagger.discs[self.is_b],
                    x_var_bkg=df["pt_btagJes"][self.is_light],
                    disc_bkg=tagger.discs[self.is_light],
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
        df,
        plot_name,
        exclude_tagger=None,
        xlabel=r"$D_{b}$",
        tag_var=None,
        **kwargs,
    ):
        """Plots discriminant

        Parameters
        ----------
        plot_roc : puma.RocPlot
            roc plot object
        """
        self.prepare_discs(df)

        flav_cat = global_config["flavour_categories"]
        line_styles = ["-", (0, (5, 1))]

        tagger_output_plot = HistogramPlot(
            n_ratio_panels=1,
            xlabel=xlabel,
            ylabel="Normalised number of jets",
            figsize=(8, 6),
            **kwargs,
        )

        for tag_i, tagger in enumerate(self.taggers):
            if exclude_tagger is not None and tagger.model_name in exclude_tagger:
                continue
            if tag_var is None:
                disc_plot = tagger.discs
            else:
                disc_plot = df[f"{tagger.model_name}_{tag_var}"]
            tagger_output_plot.add(
                Histogram(
                    disc_plot[self.is_light],
                    ratio_group="ujets",
                    label=tagger.label + " ujets",
                    colour=flav_cat["ujets"]["colour"],
                    linestyle=line_styles[tag_i],
                ),
                reference=tagger.reference,
            )
            tagger_output_plot.add(
                Histogram(
                    disc_plot[self.is_c],
                    ratio_group="cjets",
                    label=tagger.label + " cjets",
                    colour=flav_cat["cjets"]["colour"],
                    linestyle=line_styles[tag_i],
                ),
                reference=tagger.reference,
            )
            tagger_output_plot.add(
                Histogram(
                    disc_plot[self.is_b],
                    ratio_group="bjets",
                    label=tagger.label + " bjets",
                    colour=flav_cat["bjets"]["colour"],
                    linestyle=line_styles[tag_i],
                ),
                reference=tagger.reference,
            )
        tagger_output_plot.draw()
        tagger_output_plot.savefig(plot_name)
