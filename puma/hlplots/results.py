"""Results module for high level API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ftag import Cuts, Flavours, Label
from ftag.hdf5 import H5Reader
from ftag.utils import calculate_efficiency, calculate_rejection
from matplotlib.figure import Figure

from puma import (
    Histogram,
    HistogramPlot,
    Line2D,
    Line2DPlot,
    Roc,
    RocPlot,
    VarVsEff,
    VarVsEffPlot,
    fraction_scan,
)
from puma.hlplots.tagger import Tagger
from puma.hlplots.yutils import combine_suffixes
from puma.utils import get_good_colours, get_good_linestyles, logger


@dataclass
class Results:
    """Store information about several taggers and plot results."""

    signal: Label | str
    sample: str
    category: str = "single-btag"
    atlas_first_tag: str = "Simulation Internal"
    atlas_second_tag: str = None
    atlas_third_tag: str = None
    taggers: dict = field(default_factory=dict)
    perf_vars: str | tuple | list = "pt"
    output_dir: str | Path = "."
    extension: str = "pdf"
    global_cuts: Cuts | list | None = None
    num_jets: int | None = None
    remove_nan: bool = False
    label_var: str = "HadronConeExclTruthLabelID"

    def __post_init__(self):
        self.set_signal(self.signal)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.perf_vars, str):
            self.perf_vars = [self.perf_vars]
        if self.atlas_second_tag is not None and self.atlas_third_tag is not None:
            self.atlas_second_tag = f"{self.atlas_second_tag}\n{self.atlas_third_tag}"

        self.plot_funcs = {
            "probs": self.plot_probs,
            "disc": self.plot_discs,
            "roc": self.plot_rocs,
            "peff": self.plot_var_perf,
            "scan": self.plot_fraction_scans,
        }
        self.saved_plots = []

    def set_signal(self, signal: Label):
        """Set the signal flavour and define background flavours.

        Parameters
        ----------
        signal : Flavour
            Flavour which is the signal

        Raises
        ------
        ValueError
            If the signal class is not supported
        """
        if isinstance(signal, str):
            signal = Flavours[signal]
        self.signal = signal

        # Get the full flavours from the category chosen
        self.backgrounds = [*Flavours.by_category(self.category).backgrounds(self.signal)]

    @property
    def sig_str(self):
        """Returns the name of the signal as a string.

        Returns
        -------
        str
            Name of the signal as string
        """
        suffix = "jets"
        sig = str(self.signal)
        if sig.endswith(suffix):
            sig = sig[: -len(suffix)]
        return f"{sig}"

    @property
    def flavours(self):
        """Return a list of all flavours.

        Returns
        -------
        list
            List of all flavours
        """
        return [*self.backgrounds, self.signal]

    def add(self, tagger: Tagger):
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

        if not tagger.colour:
            good_colours = get_good_colours()
            current_colours = [t.colour for t in self.taggers.values()]
            tagger.colour = next(c for c in good_colours if c not in current_colours)

        self.taggers[str(tagger)] = tagger

    def load(self):
        """Iterates all taggers, and loads data if it hasn't already been loaded."""
        req_load = [tagger for tagger in self.taggers.values() if tagger.scores is None]
        tagger_paths = list({tagger.sample_path for tagger in req_load})
        for tp in tagger_paths:
            tp_taggers = [tagger for tagger in req_load if tagger.sample_path == tp]
            self.load_taggers_from_file(
                tp_taggers,
                tp,
                cuts=self.global_cuts,
                num_jets=self.num_jets,
                label_var=self.label_var,
            )

    def load_taggers_from_file(  # pylint: disable=R0913
        self,
        taggers: list[Tagger],
        file_path: Path | str,
        key: str = "jets",
        label_var: str = "HadronConeExclTruthLabelID",
        cuts: Cuts | list | None = None,
        num_jets: int | None = None,
        perf_vars: dict | None = None,
    ):
        """Load one or more taggers from a common file. Adds the tagger to this results class
        if it is not already present.

        Parameters
        ----------
        taggers : list[Tagger]
            List of taggers to add
        file_path : str | Path
            Path to file
        key : str, optional
            Key in file, by default 'jets'
        label_var : str, optional
            Label variable to use, by default 'HadronConeExclTruthLabelID'
        cuts : Cuts | list, optional
            Cuts to apply, by default None
        num_jets : int, optional
            Number of jets to load from the file, by default all jets
        perf_vars : dict, optional
            Override the performance variables to use, by default None
        """

        def check_nan(data: np.ndarray) -> np.ndarray:
            """Filter out NaN values from loaded data.

            Parameters
            ----------
            data : ndarray
                Data to filter

            Returns
            -------
            np.ndarray
                Array with NaN values removed

            Raises
            ------
            ValueError
                If NaN values are found but the setting is to keep them
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
        var_list = sum([tagger.variables for tagger in taggers], [label_var])
        var_list += cuts.variables
        var_list += sum([t.cuts.variables for t in taggers if t.cuts is not None], [])
        var_list = list(set(var_list + self.perf_vars))

        # load data
        reader = H5Reader(file_path, precision="full")
        data = reader.load({key: var_list}, num_jets)[key]

        # check for nan values
        data = check_nan(data)
        # apply common cuts
        if cuts:
            idx, data = cuts(data)
            if perf_vars is not None:
                for perf_var_name, perf_var_array in perf_vars.items():
                    perf_vars[perf_var_name] = perf_var_array[idx]

        # for each tagger
        for tagger in taggers:
            sel_data = data
            sel_perf_vars = perf_vars

            # apply tagger specific cuts
            if tagger.cuts:
                idx, sel_data = tagger.cuts(data)

            # attach data to tagger objects
            tagger.extract_tagger_scores(sel_data, source_type="structured_array")
            tagger.labels = np.array(sel_data[label_var], dtype=[(label_var, "i4")])
            if perf_vars is None:
                tagger.perf_vars = {}
                for perf_var in self.perf_vars:
                    if any(x in perf_var for x in ["pt", "mass"]):
                        tagger.perf_vars[perf_var] = sel_data[perf_var] * 0.001
                    else:
                        tagger.perf_vars[perf_var] = sel_data[perf_var]
            else:
                tagger.perf_vars = sel_perf_vars

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

    def save(
        self,
        plot: Figure,
        plot_type: str,
        base: str | None = None,
        suffix: str | None = None,
    ):
        """Get the output file path.

        Parameters
        ----------
        plot : Figure
            Matplotlib figure to save.
        plot_type : str
            Plots of the same type are saved in the same directory.
        base_fname : str
            Base filename, modified by this function.
        suffix : str, optional
            Suffix to add to the filename, by default None
        """
        tag_str = f"{self.sig_str}tag"
        out_dir = self.output_dir / tag_str / plot_type
        out_dir.mkdir(parents=True, exist_ok=True)
        if not base:
            base = plot_type
        fname = f"{self.sample}_{tag_str}"
        fname += f"_{base}"
        if suffix:
            fname += f"_{suffix}"
        fpath = out_dir / f"{fname}.{self.extension}"
        plot.savefig(fpath)
        self.saved_plots.append(fpath)

    def plot_probs(
        self,
        suffix: str | None = None,
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
        # Get good linestyles for plotting
        line_styles = get_good_linestyles()

        # Get a list of all flavours which are present
        flavours = [*self.backgrounds, self.signal]

        # Remove any flavours that are not used by all taggers
        flavours = [
            flav
            for flav in flavours
            if all(flav in tagger.output_flavours for tagger in self.taggers.values())
        ]

        # Init a default kwargs dict for the HistogramPlot
        histo_kwargs = {
            "ylabel": "Normalised number of jets",
            "figsize": (7.0, 4.5),
            "n_ratio_panels": 1,
            "atlas_first_tag": self.atlas_first_tag,
            "atlas_second_tag": self.atlas_second_tag,
        }

        # If kwargs are given, update the histo_kwargs dict
        if kwargs is not None:
            histo_kwargs.update(kwargs)

        # group by output probability
        for flav_prob in flavours:
            # Create a new histogram plot
            hist = HistogramPlot(
                xlabel=flav_prob.px,
                **histo_kwargs,
            )

            # Init a new list for the tagger labels
            tagger_labels = []

            # Loop over the taggers
            for counter, tagger in enumerate(self.taggers.values()):
                # Append labels if existing else the name
                tagger_labels.append(tagger.label if tagger.label else tagger.name)

                # Add the probability output of the given tagger for each flavour
                for flav_class in flavours:
                    hist.add(
                        Histogram(
                            tagger.probs(flav_prob, flav_class),
                            ratio_group=flav_class,
                            label=flav_class.label if counter == 0 else None,
                            colour=flav_class.colour,
                            linestyle=line_styles[counter],
                        ),
                        reference=tagger.reference,
                    )

            # Finalise the plot and draw it
            hist.draw()
            hist.make_linestyle_legend(
                linestyles=line_styles,
                labels=tagger_labels,
                bbox_to_anchor=(0.55, 1),
            )
            self.save(hist, "prob", flav_prob.px, suffix)

        # group by flavour
        for flav_class in flavours:
            # Create a new histogram plot
            hist = HistogramPlot(
                xlabel=flav_class.label,
                **histo_kwargs,
            )

            # Init a new list for the tagger labels
            tagger_labels = []

            # Loop over the taggers
            for counter, tagger in enumerate(self.taggers.values()):
                # Append labels if existing else the name
                tagger_labels.append(tagger.label if tagger.label else tagger.name)

                # Add the probability output of the given tagger for each flavour
                for flav_prob in flavours:
                    hist.add(
                        Histogram(
                            tagger.probs(flav_prob, flav_class),
                            ratio_group=flav_prob,
                            label=flav_prob.px if counter == 0 else None,
                            colour=flav_prob.colour,
                            linestyle=line_styles[counter],
                        ),
                        reference=tagger.reference,
                    )

            # Finalise the plot and draw it
            hist.draw()
            hist.make_linestyle_legend(
                linestyles=line_styles,
                labels=tagger_labels,
                bbox_to_anchor=(0.55, 1),
            )
            self.save(hist, "prob", flav_class, suffix)

    def plot_discs(
        self,
        suffix: str | None = None,
        exclude_tagger: list | None = None,
        xlabel: str | None = None,
        wp_vlines: list | None = None,
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

        # Get good linestyles for plotting
        line_styles = get_good_linestyles()

        # Init histo_kwargs
        histo_kwargs = {
            "n_ratio_panels": 0,
            "xlabel": xlabel,
            "ylabel": "Normalised number of jets",
            "figsize": (7.0, 4.5),
            "atlas_first_tag": self.atlas_first_tag,
            "atlas_second_tag": self.atlas_second_tag,
        }

        # Check if kwargs are given and update the histo_kwargs accordingly
        if kwargs is not None:
            histo_kwargs.update(kwargs)

        # Create a new histogram plot
        hist = HistogramPlot(**histo_kwargs)

        # Init a tagger label list
        tagger_labels = []

        # Loop over the defined taggers
        for counter, tagger in enumerate(self.taggers.values()):
            # Check if the tagger is excluded and skip it if so
            if exclude_tagger is not None and tagger.name in exclude_tagger:
                continue

            # Get the discriminant values from the tagger for the given signal
            discs = tagger.discriminant(self.signal)

            # Get lists for the working put cuts and labels
            wp_cuts, wp_labels = [], []

            # get working point cuts and labels and append them
            for wp in wp_vlines:
                cut = np.percentile(discs[tagger.is_flav(self.signal)], 100 - wp)
                label = None if counter > 0 else f"{wp}%"
                wp_cuts.append(cut)
                wp_labels.append(label)

            # Draw the vertical lines for the working points
            hist.draw_vlines(wp_cuts, labels=wp_labels, linestyle=line_styles[counter])

            # Loop over the flavours and add the disc values for each flavour
            for flav in self.backgrounds:
                if flav in tagger.output_flavours:
                    hist.add(
                        Histogram(
                            discs[tagger.is_flav(flav)],
                            ratio_group=flav,
                            label=flav.label if counter == 0 else None,
                            colour=flav.colour,
                            linestyle=line_styles[counter],
                        ),
                        reference=tagger.reference,
                    )

            # Add the disc values for the signal
            hist.add(
                Histogram(
                    discs[tagger.is_flav(self.signal)],
                    ratio_group=self.signal,
                    label=self.signal.label if counter == 0 else None,
                    colour=self.signal.colour,
                    linestyle=line_styles[counter],
                ),
                reference=tagger.reference,
            )

            # Add the tagger label or name to the label list
            tagger_labels.append(tagger.label if tagger.label else tagger.name)

        # Finalise the plot and draw it
        hist.draw()
        hist.make_linestyle_legend(
            linestyles=line_styles,
            labels=tagger_labels,
            bbox_to_anchor=(0.55, 1),
        )
        self.save(hist, "disc", suffix=suffix)

    def plot_rocs(
        self,
        x_range: tuple[float, float] | None = (0.5, 1.0),
        resolution: int = 50,
        suffix: str | None = None,
        skip_missing_flavours: bool = True,
        **kwargs,
    ):
        """Plots rocs.

        Parameters
        ----------
        x_range : tuple, optional
            x-axis range, by default None
        resolution : int, optional
            number of points to use for the x-axis, by default 100
        suffix : str, optional
            suffix to add to output file name, by default None
        skip_missing_flavours : bool, optional
            If True, skip making ROC curves for flavours that are
            not present in every tagger, by default True
        kwargs: dict, optional
            key word arguments being passed to `RocPlot`
        """
        # Linspace the signal efficiencies
        sig_effs = np.linspace(*x_range, resolution)

        # Define int for ratio panel number
        n_ratio_panels = 0

        # Check how many backgrounds are present
        present_flavs = []
        for background in self.backgrounds:
            is_present = False
            for tagger in self.taggers.values():
                if background in tagger.output_flavours:
                    is_present = True
                    if background not in present_flavs:
                        present_flavs.append(background)
            if is_present:
                n_ratio_panels += 1

        # Init a default kwargs dict for roc plots
        roc_kwargs = {
            "n_ratio_panels": n_ratio_panels,
            "ylabel": "Background rejection",
            "xlabel": self.signal.eff_str,
            "atlas_first_tag": self.atlas_first_tag,
            "atlas_second_tag": self.atlas_second_tag,
            "y_scale": 1.3,
            "ymin": 1,
        }

        # If kwargs are given, update the default kwargs accordingly
        if kwargs is not None:
            kwargs.update(roc_kwargs)

        # Init a new ROC plot using the given/default kwargs
        roc = RocPlot(**kwargs)

        # Iterate over the taggers
        for tagger in self.taggers.values():
            # Get the disc values for the given tagger
            discs = tagger.discriminant(self.signal)

            # Loop over all backgrouns
            for background in self.backgrounds:
                # Skip non-existing flavours
                if background not in tagger.output_flavours and skip_missing_flavours:
                    continue

                # Calculate rejection for the given background
                rej = calculate_rejection(
                    discs[tagger.is_flav(self.signal)],
                    discs[tagger.is_flav(background)],
                    sig_effs,
                    smooth=True,
                )

                # Add the rejection curve to the ROC plot
                roc.add_roc(
                    Roc(
                        sig_effs,
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
        for counter, background in enumerate(self.backgrounds):
            if background in present_flavs:
                roc.set_ratio_class(counter + 1, background)

        # Finalise the plot and draw it
        roc.draw()
        self.save(roc, "roc", suffix=suffix)

    def plot_var_perf(  # pylint: disable=too-many-locals
        self,
        suffix: str | None = None,
        xlabel: str = r"$p_{\mathrm{T}}$ [GeV]",
        perf_var: str = "pt",
        h_line: float | None = None,
        working_point: float | None = None,
        disc_cut: float | None = None,
        fixed_rejections: dict[Label, float] | None = None,
        **kwargs,
    ):
        r"""Variable vs efficiency/rejection plot.

        You can choose between different modes: "sig_eff", "bkg_eff", "sig_rej",
        "bkg_rej"

        Parameters
        ----------
        suffix : str, optional
            suffix to add to output file name, by default None
        xlabel : regexp, optional
            _description_, by default "$p_{T}$ [GeV]"
        perf_var: str, optional
            The x axis variable, default is 'pt'
        h_line : float, optional
            draws a horizonatal line in the signal efficiency plot
        working_point: float, optional
            The working point to use for the plot. Only one out of
            [working_point, disc_cut, fixed_rejections] can be set
        disc_cut: float, optional
            The cut on the discriminant to use for the plot. Only one out of
            [working_point, disc_cut, fixed_rejections] can be set
        fixed_rejections: dict[Flavour, float]
            Show signal efficiency as a function of fixed background rejection. Only one
            out of [working_point, disc_cut, fixed_rejections] can be set
        **kwargs : kwargs
            key word arguments for `puma.VarVsEff`
        """
        # Check correct setting of working_point, disc_cut and fixed_rejections
        if sum([bool(working_point), bool(disc_cut), bool(fixed_rejections)]) > 1:
            raise ValueError("Only one of working_point, disc_cut, or fixed_rejections can be set")
        if not any([working_point, disc_cut, fixed_rejections]):
            raise ValueError("Either working_point or disc_cut must be set")

        # If fixed_rejections is given, call different function
        if fixed_rejections:
            self.plot_flat_rej_var_perf(
                fixed_rejections=fixed_rejections,
                suffix=suffix,
                perf_var=perf_var,
                h_line=h_line,
                **kwargs,
            )
            return

        # Split the kwargs according to if they are used for the plot or the curve
        var_perf_plot_kwargs = {
            "xlabel": xlabel,
            "n_ratio_panels": 1,
            "atlas_first_tag": self.atlas_first_tag,
            "atlas_second_tag": self.atlas_second_tag,
            "y_scale": 1.5,
            "logy": False,
        }

        # Update the default plot kwargs if present in kwargs and remove it from kwargs
        if kwargs is not None:
            # Init a list to loop over
            iter_kwargs_list = list(kwargs.keys())

            # Loop over the kwargs list
            for key in iter_kwargs_list:
                if key in var_perf_plot_kwargs:
                    var_perf_plot_kwargs[key] = kwargs.pop(key)

        # Init new var vs eff plot
        plot_sig_eff = VarVsEffPlot(
            mode="sig_eff", ylabel=self.signal.eff_str, **var_perf_plot_kwargs
        )

        # Adapt the atlas second tag
        plot_sig_eff.apply_modified_atlas_second_tag(
            self.signal,
            working_point=working_point,
            disc_cut=disc_cut,
            flat_per_bin=kwargs.get("flat_per_bin", False),
        )

        # Init new list for background plots
        plot_bkg = []

        # Loop over all backgrounds
        for background in self.backgrounds:
            # Init and append new background plot to the list
            plot_bkg.append(
                VarVsEffPlot(mode="bkg_rej", ylabel=background.rej_str, **var_perf_plot_kwargs)
            )

            # Adapt the atlas second label accordingly
            plot_bkg[-1].apply_modified_atlas_second_tag(
                self.signal,
                working_point=working_point,
                disc_cut=disc_cut,
                flat_per_bin=kwargs.get("flat_per_bin", False),
            )

        # Loop over the given taggers
        for tagger in self.taggers.values():
            # Load the discriminant values and check if
            discs = tagger.discriminant(self.signal)
            is_signal = tagger.is_flav(self.signal)

            # Assure that the variable is in the data for the given tagger
            assert perf_var in tagger.perf_vars, f"{perf_var} not in tagger {tagger.name} data!"

            # Add the variable to the plot
            plot_sig_eff.add(
                VarVsEff(
                    x_var_sig=tagger.perf_vars[perf_var][is_signal],
                    disc_sig=discs[is_signal],
                    label=tagger.label,
                    colour=tagger.colour,
                    working_point=working_point,
                    disc_cut=disc_cut,
                    **kwargs,
                ),
                reference=tagger.reference,
            )

            # Loop over the background plots and add the variables
            for counter, background in enumerate(self.backgrounds):
                is_bkg = tagger.is_flav(background)
                plot_bkg[counter].add(
                    VarVsEff(
                        x_var_sig=tagger.perf_vars[perf_var][is_signal],
                        disc_sig=discs[is_signal],
                        x_var_bkg=tagger.perf_vars[perf_var][is_bkg],
                        disc_bkg=discs[is_bkg],
                        label=tagger.label,
                        colour=tagger.colour,
                        working_point=working_point,
                        disc_cut=disc_cut,
                        **kwargs,
                    ),
                    reference=tagger.reference,
                )

        # Finalise the plot and draw it
        plot_sig_eff.draw()
        if h_line:
            plot_sig_eff.draw_hline(h_line)

        # Generate the name according to inputs and save the figure
        plot_base = "flat_per_bin" if kwargs.get("flat_per_bin") else "fixed_cut"

        if disc_cut:
            wp_disc = f"disc_cut{disc_cut}".replace(".", "p")

        elif isinstance(working_point, list):
            wp_disc = (
                f"wp{int(working_point[0] * 100):.0f}_"
                f"{int(working_point[1] * 100):.0f}".replace(".", "p")
            )

        else:
            wp_disc = f"wp{int(working_point * 100)}".replace(".", "p")

        fname = f"{self.sig_str}eff_vs_{perf_var}_{plot_base}_{wp_disc}"
        suffix = f"{suffix}_" if suffix else ""
        self.save(plot_sig_eff, "profile", fname, suffix)

        # Save the background figures
        for counter, bkg in enumerate(self.backgrounds):
            plot_bkg[counter].draw()
            fname = f"{str(bkg)[0]}rej_vs_{perf_var}_{plot_base}_{wp_disc}"
            self.save(plot_bkg[counter], "profile", fname, suffix)

    def plot_flat_rej_var_perf(
        self,
        fixed_rejections: dict[Label, float],
        suffix: str | None = None,
        perf_var: str = "pt",
        h_line: float | None = None,
        **kwargs,
    ):
        """Plot signal efficiency as a function of a variable, with a fixed
        background rejection for each bin.

        Parameters
        ----------
        fixed_rejections : dict[Flavour, float]
            A dictionary of the fixed background rejections for each flavour, eg:
            fixed_rejections = {'cjets' : 0.1, 'ujets' : 0.01}
        suffix : str, optional
            suffix to add to output file name, by default None
        perf_var: str, optional
            The x axis variable, default is 'pt'
        h_line : float, optional
            draws a horizonatal line in the signal efficiency plot
        **kwargs : kwargs
            key word arguments for `puma.VarVsEff`
        """
        # Check for invalid inputs
        if inv_bkg := set(fixed_rejections.keys()) - {str(b) for b in self.backgrounds}:
            raise ValueError(f"Invalid background flavours: {inv_bkg}")
        if "disc_cut" in kwargs:
            raise ValueError("disc_cut should not be set for this plot")
        if "working_point" in kwargs:
            raise ValueError("working_point should not be set for this plot")

        # Split the kwargs according to if they are used for the plot or the curve
        var_perf_plot_kwargs = {
            "xlabel": r"$p_{\mathrm{T}}$ [GeV]",
            "n_ratio_panels": 1,
            "atlas_first_tag": self.atlas_first_tag,
            "atlas_second_tag": self.atlas_second_tag,
            "y_scale": 1.5,
            "logy": False,
        }

        # Update the default plot kwargs if present in kwargs and remove it from kwargs
        if kwargs is not None:
            # Init a list to loop over
            iter_kwargs_list = list(kwargs.keys())

            # Loop over the kwargs list
            for key in iter_kwargs_list:
                if key in var_perf_plot_kwargs:
                    var_perf_plot_kwargs[key] = kwargs.pop(key)

        # Get a list of all backgrounds
        backgrounds = [Flavours[b] for b in fixed_rejections]

        # Init a list for background plots
        plot_bkg = []

        # Loop over all backgrounds
        for bkg in backgrounds:
            var_perf_plot_kwargs["atlas_second_tag"] = (
                f"{self.atlas_second_tag}\nConstant {bkg.rej_str.lower()} of"
                f" {fixed_rejections[bkg.name]} per bin"
            )

            # Init and append the background plot to the list
            plot_bkg.append(
                VarVsEffPlot(
                    mode="bkg_eff_sig_err",
                    ylabel=self.signal.eff_str,
                    **var_perf_plot_kwargs,
                )
            )

        # After all plots are created, loop over the taggers
        for tagger in self.taggers.values():
            # Get the disc values
            discs = tagger.discriminant(self.signal)
            is_signal = tagger.is_flav(self.signal)

            # Loop over the backgrounds
            for counter, bkg in enumerate(backgrounds):
                is_bkg = tagger.is_flav(bkg)

                # Check that variable is present in tagger data
                assert perf_var in tagger.perf_vars, f"{perf_var} not in tagger {tagger.name} data!"

                # We want x bins to all have the same background rejection, so we
                # select the plot mode as 'bkg_eff', and then treat the signal as
                # the background here. I.e, the API plots 'bkg_eff' on the y axis,
                # while keeping the 'sig_eff' a flat rate on the x axis, we therefore
                # pass the signal as the background, and the background as the
                # signal.
                plot_bkg[counter].add(
                    VarVsEff(
                        x_var_sig=tagger.perf_vars[perf_var][is_bkg],
                        disc_sig=discs[is_bkg],
                        x_var_bkg=tagger.perf_vars[perf_var][is_signal],
                        disc_bkg=discs[is_signal],
                        label=tagger.label,
                        colour=tagger.colour,
                        working_point=1 / fixed_rejections[bkg.name],
                        flat_per_bin=True,
                        **kwargs,
                    ),
                    reference=tagger.reference,
                )

        # Update the suffix
        suffix = f"_{suffix}" if suffix else ""

        # Loop over the backgrounds, draw and save all plots
        for counter, bkg in enumerate(backgrounds):
            plot_bkg[counter].draw()
            if h_line:
                plot_bkg[counter].draw_hline(h_line)
            details = f"{self.sig_str}eff_vs_{perf_var}_"
            base = f"{str(bkg)[0]}rej_flat_{int(fixed_rejections[bkg.name])}"
            self.save(plot_bkg[counter], "profile", details + base, suffix)

    def plot_fraction_scans(
        self,
        backgrounds_to_plot: list[Label],
        suffix: str | None = None,
        efficiency: float = 0.7,
        rej: bool = False,
        plot_optimal_fraction_values: bool = False,
        fixed_fraction_values: dict | None = None,
        **kwargs,
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
        optimal_fc : bool, optional
            if True, plot optimal fc/fb, by default False
        backgrounds : list[Flavour], optional
            List of background flavours, by default None, will use self.backgrounds
        **kwargs
            Keyword arguments for `puma.Line2DPlot
        """
        # Get the background flavours in a list
        backgrounds = [Flavours[b] for b in backgrounds_to_plot]

        # Check if there are other flavours that need to be fixed
        if fixed_fraction_values is None:
            fixed_fraction_values = {}

        # Ensure that all backgrounds are provided or the fraction value is fixed
        for bkg in self.backgrounds:
            if bkg not in backgrounds and bkg.frac_str not in fixed_fraction_values:
                logger.warning(
                    f"Found flavour {bkg.name} in given label category without a fixed "
                    f"fraction value. Setting {bkg.frac_str} to 0!"
                )
                fixed_fraction_values[bkg.frac_str] = 0

        # Check that only two background flavours are given
        if len(backgrounds) != 2:
            raise ValueError("Only two background flavours are supported")

        # Adapt the plot name and suffix accordingly
        back_str = "_".join([f.name for f in backgrounds])
        suffix = combine_suffixes([f"{back_str}_eff{int(efficiency * 100)}", suffix])

        # Init a default kwargs dict
        fraction_scan_kwargs = {
            "logx": True,
            "logy": True,
        }

        # If kwargs are given, update the default dict accordingly
        if kwargs is not None:
            fraction_scan_kwargs.update(kwargs)

        # Define the granulartiy of the scan
        fxs = fraction_scan.get_fx_values(resolution=kwargs.pop("resolution", 100))

        # Adapt the tag
        tag = self.atlas_second_tag + "\n" if self.atlas_second_tag else ""
        tag += f"{self.signal.eff_str} = {efficiency:.0%}"
        tag += f"\n{backgrounds[0].frac_str} & {backgrounds[1].frac_str} Scan"

        # Define a new plot for the scan
        plot = Line2DPlot(atlas_second_tag=tag, **kwargs)

        # Get good colours and define, if the efficiency or the rejection is calculated
        eff_or_rej = calculate_efficiency if not rej else calculate_rejection
        colours = get_good_colours()

        # Loop over the taggers
        for counter, tagger in enumerate(self.taggers.values()):
            # Init zeros for both axes
            xs = np.zeros(len(fxs))
            ys = np.zeros(len(fxs))

            # Get the indices of the flavours
            sig_idx = tagger.is_flav(self.signal)
            bkg_1_idx = tagger.is_flav(backgrounds[0])
            bkg_2_idx = tagger.is_flav(backgrounds[1])

            # Loop over the fraction values
            for j, fx in enumerate(fxs):
                # Calculate disc values for the tagger for the given fraction values
                disc = tagger.discriminant(
                    self.signal,
                    fxs={
                        f"{backgrounds[0].frac_str}": fx,
                        f"{backgrounds[1].frac_str}": 1 - fx,
                        **fixed_fraction_values,
                    },
                )

                # Calculate the effciency/rejection and add it to the value arrays
                xs[j] = eff_or_rej(disc[sig_idx], disc[bkg_1_idx], efficiency)
                ys[j] = eff_or_rej(disc[sig_idx], disc[bkg_2_idx], efficiency)

            # add curve for this tagger
            tagger_fx = tagger.fxs[backgrounds[0].frac_str]
            plot.add(
                Line2D(
                    x_values=xs,
                    y_values=ys,
                    label=(
                        f"{tagger.label} "
                        f"(${backgrounds[0].frac_str}={tagger_fx:.3f}$, "
                        f"${backgrounds[1].frac_str}={1 - tagger_fx:.3f}$)"
                    ),
                    colour=tagger.colour if tagger.colour else colours[counter],
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
                    colour=tagger.colour,
                ),
                is_marker=True,
            )

            # Plot optimal fc if wanted
            if plot_optimal_fraction_values:
                opt_idx, opt_fc = fraction_scan.get_optimal_fraction_value(
                    fraction_scan=np.stack((xs, ys), axis=1),
                    fraction_space=fxs,
                    rej=rej,
                )
                plot.add(
                    Line2D(
                        x_values=xs[opt_idx],
                        y_values=ys[opt_idx],
                        marker="x",
                        markersize=15,
                        markeredgewidth=1,
                        label=(
                            f"Optimal ${backgrounds[0].frac_str}={opt_fc:.3f}$, "
                            f"${backgrounds[1].frac_str}={1 - opt_fc:.3f}$"
                        ),
                    ),
                    is_marker=True,
                )

            # Adding labels
            if not rej:
                plot.xlabel = backgrounds[0].eff_str
                plot.ylabel = backgrounds[1].eff_str

            else:
                plot.xlabel = backgrounds[0].rej_str
                plot.ylabel = backgrounds[1].rej_str

        # Draw and save the plot
        plot.draw()
        self.save(plot, "scan", "fraction_scan", suffix)

    def make_plot(self, plot_type: str, kwargs: dict):
        """Make a plot.

        Parameters
        ----------
        plot_type : str
            Type of plot
        kwargs : dict
            Keyword arguments for the plot
        """
        if plot_type not in self.plot_funcs:
            raise ValueError(f"Unknown plot type {plot_type}, choose from {self.plot_funcs.keys()}")
        self.plot_funcs[plot_type](**kwargs)
