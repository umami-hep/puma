"""Histogram plot functions."""
import matplotlib as mpl
import numpy as np
import pandas as pd

import umami.tools.PyATLASstyle.PyATLASstyle as pas
from umami.configuration import global_config, logger
from umami.helper_tools import hist_ratio, hist_w_unc
from umami.plotting.plot_base import plot_base, plot_line_object
from umami.plotting.utils import get_good_pie_colours


class histogram(plot_line_object):
    """
    histogram class storing info about histogram and allows to calculate ratio w.r.t
    other histograms.
    """

    def __init__(
        self,
        values: np.ndarray,
        flavour: str = None,
        histtype: str = "step",
        **kwargs,
    ) -> None:
        """Initialise properties of histogram curve object.

        Parameters
        ----------
        values : np.ndarray
            Input data for the histogram
        flavour: str, optional
            Jet flavour in case the histogram corresponds to one specific flavour. If
            this is specified, the correct colour will be extracted from the global
            config. Allowed values are the ones from the global config, i.e. "bjets",
            "cjets", "ujets", "bbjets", ..., by default None
        histtype: str, optional
            `histtype` parameter which is handed to matplotlib.hist() when plotting the
            histograms. Supported values are "bar", "barstacked", "step", "stepfilled".
            By default "step"
        **kwargs : kwargs
            kwargs passed to `plot_line_object`

        Raises
        ------
        ValueError
            If input data is not of type np.ndarray or list
        """
        super().__init__(**kwargs)

        if isinstance(values, (np.ndarray, list, pd.core.series.Series)):
            values = np.array(values)
            if len(values) == 0:
                logger.warning("Histogram is empty.")
        else:
            raise ValueError(
                "Invalid type of histogram input data. Allowed values are "
                "numpy.ndarray, list, pandas.core.series.Series"
            )

        self.values = values
        self.flavour = flavour
        self.histtype = histtype

        # Set histogram attributes to None. They will be defined when the histograms
        # are plotted
        self.bin_edges = None
        self.hist = None
        self.unc = None
        self.band = None
        self.key = None

        self.label_addition = (
            kwargs["label"] if "label" in kwargs and kwargs["label"] is not None else ""
        )
        # If flavour was specified, extract configuration from global config
        if self.flavour is not None:
            self.colour = global_config.flavour_categories[self.flavour]["colour"]
            logger.debug(f"Histogram colour was set to {self.colour}")

            self.label = (
                f"{global_config.flavour_categories[self.flavour]['legend_label']}"
                f" {self.label_addition}"
            )
            logger.debug(f"Histogram label was set to {self.label}")

    def divide(self, other):
        """Calculate ratio between two class objects.

        Parameters
        ----------
        other : histogram class
            Second histogram object to calculate ratio with

        Returns
        -------
        np.ndarray
            ratio
        np.ndarray
            ratio error
        Raises
        ------
        ValueError
            If binning is not identical between 2 objects
        ValueError
            If hist attribute is not set for one of the two histograms
        ValueError
            If bin_edges attribute is not set for one of the two histograms
        """
        if self.bin_edges is None or other.bin_edges is None:
            raise ValueError(
                "Can't divide histograms since bin edges are not available "
                "for both histogram. Bins are filled when they are plotted."
            )

        if self.hist is None or other.hist is None:
            raise ValueError(
                "Can't divide histograms since bin counts are not available for both"
                "histograms. Bins are filled when they are plotted."
            )

        if not np.all(self.bin_edges == other.bin_edges):
            raise ValueError("The binning of the two given objects do not match.")

        # Bins where the reference histogram is empty/zero, are given a ratio of np.inf
        # which means that the ratio plot will not have any entry in these bins.
        ratio, ratio_unc = hist_ratio(
            numerator=self.hist,
            denominator=other.hist,
            numerator_unc=self.unc,
            denominator_unc=other.unc,
            step=False,
        )
        # To use the matplotlib.step() function later on, the first bin is duplicated
        ratio = np.append(np.array([ratio[0]]), ratio)
        ratio_unc = np.append(np.array([ratio_unc[0]]), ratio_unc)

        return (ratio, ratio_unc)


class histogram_plot(plot_base):
    """Histogram plot class"""

    def __init__(
        self,
        bins=40,
        bins_range: tuple = None,
        discrete_vals: list = None,
        plot_pie: bool = False,
        pie_colours: list = None,
        pie_labels: list = None,
        norm: bool = True,
        logy: bool = False,
        bin_width_in_ylabel: bool = False,
        **kwargs,
    ) -> None:
        """histogram plot properties

        Parameters
        ----------
        bins : int or numpy.ndarray or list, optional
            If bins is an int, it defines the number of equal-width bins in the given
            range. If bins is a sequence, it defines a monotonically increasing array
            of bin edges, including the rightmost edge, allowing for non-uniform
            bin widths (like in numpy.histogram). By default 40
        bins_range : tuple, optional
            Tuple of two floats, specifying the range for the binning. If bins_range is
            specified and bins is an integer, equal-width bins from bins_range[0] to
            bins_range[1] are used for the histogram (like in numpy.histogram).
            By default None
        discrete_vals : list, optional
            List of values if a variable only has discrete values. If discrete_vals is
            specified only the bins containing these values are plotted. If a pie chart
            is plotted, discrete_vals is obligatory. By default None.
        plot_pie : bool, optional
            Set to True if you would like to plot a pie chart instead of a histogram
        pie_colours : list or str, optional
            List of colours used for a pie chart or string providing the colour scheme.
            Possible colour schemes are 'red', 'blue', 'green' or 'yellow',
            by default None
        pie_labels : list, optional
            List of labels used for a pie chart, by default None
        norm : bool, optional
            Specify if the histograms are normalised, this means that histograms are
            divided by the total numer of counts. Therefore, the sum of the bin counts
            is equal to one, but NOT the area under the curve, which would be
            sum(bin_counts * bin_width). By default True.
        logy : bool, optional
            Set log scale on y-axis, by default False.
        bin_width_in_ylabel : bool, optional
            Specify if the bin width should be added to the ylabel, by default False
        **kwargs : kwargs
            kwargs from `plot_base`

        Raises
        ------
        ValueError
            If n_ratio_panels > 1
        """
        # TODO: use union operator `|` for multiple types of `bins` in python 3.10

        super().__init__(**kwargs)
        self.logy = logy
        self.bins = bins
        self.bins_range = bins_range
        self.discrete_vals = discrete_vals
        self.plot_pie = plot_pie
        self.pie_colours = pie_colours
        self.pie_labels = pie_labels
        self.bin_width_in_ylabel = bin_width_in_ylabel
        self.norm = norm
        self.plot_objects = {}
        self.add_order = []
        self.ratios_objects = {}
        self.ratio_axes = {}
        self.reference_object = None
        if self.n_ratio_panels > 1:
            raise ValueError("Not more than one ratio panel supported.")
        self.initialise_figure(sub_plot_index=6)

    def add(self, curve: object, key: str = None, reference: bool = False):
        """Adding histogram object to figure.

        Parameters
        ----------
        curve : histogram class
            histogram curve
        key : str, optional
            Unique identifier for histogram, by default None
        reference : bool, optional
            If this histogram is used as reference for ratio calculation, by default
            False

        Raises
        ------
        KeyError
            If unique identifier key is used twice
        """
        if key is None:
            key = len(self.plot_objects) + 1
        if key in self.plot_objects:
            raise KeyError(f"Duplicated key {key} already used for unique identifier.")

        # Add key to histogram object
        curve.key = key
        logger.debug(f"Adding histogram {key}")

        # Set linestyle
        if curve.linestyle is None:
            curve.linestyle = "-"
        # Set colours
        if curve.colour is None:
            curve.colour = pas.get_good_colours()[len(self.plot_objects)]
        # Set alpha
        if curve.alpha is None:
            curve.alpha = 0.8
        # Set linewidth
        if curve.linewidth is None:
            curve.linewidth = 1.6

        self.plot_objects[key] = curve
        self.add_order.append(key)
        if reference is True:
            self.set_reference(key)

    def set_reference(self, key: str):
        """Setting the reference histogram curves used in the ratios

        Parameters
        ----------
        key : str
            unique identifier of histogram object
        """
        if self.reference_object is None:
            self.reference_object = [key]
            logger.info(f"Using '{key}' as reference histogram")
        else:
            self.reference_object.append(key)
            logger.warning(
                f"You specified another curve {key} as reference for ratio. "
                "Adding it to reference histograms. "
                f"New list of reference histograms: {self.reference_object}"
            )

    def plot(self, **kwargs):
        """Plotting curves. This also generates the bins of the histograms that are
        added to the plot. Plot objects are drawn in the same order as they were added
        to the plot.

        Parameters
        ----------
        **kwargs: kwargs
            kwargs passed to matplotlib.axes.Axes.hist()

        Returns
        -------
        Line2D
            matplotlib Line2D object

        Raises
        ------
        ValueError
            If specified bins type is not supported.
        """
        plt_handles = []

        # Calculate bins of stacked histograms to ensure all histograms fit in plot
        if isinstance(self.bins, (np.ndarray, list)):
            logger.debug("Using bin edges defined in plot instance.")
            if self.bins_range is not None:
                logger.warning(
                    "You defined a range for the histogram, but also an array with "
                    "the bin edges. The range will be ignored."
                )
        elif isinstance(self.bins, int):
            logger.debug(f"Calculating bin edges of {self.bins} equal-width bins")
            _, self.bins = np.histogram(
                np.hstack([elem.values for elem in self.plot_objects.values()]),
                bins=self.bins,
                range=self.bins_range,
            )
        else:
            raise ValueError(
                "Unsupported type for bins. Supported types: int, numpy.array, list"
            )

        # Loop over all plot objects and plot them
        bins = self.bins
        for key in self.add_order:
            elem = self.plot_objects[key]

            elem.bin_edges, elem.hist, elem.unc, elem.band = hist_w_unc(
                elem.values,
                bins=self.bins,
                bins_range=self.bins_range,
                normed=self.norm,
            )

            if self.discrete_vals is not None:
                # bins are recalculated for the discrete values
                bins = self.get_discrete_values(elem)

            # Plot histogram
            self.axis_top.hist(
                x=bins[:-1],
                bins=bins,
                weights=elem.hist,
                histtype=elem.histtype,
                color=elem.colour,
                label=elem.label,
                alpha=elem.alpha,
                linewidth=elem.linewidth,
                linestyle=elem.linestyle,
                **kwargs,
            )

            # Plot histogram uncertainty
            if self.draw_errors:
                self.axis_top.hist(
                    x=bins[:-1],
                    bins=bins,
                    bottom=elem.band,
                    weights=elem.unc * 2,
                    **global_config.hist_err_style,
                )

            plt_handles.append(
                mpl.lines.Line2D(
                    [],
                    [],
                    color=elem.colour,
                    label=elem.label,
                    linestyle=elem.linestyle,
                )
            )

        if self.draw_errors:
            plt_handles.append(
                mpl.patches.Patch(
                    label="stat. uncertainty", **global_config.hist_err_style
                )
            )

        if self.discrete_vals is not None:
            self.bins = bins

        self.plotting_done = True
        return plt_handles

    def plot_pie_chart(self):
        """Plotting Pie chart. Only variables with defined discrete values
        can be plotted. Only non-zero values are plotted.

        Returns
        -------
        Patch
            matplotlib Patch object

        Raises
        ------
        ValueError
            If plot_pie is set to True but not the vertical_split
        ValueError
            If no discrete values of the variable are provided
        ValueError
            If specified bins type is not supported.
        """
        if self.vertical_split is False:
            raise ValueError(
                'To plot a pie chart "vertical_split" must be set to True.'
            )

        if self.discrete_vals is None:
            raise ValueError("To plot a donut diagram define discrete values")

        # Calculate bins of stacked histograms to ensure all histograms fit in plot
        if isinstance(self.bins, (np.ndarray, list)):
            logger.debug("Using bin edges defined in plot instance.")
            if self.bins_range is not None:
                logger.warning(
                    "You defined a range for the histogram, but also an array with "
                    "the bin edges. The range will be ignored."
                )
        elif isinstance(self.bins, int):
            logger.debug(f"Calculating bin edges of {self.bins} equal-width bins")
            _, self.bins = np.histogram(
                np.hstack([elem.values for elem in self.plot_objects.values()]),
                bins=self.bins,
                range=self.bins_range,
            )
        else:
            raise ValueError(
                "Unsupported type for bins. Supported types: int, numpy.array, list"
            )

        if len(self.plot_objects) > 1:
            logger.warning(
                "More than one histogram object was added but a pie chart should "
                "plotted. Only the first object is plotted. If you would like to "
                "plot multiple pie charts, create separate histogram_plot objects."
            )

        plt_handles = []
        elem = list(self.plot_objects.values())[0]

        elem.bin_edges, elem.hist, elem.unc, elem.band = hist_w_unc(
            elem.values,
            bins=self.bins,
            bins_range=self.bins_range,
            normed=self.norm,
        )

        # recalculated bins are not needed for pie charts
        _ = self.get_discrete_values(elem)

        indice = [i for i in range(len(elem.hist)) if elem.hist[i] != 0]
        elem.hist = elem.hist[indice]
        self.discrete_vals = np.array(self.discrete_vals)[indice]
        self.pie_labels = np.array(self.pie_labels)[indice]
        if not isinstance(self.pie_colours, list):
            self.pie_colours = get_good_pie_colours(colour_scheme=self.pie_colours)

        # make sure to have enough colours to represent every discrete value.
        # If the number of values is larger than the number of colours,
        # the same colours are repeated.
        if len(elem.hist) > len(self.pie_colours):
            logger.warning(
                "The number of non-zero histogram entries is larger than the "
                "number of available colours for the pie chart. Colours will be"
                "repeated."
            )

        self.pie_colours = (
            self.pie_colours * (len(elem.hist) // len(self.pie_colours) + 1)
        )[: len(elem.hist)]

        self.axis_top.pie(
            x=elem.hist, labels=None, colors=self.pie_colours, autopct="%1.1f%%"
        )

        self.axis_leg.axis("off")

        for pie_label, pie_colour in zip(self.pie_labels, self.pie_colours):
            plt_handles.append(
                mpl.patches.Patch(
                    label=pie_label, linestyle=elem.linestyle, color=pie_colour
                )
            )

        self.plotting_done = True
        return plt_handles

    def get_discrete_values(self, elem: object):
        """Get discrete values of a variable and adjust the
        bins accordingly

        Parameters
        ----------
        elem : histogram class
            Histogram we want to calculate the bins containing discrete values for

        Returns
        -------
        bins : recalculated bins including only the discrete values

        Raises
        ------
        ValueError
            If the bin width is larger than 1 such that potentially not
            all discrete values are in a seperate bin
        ValueError
            If the number of bins is set to 1 such that no values can be
            distinguished
        """

        if len(elem.bin_edges) > 1:
            if abs(elem.bin_edges[1] - elem.bin_edges[0]) <= 1:
                indice = []
                for i in range(len(elem.bin_edges) - 1):
                    # Only keep this bin edge if one of the discrete
                    # values is withing this and the next bin edge
                    for discrete_val in self.discrete_vals:
                        if elem.bin_edges[i] <= discrete_val < elem.bin_edges[i + 1]:
                            indice.append(i)
                elem.hist = elem.hist[indice]
                elem.unc = elem.unc[indice]
                elem.band = elem.band[indice]
                bins = np.linspace(
                    0, len(self.discrete_vals), len(self.discrete_vals) + 1
                )
                self.axis_top.set_xticks(bins[:-1] + 0.5)
                self.axis_top.set_xticklabels(self.discrete_vals, rotation=33)
            else:
                raise ValueError(
                    "Bin width is larger than 1. Choose a binning with a bin"
                    " width<= 1 to plot only discrete values."
                )
        else:
            raise ValueError(
                "Choose a binning with more than one bin in order to plot"
                "only discrete values."
            )

        return bins

    def get_reference_histo(self, histo):
        """Get reference histogram from list of references

        Parameters
        ----------
        histo : histogram class
            Histogram we want to calculate the ratio for

        Returns
        -------
        histogram class
            Reference histogram

        Raises
        ------
        ValueError
            If no reference histo was found or multiple matches.
        """

        matches = 0
        reference_histo = None

        for key in self.reference_object:
            reference_candidate = self.plot_objects[key]
            if histo.flavour is not None:
                if histo.flavour == reference_candidate.flavour:
                    matches += 1
                    reference_histo = reference_candidate
            else:
                matches += 1
                reference_histo = reference_candidate

        if matches != 1:
            raise ValueError("Found more than one matching reference candidate.")

        logger.debug(
            f"Reference histogram for '{histo.key}' is '{reference_histo.key}'"
        )

        return reference_histo

    def plot_ratios(self):
        """Plotting ratio histograms.

        Raises
        ------
        ValueError
            If no reference histogram is defined
        """
        if self.reference_object is None:
            raise ValueError("Please specify a reference curve.")

        for key in self.add_order:
            elem = self.plot_objects[key]

            if elem.bin_edges is None:
                raise ValueError(
                    "Bin edges of plot object not set. This is done in "
                    "histogram_plot.plot(), so it has to be called before "
                    "plot_ratios() is called."
                )

            ratio, ratio_unc = elem.divide(self.get_reference_histo(elem))

            ratio_unc_band_low = np.nan_to_num(ratio - ratio_unc, nan=0, posinf=0)
            ratio_unc_band_high = np.nan_to_num(ratio + ratio_unc, nan=0, posinf=0)

            # Plot the ratio values with the step function
            self.axis_ratio_1.step(
                x=elem.bin_edges,
                y=ratio,
                color=elem.colour,
                linewidth=elem.linewidth,
                linestyle=elem.linestyle,
            )

            # Plot the ratio uncertainty
            if self.draw_errors:
                self.axis_ratio_1.fill_between(
                    x=elem.bin_edges,
                    y1=ratio_unc_band_low,
                    y2=ratio_unc_band_high,
                    step="pre",
                    facecolor="none",
                    edgecolor=global_config.hist_err_style["edgecolor"],
                    linewidth=global_config.hist_err_style["linewidth"],
                    hatch=global_config.hist_err_style["hatch"],
                )

    def add_bin_width_to_ylabel(self):
        """Adds the bin width to the ylabel of a histogram plot. If the bin with is
        smaller than 0.01, scientific notation will be used.

        Raises
        ------
        ValueError
            If plotting_done is False (therefore `bins` is not yet calculated)
        """

        if self.plotting_done is False:
            raise ValueError(
                "`add_bin_width_to_ylabel` should be called after plotting, since bins "
                "are calculated during plotting."
            )

        bin_width = abs(self.bins[1] - self.bins[0])
        if bin_width < 1e-2:
            self.ylabel = f"{self.ylabel} / {bin_width:.0e}"
        else:
            self.ylabel = f"{self.ylabel} / {bin_width:.2f}"
        self.set_ylabel(self.axis_top)

    def draw(self, labelpad: int = None):
        """Draw figure.

        Parameters
        ----------
        labelpad : int, optional
            Spacing in points from the axes bounding box including
            ticks and tick labels, by default "ratio"

        """
        if self.plot_pie:
            plt_handles = self.plot_pie_chart()
        else:
            plt_handles = self.plot()

        if not self.plot_pie:
            if self.n_ratio_panels > 0:
                self.plot_ratios()

            self.set_xlim(
                self.bins[0] if self.xmin is None else self.xmin,
                self.bins[-1] if self.xmax is None else self.xmax,
            )
            self.set_logy()
            self.set_y_lim()
            self.set_xlabel()
            self.set_tick_params()
            self.set_ylabel(self.axis_top)

            if self.n_ratio_panels > 0:
                self.set_ylabel(
                    self.axis_ratio_1,
                    self.ylabel_ratio_1,
                    align_right=False,
                    labelpad=labelpad,
                )

            if self.bin_width_in_ylabel is True:
                self.add_bin_width_to_ylabel()

            legend_axis = self.axis_top
        else:
            legend_axis = self.axis_leg

        self.make_legend(plt_handles, ax=legend_axis)
        self.set_title()
        self.fig.tight_layout()

        if self.apply_atlas_style:
            self.atlasify()
