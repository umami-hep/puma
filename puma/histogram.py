"""Histogram plot functions."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pandas as pd
from ftag import Flavours, Label

from puma.plot_base import PlotBase, PlotLineObject
from puma.utils import get_good_colours, logger
from puma.utils.histogram import hist_ratio, hist_w_unc


class Histogram(PlotLineObject):
    """
    Histogram class storing info about histogram and allows to calculate ratio w.r.t
    other histograms.
    """

    def __init__(
        self,
        values: np.ndarray,
        weights: np.ndarray = None,
        bin_edges: np.ndarray = None,
        sum_squared_weights: np.ndarray = None,
        ratio_group: str | None = None,
        flavour: str | Label = None,
        add_flavour_label: bool = True,
        histtype: str = "step",
        is_data: bool = False,
        **kwargs,
    ) -> None:
        """Initialise properties of histogram curve object.

        Parameters
        ----------
        values : np.ndarray
            Input data for the histogram. If bin_edges is specified (not None)
            then this array is treated as the bin heights.
        weights : np.ndarray, optional
            Weights for the input data. Has to be an array of same length as the input
            data with a weight for each entry. If not specified, weight 1 will be given
            to each entry. The uncertainties are calculated as the square root of the
            squared weights (for each bin separately). By default None.
        bin_edges : np.ndarray, optional
            If specified, the histogram is considered "filled": the array given to
            values is treated as if it was the bin heights corresponding to these
            bin_edges and the "weights" input is ignored. By default None.
        sum_squared_weights : np.ndarray, optional
            Only considered if the histogram is considered filled (i.e bin_edges
            is specified). It is the sum_squared_weights per bin.
            By default None.
        ratio_group : str, optional
            Name of the ratio group this histogram is compared with. The ratio group
            allows you to compare different groups of histograms within one plot.
            By default None
        flavour: str | Label, optional
            If set, the correct colour and a label prefix will be extracted from
            `puma.utils.global_config` set for this histogram.
            Allowed values are e.g. "bjets", "cjets", "ujets", "bbjets", ...
            By default None
        add_flavour_label : bool, optional
            Set to False to suppress the automatic addition of the flavour label prefix
            in the label of the curve (i.e. "$b$-jets" in the case of b-jets).
            This is ignored if `flavour` is not set. By default True
        histtype: str, optional
            `histtype` parameter which is handed to matplotlib.hist() when plotting the
            histograms. Supported values are "bar", "barstacked", "step", "stepfilled".
            By default "step"
        is_data : bool, optional
            Decide, if the plot object will be treated as data (black dots,
            no stacking), by default False
        **kwargs : kwargs
            Keyword arguments passed to `puma.plot_base.PlotLineObject`

        Raises
        ------
        ValueError
            If input data is not of type np.ndarray or list
        ValueError
            If weights are specified but have different length as the input values
        """
        super().__init__(**kwargs)

        if isinstance(values, (np.ndarray, list, pd.core.series.Series)):
            values = np.array(values)
            if len(values) == 0:
                logger.warning("Histogram is empty.")
        else:
            raise TypeError(
                "Invalid type of histogram input data. Allowed values are "
                "numpy.ndarray, list, pandas.core.series.Series"
            )
        if weights is not None and len(values) != len(weights):
            raise ValueError("`values` and `weights` are not of same length.")

        self.values = values
        self.bin_edges = bin_edges  # Important to have this defined for any histogram
        self.sum_squared_weights = sum_squared_weights

        if bin_edges is None and sum_squared_weights is not None:
            logger.warning(
                """The Histogram has no bin edges defined and is thus
                              not considered filled. Parameter `sum_squared_weights`
                              is ignored. """
            )

        # This attribute allows to know how to handle the histogram later during
        # plotting
        self.filled = bin_edges is not None

        self.weights = weights
        self.ratio_group = ratio_group
        self.flavour = Flavours[flavour] if isinstance(flavour, str) else flavour
        self.add_flavour_label = add_flavour_label
        self.histtype = histtype
        self.is_data = is_data

        # Set histogram attributes to None. They will be defined when the histograms
        # are plotted
        self.hist = None
        self.unc = None
        self.band = None
        self.key = None

        label = kwargs["label"] if "label" in kwargs and kwargs["label"] is not None else ""
        # If flavour was specified, extract configuration from global config
        if self.flavour is not None:
            if self.flavour in Flavours:
                # Use globally defined flavour colour if not specified
                if self.colour is None:
                    self.colour = self.flavour.colour
                    logger.debug("Histogram colour was set to %s", self.colour)
                # Add globally defined flavour label if not suppressed
                if self.add_flavour_label:
                    global_flavour_label = self.flavour.label
                    self.label = f"{global_flavour_label} {label}"
                else:
                    self.label = label
                logger.debug("Histogram label was set to %s", {self.label})
            else:
                logger.warning("The flavour '%s' was not found in the global config.", self.flavour)

    def divide(self, other):
        """Calculate ratio between two class objects.

        Parameters
        ----------
        other : histogram class
            Second histogram object to calculate ratio with

        Returns
        -------
        np.ndarray
            Ratio
        np.ndarray
            Ratio error

        Raises
        ------
        ValueError
            If binning is not identical between 2 objects
        ValueError
            If hist attribute is not set for one of the two histograms
        ValueError
            If bin_edges attribute is not set for one of the two histograms
        """
        if (
            self.bin_edges is None
            or other.bin_edges is None
            or self.hist is None
            or other.hist is None
        ):
            raise ValueError(
                "Can't divide histograms since bin edges and counts are not available "
                "for both histogram. Bins are filled when they are plotted."
            )

        if not np.all(self.bin_edges == other.bin_edges):
            raise ValueError("The binning of the two given objects do not match.")

        # Bins where the reference histogram is empty/zero, are given a ratio of np.inf
        # which means that the ratio plot will not have any entry in these bins.
        ratio, ratio_unc = hist_ratio(
            numerator=self.hist,
            denominator=other.hist,
            numerator_unc=self.unc,
            step=False,
        )
        # To use the matplotlib.step() function later on, the first bin is duplicated
        ratio = np.append(np.array([ratio[0]]), ratio)
        ratio_unc = np.append(np.array([ratio_unc[0]]), ratio_unc)

        return (ratio, ratio_unc)

    def divide_data_mc(
        self,
        ref_hist: np.ndarray,
    ) -> tuple:
        """
        Similar as divide, but the second item doesn't need to be a histogram object.

        Parameters
        ----------
        ref_hist : np.ndarray
            Hist weights of the reference.
        ref_unc : np.ndarray
            Uncertainties of the reference

        Returns
        -------
        tuple
            Tuple of the ratios and ratio uncertaintes for the bins
        """
        # Bins where the reference histogram is empty/zero, are given a ratio of np.inf
        # which means that the ratio plot will not have any entry in these bins.
        ratio, ratio_unc = hist_ratio(
            numerator=self.hist,
            denominator=ref_hist,
            numerator_unc=self.unc,
            step=False,
        )
        # To use the matplotlib.step() function later on, the first bin is duplicated
        ratio = np.append(np.array([ratio[0]]), ratio)
        ratio_unc = np.append(np.array([ratio_unc[0]]), ratio_unc)

        return (ratio, ratio_unc)


class HistogramPlot(PlotBase):
    """Histogram plot class."""

    def __init__(
        self,
        bins=40,
        bins_range: tuple | None = None,
        discrete_vals: list | None = None,
        norm: bool = True,
        logy: bool = False,
        bin_width_in_ylabel: bool = False,
        underoverflow: bool = True,
        grid: bool = False,
        stacked: bool = False,
        histtype: str = "bar",
        **kwargs,
    ) -> None:
        """Histogram plot properties.

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
            specified only the bins containing these values are plotted.
            By default None.
        norm : bool, optional
            Specify if the histograms are normalised, this means that histograms are
            divided by the total numer of counts. Therefore, the sum of the bin counts
            is equal to one, but NOT the area under the curve, which would be
            sum(bin_counts * bin_width). By default True.
        logy : bool, optional
            Set log scale on y-axis, by default False.
        bin_width_in_ylabel : bool, optional
            Specify if the bin width should be added to the ylabel, by default False
        underoverflow : bool, optional
            Option to include under- and overflow values in outermost bins, by default
            True.
        grid : bool, optional
            Set the grid for the plots, by default False
        stacked : bool, optional
            Decide, if all histograms (which are not data) are stacked, by default False
        histtype : str, optional
            If stacked is used, define the type of histogram you would like to have,
            default is "bar"
        **kwargs : kwargs
            Keyword arguments from `puma.PlotObject`

        Raises
        ------
        ValueError
            If n_ratio_panels > 1
        """
        super().__init__(grid=grid, **kwargs)
        self.logy = logy
        self.bins = bins
        self.bins_range = bins_range
        self.discrete_vals = discrete_vals
        self.bin_width_in_ylabel = bin_width_in_ylabel
        self.norm = norm
        self.underoverflow = underoverflow
        self.stacked = stacked
        self.histtype = histtype
        self.plot_objects = {}
        self.add_order = []
        self.ratios_objects = {}
        self.reference_object = None

        if self.norm is True and self.stacked is True:
            raise ValueError(
                "Stacked plots and normalised plots at the same time are not available."
            )

        if self.n_ratio_panels > 1:
            raise ValueError("Not more than one ratio panel supported.")
        self.initialise_figure()

    def add(
        self,
        histogram: Histogram,
        key: str | None = None,
        reference: bool = False,
    ):
        """Adding histogram object to figure.

        Parameters
        ----------
        histogram : Histogram class
            Histogram curve
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
        histogram.key = key
        logger.debug("Adding histogram %s", key)

        # Set linestyle
        if histogram.linestyle is None:
            if histogram.is_data is True:
                histogram.linestyle = ""
            else:
                histogram.linestyle = "-"
        # Set marker
        if histogram.marker is None:
            if histogram.is_data is True:
                histogram.marker = "."
            else:
                histogram.marker = ""
        # Set colours
        if histogram.colour is None:
            histogram.colour = get_good_colours()[len(self.plot_objects)]
        # Set alpha
        if histogram.alpha is None:
            histogram.alpha = 1
        # Set linewidth
        if histogram.linewidth is None:
            histogram.linewidth = 1.6
        # Set markersize
        if histogram.markersize is None:
            histogram.markersize = 10

        self.plot_objects[key] = histogram
        self.add_order.append(key)
        if reference is True:
            self.set_reference(key)

    def set_reference(self, key: str):
        """Setting the reference histogram curves used in the ratios.

        Parameters
        ----------
        key : str
            Unique identifier of histogram object
        """
        if self.reference_object is None:
            self.reference_object = [key]
        else:
            self.reference_object.append(key)
        logger.debug("Adding '%s' to reference histogram(s)", key)

    def plot(self, **kwargs):
        """Plotting curves. This also generates the bins of the histograms that are
        added to the plot. Plot objects are drawn in the same order as they were added
        to the plot.

        Parameters
        ----------
        **kwargs: kwargs
            Keyword arguments passed to matplotlib.axes.Axes.hist()

        Returns
        -------
        Line2D
            matplotlib Line2D object

        Raises
        ------
        ValueError
            If specified bins type is not supported.
        """
        if self.ylabel is not None and self.norm and "norm" not in self.ylabel.lower():
            logger.warning(
                "You are plotting normalised distributions but 'norm' is not "
                "included in your y-label."
            )
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
            logger.debug("Calculating bin edges of %i equal-width bins", self.bins)
            _, self.bins = np.histogram(
                np.hstack([elem.values for elem in self.plot_objects.values()]),  # noqa: PD011
                bins=self.bins,
                range=self.bins_range,
            )
        else:
            raise TypeError("Unsupported type for bins. Supported types: int, numpy.array, list")

        # Loop over all plot objects and plot them
        bins = self.bins

        # Stacked dict for the stacked histogram
        self.stacked_dict = {
            "x": [],
            "weights": [],
            "color": [],
            "unc": None,
        }

        for key in self.add_order:
            elem = self.plot_objects[key]

            elem.bin_edges, elem.hist, elem.unc, elem.band = hist_w_unc(
                elem.values,
                weights=elem.weights,
                bin_edges=elem.bin_edges,
                sum_squared_weights=elem.sum_squared_weights,
                bins=self.bins,
                filled=elem.filled,
                bins_range=self.bins_range,
                normed=self.norm,
                underoverflow=self.underoverflow,
            )

            # MAYBE CHECK HERE THAT self.bins and elem.bin_edges are
            # equivalent for plotting or throw error!

            if self.discrete_vals is not None:
                # bins are recalculated for the discrete values
                bins = self.get_discrete_values(elem)

            # Check if the histogram is data
            if elem.is_data is True:
                # Plot data
                self.axis_top.errorbar(
                    x=(bins[:-1] + bins[1:]) / 2,
                    y=elem.hist,
                    yerr=elem.unc if self.draw_errors else 0,
                    color=elem.colour,
                    label=elem.label,
                    alpha=elem.alpha,
                    linewidth=elem.linewidth,
                    linestyle=elem.linestyle,
                    marker=elem.marker,
                    markersize=elem.markersize,
                )

                plt_handles.append(
                    mpl.lines.Line2D(
                        [],
                        [],
                        color=elem.colour,
                        label=elem.label,
                        alpha=elem.alpha,
                        linewidth=elem.linewidth,
                        linestyle=elem.linestyle,
                        marker=elem.marker,
                    )
                )

            elif self.stacked:
                self.stacked_dict["x"].append(bins[:-1])
                self.stacked_dict["weights"].append(elem.hist)
                self.stacked_dict["color"].append(elem.colour)

                if self.stacked_dict["unc"] is None:
                    self.stacked_dict["unc"] = elem.unc

                else:
                    self.stacked_dict["unc"] = np.sqrt(self.stacked_dict["unc"] ** 2 + elem.unc**2)

                # Add the element to the legend with a "bar"
                plt_handles.append(
                    mpl.patches.Patch(
                        color=elem.colour,
                        label=elem.label,
                        alpha=elem.alpha,
                    )
                )

            else:
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
                    bottom_error = np.array([elem.band[0], *elem.band.tolist()])
                    top_error = elem.band + 2 * elem.unc
                    top_error = np.array([top_error[0], *top_error.tolist()])
                    self.axis_top.fill_between(
                        x=elem.bin_edges,
                        y1=bottom_error,
                        y2=top_error,
                        color=elem.colour,
                        alpha=0.3,
                        zorder=1,
                        step="pre",
                        edgecolor="none",
                    )

                # Add standard "Line" to legend
                plt_handles.append(
                    mpl.lines.Line2D(
                        [],
                        [],
                        color=elem.colour,
                        label=elem.label,
                        alpha=elem.alpha,
                        linewidth=elem.linewidth,
                        linestyle=elem.linestyle,
                        marker=elem.marker,
                    )
                )

        if self.stacked:
            self.axis_top.hist(
                x=self.stacked_dict["x"],
                bins=bins,
                weights=self.stacked_dict["weights"],
                color=self.stacked_dict["color"],
                histtype=self.histtype,
                alpha=elem.alpha,
                linewidth=elem.linewidth,
                linestyle=elem.linestyle,
                stacked=self.stacked,
                **kwargs,
            )

            # Create a total weights entry to correctly plot the ratio
            # Total weights is here the y-value of all contributions stacked
            self.stacked_dict["total_weights"] = np.sum(self.stacked_dict["weights"], axis=0)

        # Check if errors should be drawn
        # If stacked is true, plot the combined uncertainty
        if self.draw_errors and self.stacked:
            # Calculate the y-values of the bottom error
            bottom_error = self.stacked_dict["total_weights"] - self.stacked_dict["unc"]
            bottom_error = np.array([bottom_error[0], *bottom_error.tolist()])

            # Calculate the y-values of the top error
            top_error = self.stacked_dict["total_weights"] + self.stacked_dict["unc"]
            top_error = np.array([top_error[0], *top_error.tolist()])

            # Fill the space between bottom and top with the unc. band
            self.axis_top.fill_between(
                x=elem.bin_edges,
                y1=bottom_error,
                y2=top_error,
                alpha=0.5,
                zorder=1,
                step="pre",
                facecolor="white",
                edgecolor="black",
                linewidth=0,
                hatch="/////",
            )

            # Add a label for the unc. in the legend
            plt_handles.append(
                mpl.patches.Patch(
                    facecolor="white",
                    edgecolor="black",
                    label="Stat. unc.",
                    linewidth=0,
                    alpha=0.5,
                    hatch="/////",
                )
            )

        if self.discrete_vals is not None:
            self.bins = bins

        self.plotting_done = True
        return plt_handles

    def get_discrete_values(self, elem: object):
        """Get discrete values of a variable and adjust the
        bins accordingly.

        Parameters
        ----------
        elem : histogram class
            Histogram we want to calculate the bins containing discrete values for

        Returns
        -------
        bins : numpy.ndarray
            Recalculated bins including only the discrete values

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
                indice = [
                    i
                    for i in range(len(elem.bin_edges) - 1)
                    for discrete_val in self.discrete_vals
                    if elem.bin_edges[i] <= discrete_val < elem.bin_edges[i + 1]
                ]
                elem.hist = elem.hist[indice]
                elem.unc = elem.unc[indice]
                elem.band = elem.band[indice]
                bins = np.linspace(0, len(self.discrete_vals), len(self.discrete_vals) + 1)
                elem.bin_edges = bins
                self.axis_top.set_xticks(bins[:-1] + 0.5)
                self.axis_top.set_xticklabels(self.discrete_vals, rotation=33)
            else:
                raise ValueError(
                    "Bin width is larger than 1. Choose a binning with a bin"
                    " width<= 1 to plot only discrete values."
                )
        else:
            raise ValueError(
                "Choose a binning with more than one bin in order to plot only discrete" " values."
            )

        return bins

    def get_reference_histo(self, histo):
        """Get reference histogram from list of references.

        Parameters
        ----------
        histo : puma.histogram.Histogram
            Histogram we want to calculate the ratio for

        Returns
        -------
        reference_histo_name : str, int
            Identifier of the corresponding reference histogram

        Raises
        ------
        ValueError
            If no reference histo was found or multiple matches.
        """
        matches = 0
        reference_histo = None

        for key in self.reference_object:
            reference_candidate = self.plot_objects[key]
            if histo.ratio_group is not None:
                if histo.ratio_group == reference_candidate.ratio_group:
                    matches += 1
                    reference_histo = reference_candidate
            else:
                matches += 1
                reference_histo = reference_candidate

        if matches != 1:
            raise ValueError(
                f"Found {matches} matching reference candidates, but only one match is" " allowed."
            )

        logger.debug("Reference histogram for '%s' is '%s'", histo.key, reference_histo.key)

        return reference_histo

    def plot_ratios(self):
        """Plotting ratio histograms.

        Raises
        ------
        ValueError
            If no reference histogram is defined
        """
        # Check if this is a stacked plot
        # Plot ratio only between data and the stacked histos
        for key in self.add_order:
            # Get the object which is to be plotted
            elem = self.plot_objects[key]

            if elem.bin_edges is None:
                raise ValueError(
                    "Bin edges of plot object not set. This is done in "
                    "histogram_plot.plot(), so it has to be called before "
                    "plot_ratios() is called."
                )

            # Check if this is going to be Data/MC (Data/stacked plot)
            if self.stacked:
                # Check this is data
                if not elem.is_data:
                    continue

                # Using the total weights (full stacked histo) as reference for data
                ratio, ratio_unc = elem.divide_data_mc(
                    ref_hist=self.stacked_dict["total_weights"],
                )

            else:
                if self.reference_object is None:
                    raise ValueError("Please specify a reference curve.")

                ratio, ratio_unc = elem.divide(self.get_reference_histo(elem))

            # Plot the ratio values with the step function
            if self.stacked:
                if elem.is_data is True:
                    self.ratio_axes[0].errorbar(
                        x=(elem.bin_edges[:-1] + elem.bin_edges[1:]) / 2,
                        y=ratio[1:],
                        yerr=ratio_unc[1:] if self.draw_errors else 0,
                        color=elem.colour,
                        label=elem.label,
                        alpha=elem.alpha,
                        linewidth=elem.linewidth,
                        linestyle=elem.linestyle,
                        marker=elem.marker,
                        markersize=elem.markersize,
                    )

            else:
                self.ratio_axes[0].step(
                    x=elem.bin_edges,
                    y=ratio,
                    color=elem.colour,
                    linewidth=elem.linewidth,
                    linestyle=elem.linestyle,
                )

                # Plot the ratio uncertainty
                if self.draw_errors:
                    self.ratio_axes[0].fill_between(
                        x=elem.bin_edges,
                        y1=np.nan_to_num(ratio - ratio_unc, nan=0, posinf=0),
                        y2=np.nan_to_num(ratio + ratio_unc, nan=0, posinf=0),
                        color=elem.colour,
                        alpha=0.3,
                        zorder=1,
                        step="pre",
                        edgecolor="none",
                    )

        if self.stacked and self.draw_errors:
            self.ratio_axes[0].fill_between(
                x=elem.bin_edges,
                y1=np.nan_to_num((ratio - ratio_unc) / ratio, nan=0, posinf=0),
                y2=np.nan_to_num((ratio + ratio_unc) / ratio, nan=0, posinf=0),
                color=elem.colour,
                alpha=0.3,
                zorder=1,
                step="pre",
                edgecolor="none",
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

    def draw(self, labelpad: int | None = None):
        """Draw figure.

        Parameters
        ----------
        labelpad : int, optional
            Spacing in points from the axes bounding box including
            ticks and tick labels, by default "ratio"

        """
        plt_handles = self.plot()

        if self.n_ratio_panels > 0:
            self.plot_ratios()

        self.set_xlim(
            self.bins[0] if self.xmin is None else self.xmin,
            self.bins[-1] if self.xmax is None else self.xmax,
        )
        self.set_log()
        self.set_y_lim()
        self.set_xlabel()
        self.set_tick_params()
        self.set_ylabel(self.axis_top)

        if self.n_ratio_panels > 0:
            self.set_ylabel(
                self.ratio_axes[0],
                self.ylabel_ratio[0],
                align="center",
                labelpad=labelpad,
            )

        if self.bin_width_in_ylabel is True:
            self.add_bin_width_to_ylabel()

        legend_axis = self.axis_top

        self.make_legend(plt_handles, ax_mpl=legend_axis)
        self.set_title()

        if self.apply_atlas_style:
            self.atlasify()
