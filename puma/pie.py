"""Pie plot functions."""

from __future__ import annotations

import matplotlib as mpl

from puma.plot_base import PlotBase
from puma.utils import get_good_pie_colours, logger


class PiePlot(PlotBase):
    """Pie plot class."""

    def __init__(
        self,
        wedge_sizes,
        colours: list | None = None,
        colour_scheme: str | None = None,
        labels: list | None = None,
        draw_legend: bool = False,
        mpl_pie_kwargs: dict | None = None,
        **kwargs,
    ):
        """Initialise the pie plot.

        Parameters
        ----------
        wedge_sizes : 1D array like
            The size of the wedges. Will be translated into the fractions automatically.
            So they don't have to add up to 1 or 100. The fractional area of each
            wedge is given by x/sum(x).
        colours : list, optional
            List of colours for the separate wedges. You have to specify as many
            colours as you have wedges. Instead, you can also specify a colour scheme
            with the `colour_scheme` argument, by default None
        colour_scheme : str, optional
            Name of the colour schemes as defined in puma.utils.get_good_pie_colours,
            by default None
        labels : list, optional
            A sequence of strings providing the labels for each wedge, by default None
        draw_legend : bool, optional
            If True, a legend will be drawn on the right side of the plot.
            If False, the labels will be drawn directly to the wedges. By default True
        mpl_pie_kwargs : dict, optional
            Keyword arguments that are handed to the matplotlib.pyplot.pie function.
            All arguments are allowed, except [`x`, `labels`, `colors`], by default None
        **kwargs : kwargs
            Keyword arguments from `puma.PlotObject`
        """
        super().__init__(vertical_split=draw_legend, **kwargs)
        self.wedge_sizes = wedge_sizes
        self.draw_legend = draw_legend

        # set colours
        if colours is not None:
            logger.info("Using specified colours list.")
            self.colours = colours
        else:
            # Using one of the colour schemes defined in puma.utils.get_good_pie_colours
            logger.info("Using specified colour scheme (%s).", colour_scheme)
            self.colours = get_good_pie_colours(colour_scheme)[: len(wedge_sizes)]

        self.labels = labels if labels is not None else ["" for i in range(len(wedge_sizes))]

        # Add some good defaults if not specified:
        self.mpl_pie_kwargs = {
            "autopct": "%1.1f%%",
            "startangle": 90,
        }
        # If mpl.pie kwargs were specified, overwrite the defaults
        if mpl_pie_kwargs is not None:
            for key, value in mpl_pie_kwargs.items():
                self.mpl_pie_kwargs[key] = value

        self.initialise_figure()
        self.plot()

    def plot(
        self,
    ):
        """Plot the pie chart."""
        self.axis_top.pie(
            x=self.wedge_sizes,
            labels=None if self.draw_legend else self.labels,
            colors=self.colours,
            **self.mpl_pie_kwargs,
        )

        # If the legend should be drawn, get the handles and plot it on the right axis
        if self.draw_legend:
            plt_handles = []
            for pie_label, pie_colour in zip(self.labels, self.colours):
                plt_handles.append(
                    mpl.patches.Patch(
                        label=pie_label,
                        color=pie_colour,
                    )
                )
            self.axis_leg.axis("off")
            self.make_legend(plt_handles, ax_mpl=self.axis_leg)
        else:
            self.axis_top.axis("equal")

        self.plotting_done = True

        if self.apply_atlas_style:
            self.atlasify()
            # Remove the legend that is automatically created by atlasify
            if not self.draw_legend:
                self.axis_top.legend().remove()

        self.set_title()
        self.set_y_lim()
