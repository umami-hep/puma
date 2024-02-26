"""Classes for 2D line plots."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pandas as pd

from puma.plot_base import PlotBase, PlotLineObject
from puma.utils import get_good_colours, get_good_markers, logger


class Line2D(PlotLineObject):
    """Line2D class storing info about the x and y values and style."""

    def __init__(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        **kwargs,
    ) -> None:
        """Initialise properties of Line2D object.

        Parameters
        ----------
        x_values : np.ndarray
            x values of the curve
        y_values : np.ndarray
            y values of the curve
        **kwargs : kwargs
            kwargs passed to `PlotLineObject`

        Raises
        ------
        ValueError
            If the dtype of x_values and y_values is different.
        ValueError
            If provided x_values array is empty.
        ValueError
            If provided y_values array is empty.
        ValueError
            If provided x_values and y_values arrays have different
            shapes.
        ValueError
            If an invalid type was given for x_values
        """
        super().__init__(**kwargs)

        # Check input dtype
        if isinstance(x_values, (np.ndarray, list, int, float, pd.Series)):
            if type(x_values) != type(y_values):  # pylint: disable=C0123
                raise ValueError(
                    "Invalid types of input given! Both must be one of the following: "
                    "numpy.ndarray, list, int, float, pandas.Series \n"
                    f"You specified: x_values: {type(x_values)} , "
                    f"y_values: {type(y_values)}"
                )

            if isinstance(x_values, (int, float)) and isinstance(y_values, (int, float)):
                # Convert input into numpy array
                x_values = np.array([x_values])
                y_values = np.array([y_values])

            # Convert input into numpy array
            x_values = np.array(x_values)
            y_values = np.array(y_values)

            # Check that given arrays/lists are not empty
            if len(x_values) == 0:
                raise ValueError("Provided x_values is empty!")

            if len(y_values) == 0:
                raise ValueError("Provided y_values is empty!")

            # Check that both inputs have the same dimension
            if len(x_values) != len(y_values):
                raise ValueError(
                    "x_values and y_values have different dimensionalities! "
                    f"x_values: {len(x_values)}, y_values: {len(y_values)}"
                )

        else:
            raise TypeError(
                "Invalid type of input data. Allowed values are numpy.ndarray, list," " int, float"
            )

        # Set inputs as attributes
        self.x_values = x_values
        self.y_values = y_values

        # Set key to None. Will be defined when plotting starts
        self.key = None


class Line2DPlot(PlotBase):
    """Line2DPlot plot class for basic x-y line plots."""

    def __init__(
        self,
        logy: bool = False,
        grid: bool = True,
        **kwargs,
    ) -> None:
        """Plot properties.

        Parameters
        ----------
        logy : bool, optional
            Decide, if the y-axis of the plot will be in log, by default False
        grid : bool, optional
            Set the grid for the plots.
        **kwargs : kwargs
            Keyword arguments from `puma.PlotObject`
        """
        super().__init__(grid=grid, **kwargs)

        # Set inputs as attributes
        self.logy = logy

        # Init needed lists and dicts
        self.plot_objects = {}
        self.add_order = []

        self.initialise_figure()

    def add(
        self,
        curve: object,
        key: str | None = None,
        is_marker: bool = False,
    ):
        """Adding `puma.Line2D` object to figure.

        Parameters
        ----------
        curve : puma.line_plot_2D.Line2D
            Line2D object
        key : str, optional
            Unique identifier for the curve, by default None
        is_marker : bool, optional
            Defines if this is a marker (True) or a line (False). By default False.

        Raises
        ------
        KeyError
            If unique identifier key is used twice
        """
        # If key not defined, set it to a numerical value
        if key is None:
            key = len(self.plot_objects) + 1

        # Check that key is not double used
        if key in self.plot_objects:
            raise KeyError(f"Duplicated key {key} already used for unique identifier.")

        # Add key to Line2D object
        curve.key = key
        logger.debug("Adding Line2D object with key %s", key)

        # Set alpha
        if curve.alpha is None:
            curve.alpha = 1

        if is_marker is False:
            # Set colours
            if curve.colour is None:
                curve.colour = get_good_colours()[len(self.plot_objects)]
            curve.is_marker = False
            # Set linestyle
            if curve.linestyle is None:
                curve.linestyle = "-"
            # Set linewidth
            if curve.linewidth is None:
                curve.linewidth = 1.6

        else:
            curve.is_marker = True
            # Set colour of the marker the same as the last line plotted
            if curve.colour is None:
                curve.colour = self.plot_objects[len(self.plot_objects)].colour
            # Set markerstyle
            if curve.marker is None:
                curve.marker = get_good_markers()[len(self.plot_objects)]
            # Set markersize
            if curve.markersize is None:
                curve.markersize = 15
            # Set markersize
            if curve.markeredgewidth is None:
                curve.markeredgewidth = 2
            # Set the linestyle for markers to None as string
            # to suppress the line in the legend so only the marker
            # is shown in the legend
            curve.linestyle = "None"

        self.plot_objects[key] = curve
        self.add_order.append(key)

    def plot(self, **kwargs):
        """Plotting curves. Plot objects are drawn in the same order as they
        were added to the plot.

        Parameters
        ----------
        **kwargs: kwargs
            Keyword arguments passed to matplotlib.axes.Axes.plot()

        Returns
        -------
        Line2D
            matplotlib Line2D object
        """
        plt_handles = []

        # Loop over all plot objects and plot them
        for key in self.add_order:
            elem = self.plot_objects[key]

            self.axis_top.plot(
                elem.x_values,
                elem.y_values,
                color=elem.colour,
                label=elem.label,
                alpha=elem.alpha,
                linewidth=elem.linewidth,
                linestyle=elem.linestyle,
                marker=elem.marker,
                markersize=elem.markersize,
                markeredgewidth=elem.markeredgewidth,
                **kwargs,
            )

            plt_handles.append(
                mpl.lines.Line2D(
                    [],
                    [],
                    color=elem.colour,
                    label=elem.label,
                    linestyle=elem.linestyle,
                    marker=elem.marker,
                )
            )

        self.plotting_done = True
        return plt_handles

    def draw(self):
        """Draw figure."""
        plt_handles = self.plot()

        # Make the legend
        self.make_legend(plt_handles, ax_mpl=self.axis_top)

        self.set_title()
        self.set_log()
        self.set_y_lim()
        self.set_xlabel()
        self.set_tick_params()
        self.set_ylabel(self.axis_top)

        # Apply atlas style if defined
        if self.apply_atlas_style:
            self.atlasify()
