"""Fraction scan plot functions."""
import matplotlib as mpl
import numpy as np

from puma.plot_base import PlotBase, PlotLineObject
from puma.utils import get_good_colours, logger


class FractionScan(PlotLineObject):  # pylint: disable=too-few-public-methods
    """FractionScan class storing info about the fractions."""

    def __init__(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        **kwargs,
    ) -> None:
        """Initialise properties of fraction scan curve object.

        Parameters
        ----------
        x_values : np.ndarray
            x values of the fraction scan curve
        y_values : np.ndarray
            y values of the fraction scan curve
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
        if isinstance(x_values, (np.ndarray, list, int, float)):

            if type(x_values) != type(y_values):  # pylint: disable=C0123
                raise ValueError(
                    "Invalid types of input given! Both must be either "
                    "numpy.ndarray, list or int! \n"
                    f"You specified: x_values: {type(x_values)} , "
                    f"y_values: {type(y_values)}"
                )

            if isinstance(x_values, (int, float)) and isinstance(
                y_values, (int, float)
            ):
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
            raise ValueError(
                "Invalid type of fraction scan input data. Allowed values are "
                "numpy.ndarray, list, int, float"
            )

        # Set inputs as attributes
        self.x_values = x_values
        self.y_values = y_values

        # Set fraction scan attributes to None. Will be defined when plotting starts
        self.key = None


class FractionScanPlot(PlotBase):
    """Fraction scan plot class"""

    def __init__(
        self,
        logy: bool = False,
        **kwargs,
    ) -> None:
        """Fraction scan plot properties

        Parameters
        ----------
        logy : bool, optional
            Decide, if the y-axis of the plot will
            be in log, by default False
        **kwargs : kwargs
            kwargs from `plot_base`
        """

        super().__init__(**kwargs)

        # Set inputs as attributes
        self.logy = logy

        # Init needed lists and dicts
        self.plot_objects = {}
        self.add_order = []

        self.initialise_figure(sub_plot_index=6)

    def add(
        self,
        curve: object,
        key: str = None,
        is_marker: bool = False,
    ):
        """Adding FractionScan object to figure.

        Parameters
        ----------
        curve : FractionScan
            Fraction scan curve
        key : str, optional
            Unique identifier for FractionScan, by default None
        is_marker : bool, optional
            Defines if this is a marker (True) or a line (False).
            By default False.

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

        # Add key to FractionScan object
        curve.key = key
        logger.debug("Adding fraction scan of %s", key)

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
                curve.marker = "x"
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
            kwargs passed to matplotlib.axes.Axes.plot()

        Returns
        -------
        Line2D
            matplotlib Line2D object
        """
        plt_handles = []

        # Loop over all plot objects and plot them
        for key in self.add_order:
            elem = self.plot_objects[key]

            # Plot fraction scan
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

        # Plot the fraction scans and markers
        plt_handles = self.plot()

        # Make the legend
        self.make_legend(plt_handles, ax_mpl=self.axis_top)

        self.set_title()
        self.set_logy()
        self.set_y_lim()
        self.set_xlabel()
        self.set_tick_params()
        self.set_ylabel(self.axis_top)
        self.fig.tight_layout()

        # Set grid if grid is true
        if self.grid:
            self.axis_top.grid()

        # Apply atlas style if defined
        if self.apply_atlas_style:
            self.atlasify()
