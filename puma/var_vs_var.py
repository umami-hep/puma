"""Variable vs another variable plot"""
import matplotlib as mpl
import numpy as np

# TODO: fix the import below
from puma.plot_base import PlotBase, PlotLineObject
from puma.utils import get_good_colours, logger
from puma.utils.histogram import hist_ratio


class VarVsVar(PlotLineObject):  # pylint: disable=too-many-instance-attributes
    """
    var_vs_eff class storing info about curve and allows to calculate ratio w.r.t other
    efficiency plots.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        x_var_mean: np.ndarray,
        y_var_mean: np.ndarray,
        y_var_std: np.ndarray,
        x_var_widths: np.ndarray = None,
        key: str = None,
        **kwargs,
    ) -> None:
        """Initialise properties of roc curve object.

        Parameters
        ----------
        x_var_mean : np.ndarray
            Values for x-axis variable
        y_var_mean : np.ndarray
            Mean value for y-axis variable
        y_var_std : np.ndarray
            Std value for y-axis variable
        x_var_widths : np.ndarray, optional
            Std value for x-axis variable
        key : str, optional
            Identifier for the curve e.g. tagger, by default None
        **kwargs : kwargs
            Keyword arguments passed to `PlotLineObject`

        Raises
        ------
        ValueError
            If provided options are not compatible with each other
        """

        super().__init__(**kwargs)
        if len(x_var_mean) != len(y_var_mean):
            raise ValueError(
                f"Length of `x_var_mean` ({len(x_var_mean)}) and `y_var_mean` "
                f"({len(y_var_mean)}) have to be identical."
            )
        if len(x_var_mean) != len(y_var_std):
            raise ValueError(
                f"Length of `x_var_mean` ({len(x_var_mean)}) and `y_var_std` "
                f"({len(y_var_std)}) have to be identical."
            )

        self.x_var_mean = np.array(x_var_mean)
        self.x_var_widths = None if x_var_widths is None else np.array(x_var_widths)
        self.y_var_mean = np.array(y_var_mean)
        self.y_var_std = np.array(y_var_std)
        self.key = key

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                np.all(self.x_var_mean == other.x_var_mean)
                and np.all(self.y_var_mean == other.y_var_mean)
                and np.all(self.y_var_std == other.y_var_std)
                and self.key == other.key
            )
        return False

    def divide(self, other, inverse: bool = False):
        """Calculate ratio between two class objects.

        Parameters
        ----------
        other : var_vs_var class
            Second var_vs_var object to calculate ratio with
        inverse : bool
            If False the ratio is calculated `this / other`,
            if True the inverse is calculated

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
        """
        if not np.array_equal(self.x_var_mean, other.x_var_mean):
            raise ValueError("The x variables of the two given objects do not match.")
        # TODO: python 3.10 switch to cases syntax
        nom, nom_err = self.y_var_mean, self.y_var_std
        denom, denom_err = other.y_var_mean, other.y_var_std

        ratio, ratio_err = hist_ratio(
            denom if inverse else nom,
            nom if inverse else denom,
            denom_err if inverse else nom_err,
            nom_err if inverse else denom_err,
            step=False,
        )
        return (ratio, ratio_err)


class VarVsVarPlot(PlotBase):  # pylint: disable=too-many-instance-attributes
    """var_vs_eff plot class"""

    def __init__(self, grid: bool = False, **kwargs) -> None:
        """var_vs_eff plot properties

        Parameters
        ----------
        grid : bool, optional
            Set the grid for the plots.
        **kwargs : kwargs
            Keyword arguments from `puma.PlotObject`

        Raises
        ------
        ValueError
            If incompatible mode given or more than 1 ratio panel requested
        """
        super().__init__(grid=grid, **kwargs)

        self.plot_objects = {}
        self.add_order = []
        self.ratios_objects = {}
        self.reference_object = None
        self.x_var_min = np.inf
        self.x_var_max = -np.inf
        self.inverse_cut = False
        if self.n_ratio_panels > 1:
            raise ValueError("Not more than one ratio panel supported.")
        self.initialise_figure()

    def add(self, curve: object, key: str = None, reference: bool = False):
        """Adding var_vs_eff object to figure.

        Parameters
        ----------
        curve : var_vs_eff class
            Var_vs_eff curve
        key : str, optional
            Unique identifier for var_vs_eff, by default None
        reference : bool, optional
            If var_vs_eff is used as reference for ratio calculation, by default False

        Raises
        ------
        KeyError
            If unique identifier key is used twice
        """
        if key is None:
            key = len(self.plot_objects) + 1
        if key in self.plot_objects:
            raise KeyError(f"Duplicated key {key} already used for unique identifier.")

        self.plot_objects[key] = curve
        self.add_order.append(key)
        # set linestyle
        if curve.linestyle is None:
            curve.linestyle = "-"
        # set colours
        if curve.colour is None:
            curve.colour = get_good_colours()[len(self.plot_objects) - 1]
        # set alpha
        if curve.alpha is None:
            curve.alpha = 0.8
        # set linewidth
        if curve.linewidth is None:
            curve.linewidth = 1.6

        # set min and max bin edges
        self.x_var_min = min(self.x_var_min, np.sort(curve.x_var_mean)[0])
        self.x_var_max = max(self.x_var_max, np.sort(curve.x_var_mean)[-1])

        if reference:
            logger.debug("Setting roc %s as reference.", key)
            self.set_reference(key)

    def set_reference(self, key: str):
        """Setting the reference roc curves used in the ratios

        Parameters
        ----------
        key : str
            Unique identifier of roc object
        """
        if self.reference_object is None:
            self.reference_object = key
        else:
            logger.warning(
                (
                    "You specified a second curve %s as reference for ratio. "
                    "Using it as new reference instead of %s."
                ),
                key,
                self.reference_object,
            )
            self.reference_object = key

    def plot(self, **kwargs):
        """Plotting curves

        Parameters
        ----------
        **kwargs: kwargs
            Keyword arguments passed to plt.axis.errorbar

        Returns
        -------
        Line2D
            matplotlib Line2D object
        """
        logger.debug("Plotting curves")
        plt_handles = []
        for key in self.add_order:
            elem = self.plot_objects[key]
            error_bar = self.axis_top.errorbar(
                elem.x_var_mean,
                elem.y_var_mean,
                xerr=elem.x_var_widths / 2,
                yerr=elem.y_var_std,
                color=elem.colour,
                fmt="none",
                label=elem.label,
                alpha=elem.alpha,
                linewidth=elem.linewidth,
                **kwargs,
            )
            # # set linestyle for errorbar
            error_bar[-1][0].set_linestyle(elem.linestyle)

            # Draw markers
            self.axis_top.scatter(
                x=elem.x_var_mean,
                y=elem.y_var_mean,
                marker=elem.marker,
                color=elem.colour,
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
        return plt_handles

    def plot_ratios(self):
        """Plotting ratio curves.

        Raises
        ------
        ValueError
            If no reference curve is defined
        """
        if self.reference_object is None:
            raise ValueError("Please specify a reference curve.")
        for key in self.add_order:
            elem = self.plot_objects[key]
            (ratio, ratio_err) = elem.divide(self.plot_objects[self.reference_object])
            error_bar = self.ratio_axes[0].errorbar(
                elem.x_var_mean,
                ratio,
                xerr=elem.x_var_widths / 2,
                yerr=ratio_err,
                color=elem.colour,
                fmt="none",
                alpha=elem.alpha,
                linewidth=elem.linewidth,
            )
            # set linestyle for errorbar
            error_bar[-1][0].set_linestyle(elem.linestyle)

            self.ratio_axes[0].scatter(
                x=elem.x_var_mean, y=ratio, marker=elem.marker, color=elem.colour
            )

    def draw_hline(self, y_val: float):
        """Draw hline in top plot panel.

        Parameters
        ----------
        y_val : float
            y value of the horizontal line
        """
        self.axis_top.hlines(
            y=y_val,
            xmin=self.x_var_min,
            xmax=self.x_var_max,
            colors="black",
            linestyle="dotted",
            alpha=0.5,
        )

    def draw(
        self,
        labelpad: int = None,
    ):
        """Draw figure.

        Parameters
        ----------
        labelpad : int, optional
            Spacing in points from the axes bounding box including
            ticks and tick labels, by default "ratio"
        """
        self.set_xlim(
            self.x_var_min if self.xmin is None else self.xmin,
            self.x_var_max if self.xmax is None else self.xmax,
        )
        plt_handles = self.plot()
        if self.n_ratio_panels == 1:
            self.plot_ratios()
        self.set_title()
        self.set_log()
        self.set_y_lim()
        self.set_xlabel()
        self.set_tick_params()
        self.set_ylabel(self.axis_top)

        if self.n_ratio_panels > 0:
            self.set_ylabel(
                self.ratio_axes[0],
                self.ylabel_ratio[0],
                align_right=False,
                labelpad=labelpad,
            )
        self.make_legend(plt_handles, ax_mpl=self.axis_top)
        self.plotting_done = True
        if self.apply_atlas_style is True:
            self.atlasify(use_tag=self.use_atlas_tag)
