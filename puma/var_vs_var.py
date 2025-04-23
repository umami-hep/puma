"""Variable vs another variable plot."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.patches import Rectangle

from puma.plot_base import PlotBase, PlotLineObject
from puma.utils import get_good_colours, get_good_markers, logger
from puma.utils.histogram import hist_ratio


class VarVsVar(PlotLineObject):
    """
    VarVsVar class storing info about curve and allows to calculate ratio w.r.t other
    efficiency plots.
    """

    def __init__(
        self,
        x_var: np.ndarray,
        y_var_mean: np.ndarray,
        y_var_std: np.ndarray,
        x_var_widths: np.ndarray = None,
        key: str | None = None,
        fill: bool = True,
        plot_y_std: bool = True,
        ratio_group: str | None = None,
        **kwargs,
    ) -> None:
        """Initialise properties of VarVsVar curve object.

        Parameters
        ----------
        x_var : np.ndarray
            Values for x-axis variable, e.g. bin midpoints for binned data
        y_var_mean : np.ndarray
            Mean value for y-axis variable
        y_var_std : np.ndarray
            Std value for y-axis variable
        x_var_widths : np.ndarray, optional
            Widths for x-axis variable, e.g. bin widths for binned data
        key : str, optional
            Identifier for the curve e.g. tagger, by default None
        fill : bool, optional
            Defines do we need to fill box around point, by default True
        plot_y_std : bool, optional
            Defines do we need to plot y_var_std, by default True
        ratio_group : str, optional
            Name of the ratio group this VarVsVar is compared with. The ratio group
            allows you to compare different groups of VarVsVar within one plot.
            By default None
        **kwargs : kwargs
            Keyword arguments passed to `PlotLineObject`

        Raises
        ------
        ValueError
            If provided options are not compatible with each other
        """
        super().__init__(**kwargs)
        if len(x_var) != len(y_var_mean):
            raise ValueError(
                f"Length of `x_var` ({len(x_var)}) and `y_var_mean` "
                f"({len(y_var_mean)}) have to be identical."
            )
        if len(x_var) != len(y_var_std):
            raise ValueError(
                f"Length of `x_var` ({len(x_var)}) and `y_var_std` "
                f"({len(y_var_std)}) have to be identical."
            )
        if x_var_widths is not None and len(x_var) != len(x_var_widths):
            raise ValueError(
                f"Length of `x_var` ({len(x_var)}) and `x_var_widths` "
                f"({len(x_var_widths)}) have to be identical."
            )
        self.x_var = np.array(x_var)
        self.x_var_widths = None if x_var_widths is None else np.array(x_var_widths)
        self.y_var_mean = np.array(y_var_mean)
        self.y_var_std = np.array(y_var_std)

        self.key = key
        self.fill = fill
        self.plot_y_std = plot_y_std
        self.ratio_group = ratio_group

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                np.all(self.x_var == other.x_var)
                and np.all(self.y_var_mean == other.y_var_mean)
                and np.all(self.y_var_std == other.y_var_std)
                and self.key == other.key
            )
        return False

    def divide(self, other, inverse: bool = False, method: str = "divide"):
        """Calculate ratio between two class objects.

        Parameters
        ----------
        other : VarVsVar class
            Second VarVsVar object to calculate ratio with
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
        if not np.array_equal(self.x_var, other.x_var):
            raise ValueError("The x variables of the two given objects do not match.")
        nom, nom_err = self.y_var_mean, self.y_var_std
        denom, denom_err = other.y_var_mean, other.y_var_std

        ratio, ratio_err = hist_ratio(
            numerator=denom if inverse else nom,
            denominator=nom if inverse else denom,
            numerator_unc=denom_err if inverse else nom_err,
            step=False,
            method=method,
        )
        return (ratio, ratio_err)


class VarVsVarPlot(PlotBase):
    """var_vs_eff plot class."""

    def __init__(self, grid: bool = False, ratio_method: str = "divide", **kwargs) -> None:
        """var_vs_eff plot properties.

        Parameters
        ----------
        grid : bool, optional
            Set the grid for the plots.
        ratio_method:
            Method for ratio calculations. Accepted values: "divide", "root_square_diff".
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
        self.ratio_method = ratio_method
        self.initialise_figure()

    def add(self, curve: VarVsVar, key: str | None = None, reference: bool = False):
        """Adding VarVsVar object to figure.

        Parameters
        ----------
        curve : VarVsVar class
            VarVsVar curve
        key : str, optional
            Unique identifier for VarVsVar curve, by default None
        reference : bool, optional
            If VarVsVar is used as reference for ratio calculation, by default False

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

        if curve.is_marker is True:
            if curve.marker is None:
                curve.marker = get_good_markers()[len(self.plot_objects)]
            # Set markersize
            if curve.markersize is None:
                curve.markersize = 8
            if curve.markeredgewidth is None:
                curve.markeredgewidth = 2

        # set min and max edges
        if curve.x_var_widths is not None:
            left_edge = curve.x_var - curve.x_var_widths / 2
            right_edge = curve.x_var + curve.x_var_widths / 2
        else:
            left_edge = curve.x_var
            right_edge = curve.x_var
        self.x_var_min = min(self.x_var_min, np.sort(left_edge)[0])
        self.x_var_max = max(self.x_var_max, np.sort(right_edge)[-1])

        if reference:
            logger.debug("Setting roc %s as reference.", key)
            self.set_reference(key)

    def set_reference(self, key: str):
        """Setting the reference VarVsVar curves used in the ratios.

        Parameters
        ----------
        key : str
            Unique identifier of roc object
        """
        if self.reference_object is None:
            self.reference_object = [key]
        else:
            self.reference_object.append(key)
        logger.debug("Adding '%s' to reference VarVsVar(s)", key)

    def plot(self, **kwargs):
        """Plotting curves.

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
                elem.x_var,
                elem.y_var_mean,
                xerr=elem.x_var_widths / 2 if elem.x_var_widths is not None else None,
                yerr=(elem.y_var_std if elem.plot_y_std else np.zeros_like(elem.x_var)),
                color=elem.colour,
                fmt="none",
                label=elem.label,
                alpha=elem.alpha,
                linewidth=elem.linewidth,
                ms=elem.markersize,
                **kwargs,
            )
            # # set linestyle for errorbar
            error_bar[-1][0].set_linestyle(elem.linestyle)
            # Draw markers
            if elem.is_marker is True:
                self.axis_top.scatter(
                    x=elem.x_var,
                    y=elem.y_var_mean,
                    marker=elem.marker,
                    s=elem.markersize**2,
                    color=elem.colour,
                )
            if elem.x_var_widths is not None and elem.fill:
                for x_pos, y_pos, width, height in zip(
                    elem.x_var,
                    elem.y_var_mean,
                    elem.x_var_widths,
                    2 * elem.y_var_std,
                ):
                    self.axis_top.add_patch(
                        Rectangle(
                            xy=(
                                x_pos - width / 2,
                                y_pos - height / 2,
                            ),
                            width=width,
                            height=height,
                            linewidth=0,
                            color=elem.colour,
                            alpha=0.3,
                            zorder=1,
                        )
                    )
            plt_handles.append(
                mpl.lines.Line2D(
                    [],
                    [],
                    color=elem.colour,
                    label=elem.label,
                    linestyle=elem.linestyle,
                    marker=elem.marker,
                    markersize=elem.markersize,
                )
            )
        return plt_handles

    def get_reference_name(self, var_object: VarVsVar):
        """Get reference VarVsVar object from list of references.

        Parameters
        ----------
        var_object : puma.var_vs_var.VarVsVar
            VarVsVar we want to calculate the ratio for

        Returns
        -------
        reference_name : str, int
            Identifier of the corresponding reference VarVsVar

        Raises
        ------
        ValueError
            If no reference VarVsVar was found or multiple matches.
        """
        matches = 0
        reference_name = None

        for key in self.reference_object:
            reference_candidate = self.plot_objects[key]
            if var_object.ratio_group is not None:
                if var_object.ratio_group == reference_candidate.ratio_group:
                    matches += 1
                    reference_name = reference_candidate
            else:
                matches += 1
                reference_name = reference_candidate

        if matches != 1:
            raise ValueError(
                f"Found {matches} matching reference candidates, but only one match is allowed."
            )

        logger.debug("Reference var_object for '%s' is '%s'", var_object.key, reference_name.key)

        return reference_name

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
            ratio, ratio_err = elem.divide(
                other=self.get_reference_name(elem),
                method=self.ratio_method,
            )
            error_bar = self.ratio_axes[0].errorbar(
                elem.x_var,
                ratio,
                xerr=elem.x_var_widths / 2 if elem.x_var_widths is not None else None,
                yerr=ratio_err if elem.plot_y_std else np.zeros_like(elem.x_var),
                color=elem.colour,
                fmt="none",
                alpha=elem.alpha,
                linewidth=elem.linewidth,
                ms=elem.markersize,
            )
            # set linestyle for errorbar
            error_bar[-1][0].set_linestyle(elem.linestyle)
            # draw markers
            if elem.is_marker is True:
                self.ratio_axes[0].scatter(
                    x=elem.x_var,
                    y=ratio,
                    marker=elem.marker,
                    color=elem.colour,
                    s=elem.markersize**2,
                )
            if elem.x_var_widths is not None and elem.fill:
                for x_pos, y_pos, width, height in zip(
                    elem.x_var, ratio, elem.x_var_widths, 2 * ratio_err
                ):
                    self.ratio_axes[0].add_patch(
                        Rectangle(
                            xy=(
                                x_pos - width / 2,
                                y_pos - height / 2,
                            ),
                            width=width,
                            height=height,
                            linewidth=0,
                            color=elem.colour,
                            alpha=0.3,
                            zorder=1,
                        )
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
        labelpad: int | None = None,
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
                align="center",
                labelpad=labelpad,
            )
        self.make_legend(plt_handles, ax_mpl=self.axis_top)
        self.plotting_done = True
        if self.apply_atlas_style is True:
            self.atlasify()
