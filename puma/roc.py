"""ROC curve functions."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ftag import Flavour, Flavours
from scipy.interpolate import pchip

from puma.metrics import rej_err
from puma.plot_base import PlotBase, PlotLineObject
from puma.utils import get_good_colours, get_good_linestyles, logger


class Roc(PlotLineObject):
    """Represent a single ROC curve and allows to calculate ratio w.r.t other ROCs."""

    def __init__(
        self,
        sig_eff: np.ndarray,
        bkg_rej: np.ndarray,
        n_test: int = None,
        rej_class: str | Flavour = None,
        signal_class: str = None,
        key: str = None,
        **kwargs,
    ) -> None:
        """Initialise properties of roc curve object.

        Parameters
        ----------
        sig_eff : np.array
            Array of signal efficiencies
        bkg_rej : np.array
            Array of background rejection
        n_test : int
            Number of events used to calculate the background efficiencies,
            by default None
        signal_class : str
            Signal class, e.g. for b-tagging "bjets", by default None
        rej_class : str or Flavour
            Rejection class, e.g. for b-tagging anc charm rejection "cjets",
            by default None
        key : str
            Identifier for roc curve e.g. tagger, by default None
        **kwargs : kwargs
            Keyword arguments passed to `puma.PlotLineObject`

        Raises
        ------
        ValueError
            If `sig_eff` and `bkg_rej` have a different shape
        """
        super().__init__(**kwargs)
        if len(sig_eff) != len(bkg_rej):
            raise ValueError(
                f"The shape of `sig_eff` ({np.shape(sig_eff)}) and `bkg_rej` "
                f"({np.shape(bkg_rej)}) have to be identical."
            )
        self.sig_eff = sig_eff
        self.bkg_rej = bkg_rej
        self.n_test = None if n_test is None else int(n_test)
        self.signal_class = signal_class
        self.rej_class = (
            Flavours[rej_class] if isinstance(rej_class, str) else rej_class
        )
        self.key = key

    def binomial_error(self, norm: bool = False, n_test: int = None) -> np.ndarray:
        """Calculate binomial error of roc curve.

        Parameters
        ----------
        norm : bool
            If True calulate relative error, by default False
        n_test : int
            Number of events used to calculate the background efficiencies,
            by default None

        Returns
        -------
        numpy.array
            Binomial error

        Raises
        ------
        ValueError
            If no `n_test` was provided
        """
        if n_test is None:
            n_test = self.n_test
        if n_test is None:
            raise ValueError("No `n_test` provided, cannot calculate binomial error!")
        return rej_err(self.bkg_rej[self.non_zero_mask], n_test, norm=norm)

    def divide(self, roc_comp, inverse: bool = False):
        """Calculate ratio between the roc curve and another roc.

        Parameters
        ----------
        roc_comp : roc class
            Second roc curve to calculate ratio with
        inverse : bool
            If False the ratio is calculated `this_roc / roc_comp`,
            if True the inverse is calculated

        Returns
        -------
        np.array
            Signal efficiency used for the ratio calculation which is the overlapping
            interval of the two roc curves
        np.array
            Ratio
        np.array or None
            Ratio_err if `n_test` was provided to class
        """
        # if same objects return array with value 1
        if np.array_equal(
            np.array([self.sig_eff, self.bkg_rej]),
            np.array([roc_comp.sig_eff, roc_comp.bkg_rej]),
        ):
            logger.debug("roc objects are identical -> ratio is 1.")
            ratio = np.ones(len(self.sig_eff))
            if self.n_test is None:
                return self.sig_eff, ratio, None
            ratio_err = self.binomial_error(norm=True) * ratio
            return self.sig_eff, ratio, ratio_err

        # get overlapping sig_eff interval of the two roc curves
        min_eff = max(self.sig_eff.min(), roc_comp.sig_eff.min())
        max_eff = min(self.sig_eff.max(), roc_comp.sig_eff.max())
        eff_mask = np.all([self.sig_eff >= min_eff, self.sig_eff <= max_eff], axis=0)
        ratio_sig_eff = self.sig_eff[eff_mask]

        # Ratio of interpolated rejection functions
        ratio = self.fct_inter(ratio_sig_eff) / roc_comp.fct_inter(ratio_sig_eff)
        if inverse:
            ratio = 1 / ratio
        if self.n_test is None:
            return ratio_sig_eff, ratio, None
        ratio_err = self.binomial_error(norm=True) * ratio
        return ratio_sig_eff, ratio, ratio_err[eff_mask]

    @property
    def fct_inter(self):
        """
        Interpolate the rejection function for better ratio calculation plotting etc.

        Returns
        -------
        pchip
            Interpolation function
        """
        return pchip(self.sig_eff, self.bkg_rej)

    @property
    def non_zero_mask(self):
        """Masking points where rejection is 0 and no signal efficiency change present.

        Returns
        -------
        numpy.array
            Masked indices
        """
        # Mask the points where there was no change in the signal eff
        delta_x = np.concatenate((np.ones(1), np.diff(self.sig_eff)))

        # Also mask the rejections that are 0
        nonzero = (self.bkg_rej != 0) & (delta_x > 0)
        if self.xmin is not None:
            nonzero = nonzero & (self.sig_eff >= self.xmin)
        if self.xmax is not None:
            nonzero = nonzero & (self.sig_eff <= self.xmax)
        return nonzero

    @property
    def non_zero(self):
        """Abstraction of `non_zero_mask`.

        Returns
        -------
        numpy.array
            Masked signal efficiency
        numpy.array
            Masked background rejection
        """
        return self.sig_eff[self.non_zero_mask], self.bkg_rej[self.non_zero_mask]


class RocPlot(PlotBase):
    """ROC plot class."""

    def __init__(self, grid: bool = True, **kwargs) -> None:
        """ROC plot properties.

        Parameters
        ----------
        grid : bool, optional
            Set the grid for the plots.
        **kwargs : kwargs
            Keyword arguments from `puma.PlotObject`
        """
        super().__init__(grid=grid, **kwargs)
        self.test = ""
        self.rocs = {}
        self.roc_ratios = {}
        self.rej_axes = {}
        self.rej_class_ls = {}
        self.label_colours = {}
        self.leg_rej_labels = {}
        self.reference_roc = None
        self.initialise_figure()
        self.eff_min, self.eff_max = (1, 0)
        self.default_linestyles = get_good_linestyles()
        self.legend_flavs = None
        self.leg_rej_loc = "lower left"

    def add_roc(self, roc_curve: object, key: str = None, reference: bool = False):
        """Adding puma.Roc object to figure.

        Parameters
        ----------
        roc_curve : puma.Roc
            ROC curve
        key : str, optional
            Unique identifier for roc_curve, by default None
        reference : bool, optional
            If roc is used as reference for ratio calculation, by default False

        Raises
        ------
        KeyError
            If unique identifier key is used twice
        """
        if key is None:
            key = len(self.rocs) + 1
        if key in self.rocs:
            raise KeyError(
                f"Duplicated key {key} already used for roc unique identifier."
            )

        self.rocs[key] = roc_curve
        # set linestyle
        if roc_curve.rej_class not in self.rej_class_ls:
            self.rej_class_ls[roc_curve.rej_class] = (
                self.default_linestyles[len(self.rej_class_ls)]
                if roc_curve.linestyle is None
                else roc_curve.linestyle
            )
        elif (
            roc_curve.linestyle != self.rej_class_ls[roc_curve.rej_class]
            and roc_curve.linestyle is not None
        ):
            logger.warning(
                (
                    "You specified a different linestyle for the same rejection class "
                    "%s. Will keep the linestyle defined first."
                ),
                roc_curve.rej_class,
            )
        if roc_curve.linestyle is None:
            roc_curve.linestyle = self.rej_class_ls[roc_curve.rej_class]

        # set colours
        if roc_curve.label not in self.label_colours:
            self.label_colours[roc_curve.label] = (
                get_good_colours()[len(self.label_colours)]
                if roc_curve.colour is None
                else roc_curve.colour
            )
        elif (
            roc_curve.colour != self.label_colours[roc_curve.label]
            and roc_curve.colour is not None
        ):
            logger.warning(
                (
                    "You specified a different colour for the same label"
                    " %s. This will lead to a mismatch in the line colours"
                    " and the legend."
                ),
                roc_curve.label,
            )
        if roc_curve.colour is None:
            roc_curve.colour = self.label_colours[roc_curve.label]

        if reference:
            logger.debug(
                "Setting roc %s as reference for %s.", key, roc_curve.rej_class
            )
            self.set_roc_reference(key, roc_curve.rej_class)

    def set_roc_reference(self, key: str, rej_class: Flavour):
        """Setting the reference roc curves used in the ratios.

        Parameters
        ----------
        key : str
            Unique identifier of roc object
        rej_class : str
            Rejection class encoded in roc curve

        Raises
        ------
        ValueError
            If more rejection classes are set than actual ratio panels available.
        """
        if self.reference_roc is None:
            self.reference_roc = {}
            self.reference_roc[rej_class] = key
        elif rej_class not in self.reference_roc:
            if len(self.reference_roc) >= self.n_ratio_panels:
                raise ValueError(
                    "You cannot set more rejection classes than available ratio panels."
                )
            self.reference_roc[rej_class] = key
        else:
            logger.warning(
                (
                    "You specified a second roc curve %s as reference for ratio. "
                    "Using it as new reference instead of %s."
                ),
                key,
                self.reference_roc[rej_class],
            )
            self.reference_roc[rej_class] = key

    def set_ratio_class(self, ratio_panel: int, rej_class: str | Flavour):
        """Associate the rejection class to a ratio panel adn set the legend label.

        Parameters
        ----------
        ratio_panel : int
            Ratio panel either 1 or 2
        rej_class : Flavour
            Rejeciton class associated to that panel

        Raises
        ------
        ValueError
            if requested ratio panels and given ratio_panel do not match.
        """
        rej_class = Flavours[rej_class] if isinstance(rej_class, str) else rej_class
        if self.n_ratio_panels < ratio_panel:
            raise ValueError(
                "Requested ratio panels and given ratio_panel do not match."
            )
        self.rej_axes[rej_class] = self.ratio_axes[ratio_panel - 1]
        label = rej_class.label.replace("jets", "jet")
        self.set_ratio_label(ratio_panel, f"{label} ratio")
        self.leg_rej_labels[rej_class] = rej_class.label

    def add_ratios(self):
        """Calculating ratios.

        Raises
        ------
        ValueError
            If number of reference rocs and ratio panels don't match
        ValueError
            If no ratio classes are set
        """
        if len(self.reference_roc) != self.n_ratio_panels:
            raise ValueError(
                f"{len(self.reference_roc)} reference rocs defined but requested "
                f"{self.n_ratio_panels} ratio panels."
            )
        if len(self.rej_axes) != self.n_ratio_panels:
            raise ValueError(
                "Ratio classes not set, set them first with `set_ratio_class`."
            )

        for rej_class, axis in self.rej_axes.items():
            self.plot_ratios(axis=axis, rej_class=rej_class)

    def get_xlim_auto(self):
        """Returns min and max efficiency values.

        Returns
        -------
        float
            Min and max efficiency values
        """
        for elem in self.rocs.values():
            self.eff_min = min(np.min(elem.sig_eff), self.eff_min)
            self.eff_max = max(np.max(elem.sig_eff), self.eff_min)

        return self.eff_min, self.eff_max

    def plot_ratios(self, axis: plt.axis, rej_class: str):
        """Plotting ratio curves.

        Parameters
        ----------
        axis : plt.axis
            matplotlib axis object
        rej_class : str
            Rejection class
        """
        for key, elem in self.rocs.items():
            if elem.rej_class != rej_class:
                continue
            ratio_sig_eff, ratio, ratio_err = elem.divide(
                self.rocs[self.reference_roc[rej_class]]
            )
            self.roc_ratios[key] = (ratio_sig_eff, ratio, ratio_err)
            axis.plot(
                ratio_sig_eff,
                ratio,
                color=elem.colour,
                linestyle=elem.linestyle,
                linewidth=1.6,
            )
            if ratio_err is not None:
                axis.fill_between(
                    ratio_sig_eff,
                    ratio - ratio_err,
                    ratio + ratio_err,
                    color=elem.colour,
                    alpha=0.3,
                    zorder=1,
                )

    def set_leg_rej_loc(self, option: str):
        """Set the position of the rejection class legend. Only if 2 ratio panels are
        defined.

        Parameters
        ----------
        option : str
            Defines where to place the legend for rejection class. Accepts all options
            from `matplotlib.axes.Axes.legend` as well as the option `ratio_legend`,
            which adds the legend into the ratio panels

        Raises
        ------
        ValueError
            If not 2 ratios requested
        """
        if self.n_ratio_panels != 2:
            raise ValueError("For a rejection class legend you need 2 ratio panels.")

        self.leg_rej_loc = option

    def make_split_legend(self, handles):
        """Draw legend for the case of 2 ratios, splitting up legend into models and
        rejection class.

        Parameters
        ----------
        handles : list
            List of Line2D objects to extract info for legend

        Raises
        ------
        ValueError
            If not 2 ratios requested
        """
        if self.n_ratio_panels < 2:
            raise ValueError("For a split legend you need 2 ratio panels.")

        if self.leg_rej_loc == "ratio_legend":
            for rej_class, axis in self.rej_axes.items():
                legend_line = mpl.lines.Line2D(
                    [],
                    [],
                    color="k",
                    label=self.leg_rej_labels[rej_class],
                    linestyle=self.rej_class_ls[rej_class],
                )
                axis.legend(
                    handles=[legend_line],
                    labels=[legend_line.get_label()],
                    loc="upper right",
                    fontsize=self.leg_fontsize,
                )

        else:
            line_list_rej = []
            for rej_class in self.rej_axes:
                line_list_rej.append(
                    mpl.lines.Line2D(
                        [],
                        [],
                        color="k",
                        label=self.leg_rej_labels[rej_class],
                        linestyle=self.rej_class_ls[rej_class],
                    )
                )

            self.legend_flavs = self.axis_top.legend(
                handles=line_list_rej,
                labels=[handle.get_label() for handle in line_list_rej],
                loc=self.leg_rej_loc,
                fontsize=self.leg_fontsize,
                ncol=self.leg_ncol,
            )

            # Add the second legend to plot
            self.axis_top.add_artist(self.legend_flavs)

        # Get the labels for the legends
        labels_list = []
        lines_list = []

        for line in handles:
            if line.get_label() not in labels_list:
                labels_list.append(line.get_label())
                lines_list.append(line)

        # Define the legend
        self.axis_top.legend(
            handles=lines_list,
            labels=labels_list,
            loc=self.leg_loc,
            fontsize=self.leg_fontsize,
            ncol=self.leg_ncol,
        )

    def draw(
        self,
        labelpad: int = None,
    ):
        """Draw plotting.

        Parameters
        ----------
        labelpad : int, optional
            Spacing in points from the axes bounding box including
            ticks and tick labels, by default None
        """
        plt_handles = self.plot_roc()
        xmin, xmax = self.get_xlim_auto()

        self.set_xlim(
            xmin if self.xmin is None else self.xmin,
            xmax if self.xmax is None else self.xmax,
        )
        if self.n_ratio_panels > 0:
            self.add_ratios()
        self.set_title()
        self.set_log()
        self.set_y_lim()
        self.set_xlabel()
        self.set_ylabel(self.axis_top)

        for i, axis in enumerate(self.rej_axes.values()):
            self.set_ylabel(
                axis,
                self.ylabel_ratio[i],
                align_right=False,
                labelpad=labelpad,
            )

        if self.n_ratio_panels < 2:
            self.make_legend(plt_handles, ax_mpl=self.axis_top)
        else:
            if not self.leg_rej_labels:
                for rej_class in self.rej_axes:
                    self.leg_rej_labels[rej_class] = rej_class

            self.make_split_legend(handles=plt_handles)

        self.plotting_done = True
        if self.apply_atlas_style is True:
            self.atlasify()
            # atlasify can only handle one legend. Therefore, we remove the frame of
            # the second legend by hand
            if self.legend_flavs is not None:
                self.legend_flavs.set_frame_on(False)

    def plot_roc(self, **kwargs) -> mpl.lines.Line2D:
        """Plotting roc curves.

        Parameters
        ----------
        **kwargs: kwargs
            Keyword arguments passed to plt.axis.plot

        Returns
        -------
        Line2D
            matplotlib Line2D object
        """
        plt_handles = []
        for key, elem in self.rocs.items():
            plt_handles = plt_handles + self.axis_top.plot(
                elem.sig_eff[elem.non_zero_mask],
                elem.bkg_rej[elem.non_zero_mask],
                linestyle=elem.linestyle,
                color=elem.colour,
                label=elem.label if elem is not None else key,
                zorder=2,
                **kwargs,
            )
            if elem.n_test is not None:
                # if uncertainties are available for roc plotting their uncertainty as
                # a band around the roc itself
                rej_band_down = (
                    elem.bkg_rej[elem.non_zero_mask]
                    - elem.binomial_error()[elem.non_zero_mask]
                )
                rej_band_up = (
                    elem.bkg_rej[elem.non_zero_mask]
                    + elem.binomial_error()[elem.non_zero_mask]
                )
                self.axis_top.fill_between(
                    elem.sig_eff[elem.non_zero_mask],
                    rej_band_down,
                    rej_band_up,
                    color=elem.colour,
                    alpha=0.3,
                    zorder=2,
                )
        return plt_handles
