"""ROC curve functions."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import pchip

import umami.tools.PyATLASstyle.PyATLASstyle as pas
from umami.configuration import logger
from umami.metrics import rej_err
from umami.plotting.plot_base import plot_base, plot_line_object


class roc(plot_line_object):
    """
    ROC class storing info about curve and allows to calculate ratio w.r.t other roc.
    """

    def __init__(
        self,
        sig_eff: np.ndarray,
        bkg_rej: np.ndarray,
        n_test: int = None,
        rej_class: str = None,
        signal_class: str = None,
        key: str = None,
        **kwargs,
    ) -> None:
        """Initialise properties of roc curve object.

        Parameters
        ----------
        sig_eff : np.array
            array of signal efficiencies
        bkg_rej : np.array
            array of background rejection
        n_test : int
            Number of events used to calculate the background efficiencies,
            by default None
        signal_class : str
            Signal class, e.g. for b-tagging "bjets", by default None
        rej_class : str
            Rejection class, e.g. for b-tagging anc charm rejection "cjets",
            by default None
        key : str
            identifier for roc curve e.g. tagger, by default None
        **kwargs : kwargs
            kwargs passed to `plot_line_object`

        Raises
        ------
        ValueError
            if `sig_eff` and `bkg_rej` have a different shape
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
        self.rej_class = rej_class
        self.key = key

    def binomial_error(self, norm: bool = False, n_test: int = None) -> np.ndarray:
        """Calculate binomial error of roc curve.

        Parameters
        ----------
        norm : bool
            if True calulate relative error, by default False
        n_test : int
            Number of events used to calculate the background efficiencies,
            by default None

        Returns
        -------
        numpy.array
            binomial error

        Raises
        ------
        ValueError
            if no `n_test` was provided
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
            second roc curve to calculate ratio with
        inverse : bool
            if False the ratio is calculated `this_roc / roc_comp`,
            if True the inverse is calculated

        Returns
        -------
        np.array
            signal efficiency used for the ratio calculation which is the overlapping
            interval of the two roc curves
        np.array
            ratio
        np.array or None
            ratio_err if `n_test` was provided to class
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
            ratio_err = self.binomial_error(norm=True)
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
        ratio_err = self.binomial_error(norm=True)
        return ratio_sig_eff, ratio, ratio_err[eff_mask]

    @property
    def fct_inter(self):
        """
        Interpolate the rejection function for better ratio calculation plotting etc.

        Returns
        -------
        pchip
            interpolation function
        """
        return pchip(self.sig_eff, self.bkg_rej)

    @property
    def non_zero_mask(self):
        """Masking points where rejection is 0 and no signal efficiency change present

        Returns
        -------
        numpy.array
            masked indices
        """
        # Mask the points where there was no change in the signal eff
        dx = np.concatenate((np.ones(1), np.diff(self.sig_eff)))

        # Also mask the rejections that are 0
        nonzero = (self.bkg_rej != 0) & (dx > 0)
        if self.xmin is not None:
            nonzero = nonzero & (self.sig_eff >= self.xmin)
        if self.xmax is not None:
            nonzero = nonzero & (self.sig_eff <= self.xmax)
        return nonzero

    @property
    def non_zero(self):
        """Abstraction of `non_zero_mask`

        Returns
        -------
        numpy.array
            masked signal efficiency
        numpy.array
            masked background rejection
        """
        return self.sig_eff[self.non_zero_mask], self.bkg_rej[self.non_zero_mask]


class roc_plot(plot_base):
    """Roc plot class"""

    def __init__(self, **kwargs) -> None:
        """roc plot properties

        Parameters
        ----------
        **kwargs : kwargs
            kwargs from `plot_base`
        """
        super().__init__(**kwargs)
        self.test = ""
        self.rocs = {}
        self.roc_ratios = {}
        self.ratio_axes = {}
        self.rej_class_ls = {}
        self.label_colours = {}
        self.leg_rej_labels = {}
        self.reference_roc = None
        self.initialise_figure()
        self.eff_min, self.eff_max = (1, 0)
        # setting default linestyles if no linestyles provided
        # solid line and densed dotted dashed
        self.default_linestyles = ["-", (0, (3, 1, 1, 1))]
        self.legend_flavs = None

    def add_roc(self, roc_curve: object, key: str = None, reference: bool = False):
        """Adding roc object to figure.

        Parameters
        ----------
        roc_curve : roc class
            roc curve
        key : str, optional
            unique identifier for roc, by default None
        reference : bool, optional
            if roc is used as reference for ratio calculation, by default False

        Raises
        ------
        KeyError
            if unique identifier key is used twice
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
                "You specified a different linestyle for the same rejection class "
                f"{roc_curve.rej_class}. Will keep the linestyle defined first."
            )
        if roc_curve.linestyle is None:
            roc_curve.linestyle = self.rej_class_ls[roc_curve.rej_class]

        # set colours
        if roc_curve.label not in self.label_colours:
            self.label_colours[roc_curve.label] = (
                pas.get_good_colours()[len(self.label_colours)]
                if roc_curve.colour is None
                else roc_curve.colour
            )
        elif (
            roc_curve.colour != self.label_colours[roc_curve.label]
            and roc_curve.colour is not None
        ):
            logger.warning(
                "You specified a different colour for the same label"
                f" {roc_curve.label}. This will lead to a mismatch in the line colours"
                " and the legend."
            )
        if roc_curve.colour is None:
            roc_curve.colour = self.label_colours[roc_curve.label]

        if reference:
            logger.debug(f"Setting roc {key} as reference for {roc_curve.rej_class}.")
            self.set_roc_reference(key, roc_curve.rej_class)

    def set_roc_reference(self, key: str, rej_class: str):
        """Setting the reference roc curves used in the ratios

        Parameters
        ----------
        key : str
            unique identifier of roc object
        rej_class : str
            rejection class encoded in roc curve

        Raises
        ------
        ValueError
            if more rejection classes are set than actual ratio panels available.
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
                f"You specified a second roc curve {key} as reference for ratio. "
                f"Using it as new reference instead of {self.reference_roc[rej_class]}."
            )
            self.reference_roc[rej_class] = key

    def set_leg_rej_labels(self, rej_class: str, label: str):
        """Set legend label for rejection class

        Parameters
        ----------
        rej_class : str
            rejection class
        label : str
            label added in legend
        """
        self.leg_rej_labels[rej_class] = label

    def set_ratio_class(self, ratio_panel: int, rej_class: str, label: str):
        """Associate the rejection class to a ratio panel

        Parameters
        ----------
        ratio_panel : int
            ratio panel either 1 or 2
        rej_class : str
            rejeciton class associated to that panel
        label : str
            y-axis label of the ratio panel

        Raises
        ------
        ValueError
            if requested ratio panels and given ratio_panel do not match.
        """
        if self.n_ratio_panels < ratio_panel and ratio_panel not in [1, 2]:
            raise ValueError(
                "Requested ratio panels and given ratio_panel do not match."
            )
        self.ratio_axes[ratio_panel] = rej_class
        self.set_ratio_label(ratio_panel, label)

    def add_ratios(self):
        """Calculating ratios.

        Raises
        ------
        ValueError
            if number of reference rocs and ratio panels don't match
        ValueError
            if no ratio classes are set
        """
        if len(self.reference_roc) != self.n_ratio_panels:
            raise ValueError(
                f"{len(self.reference_roc)} reference rocs defined but requested "
                f"{self.n_ratio_panels} ratio panels."
            )
        if len(self.ratio_axes) != self.n_ratio_panels:
            raise ValueError(
                "Ratio classes not set, set them first with `set_ratio_class`."
            )
        self.plot_ratios(ax=self.axis_ratio_1, rej_class=self.ratio_axes[1])

        if self.grid:
            self.axis_ratio_1.grid()

        if self.n_ratio_panels == 2:
            self.plot_ratios(ax=self.axis_ratio_2, rej_class=self.ratio_axes[2])

            if self.grid:
                self.axis_ratio_2.grid()

    def get_xlim_auto(self):
        """Returns min and max efficiency values

        Returns
        -------
        float
            min and max efficiency values
        """

        for elem in self.rocs.values():
            self.eff_min = min(np.min(elem.sig_eff), self.eff_min)
            self.eff_max = max(np.max(elem.sig_eff), self.eff_min)

        return self.eff_min, self.eff_max

    def plot_ratios(self, ax: plt.axis, rej_class: str):
        """Plotting ratio curves

        Parameters
        ----------
        ax : plt.axis
            matplotlib axis object
        rej_class : str
            rejection class
        """
        for key, elem in self.rocs.items():
            if elem.rej_class != rej_class:
                continue
            ratio_sig_eff, ratio, ratio_err = elem.divide(
                self.rocs[self.reference_roc[rej_class]]
            )
            self.roc_ratios[key] = (ratio_sig_eff, ratio, ratio_err)
            ax.plot(
                ratio_sig_eff,
                ratio,
                color=elem.colour,
                linestyle=elem.linestyle,
                linewidth=1.6,
            )
            if ratio_err is not None:
                ax.fill_between(
                    ratio_sig_eff,
                    ratio - ratio_err,
                    ratio + ratio_err,
                    color=elem.colour,
                    alpha=0.3,
                    zorder=1,
                )

    def make_split_legend(self, handles):
        """Draw legend for the case of 2 ratios, splitting up legend into models and
        rejection class.

        Parameters
        ----------
        handles : list
            list of Line2D objects to extract info for legend

        Raises
        ------
        ValueError
            if not 2 ratios requested
        """

        if self.n_ratio_panels != 2:
            raise ValueError("For a split legend you need 2 ratio panels.")

        line_list_rej = []
        for elem in [self.ratio_axes[1], self.ratio_axes[2]]:
            line_list_rej.append(
                mpl.lines.Line2D(
                    [],
                    [],
                    color="k",
                    label=self.leg_rej_labels[elem],
                    linestyle=self.rej_class_ls[elem],
                )
            )

        self.legend_flavs = self.axis_top.legend(
            handles=line_list_rej,
            labels=[handle.get_label() for handle in line_list_rej],
            loc="upper center",
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
        """Draw plotting

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
        self.add_ratios()
        self.set_title()
        self.set_logy()
        self.set_y_lim()
        self.set_xlabel()
        self.set_ylabel(self.axis_top)
        if self.grid:
            self.axis_top.grid()

        if self.n_ratio_panels > 0:
            self.set_ylabel(
                self.axis_ratio_1,
                self.ylabel_ratio_1,
                align_right=False,
                labelpad=labelpad,
            )
        if self.n_ratio_panels == 2:
            self.set_ylabel(
                self.axis_ratio_2,
                self.ylabel_ratio_2,
                align_right=False,
                labelpad=labelpad,
            )

        if self.n_ratio_panels < 2:
            self.make_legend(plt_handles, ax=self.axis_top)
        else:
            if not self.leg_rej_labels:
                self.leg_rej_labels[self.ratio_axes[1]] = self.ratio_axes[1]
                self.leg_rej_labels[self.ratio_axes[2]] = self.ratio_axes[2]

            self.make_split_legend(handles=plt_handles)

        self.plotting_done = True
        if self.apply_atlas_style is True:
            self.atlasify(use_tag=self.use_atlas_tag)
            # atlasify can only handle one legend. Therefore, we remove the frame of
            # the second legend by hand
            if self.legend_flavs is not None:
                self.legend_flavs.set_frame_on(False)

        self.tight_layout()

    def plot_roc(self, **kwargs) -> mpl.lines.Line2D:
        """Plotting roc curves

        Parameters
        ----------
        **kwargs: kwargs
            kwargs passed to plt.axis.plot

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
        return plt_handles
