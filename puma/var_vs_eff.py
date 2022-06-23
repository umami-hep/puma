"""Efficiency plots vs. specific variable."""
import matplotlib as mpl
import numpy as np

# TODO: fix the import below
from puma.metrics import eff_err, rej_err
from puma.plot_base import PlotBase, PlotLineObject
from puma.utils import get_good_colours, logger
from puma.utils.histogram import hist_ratio, save_divide


class VarVsEff(PlotLineObject):  # pylint: disable=too-many-instance-attributes
    """
    var_vs_eff class storing info about curve and allows to calculate ratio w.r.t other
    efficiency plots.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        x_var_sig: np.ndarray,
        disc_sig: np.ndarray,
        x_var_bkg: np.ndarray = None,
        disc_bkg: np.ndarray = None,
        bins=10,
        working_point: float = None,
        disc_cut=None,
        fixed_eff_bin: bool = False,
        key: str = None,
        **kwargs,
    ) -> None:
        """Initialise properties of roc curve object.

        Parameters
        ----------
        x_var_sig : np.ndarray
            Values for x-axis variable for signal
        disc_sig : np.ndarray
            Discriminant values for signal
        x_var_bkg : np.ndarray, optional
            Values for x-axis variable for background, by default None
        disc_bkg : np.ndarray, optional
            Discriminant values for background, by default None
        bins : int or sequence of scalars, optional
            If bins is an int, it defines the number of equal-width bins in the
            given range (10, by default). If bins is a sequence, it defines a
            monotonically increasing array of bin edges, including the
            rightmost edge, allowing for non-uniform bin widths, by default 10
        working_point : float, optional
            Working point, by default None
        disc_cut : float or  sequence of floats, optional
            Cut value for discriminant, if it is a sequence it has to have the same
            length as number of bins, by default None
        fixed_eff_bin : bool, optional
            If True and no `disc_cut` is given the signal efficiency is held constant
            in each bin, by default False
        key : str, optional
            Identifier for the curve e.g. tagger, by default None
        **kwargs : kwargs
            Keyword arguments passed to `PlotLineObject`

        Raises
        ------
        ValueError
            If provided options are not compatible with each other
        """
        # TODO: in python 3.10 add multipe type operator | for bins and disc_cut

        super().__init__(**kwargs)
        if len(x_var_sig) != len(disc_sig):
            raise ValueError(
                f"Length of `x_var_sig` ({len(x_var_sig)}) and `disc_sig` "
                f"({len(disc_sig)}) have to be identical."
            )
        if x_var_bkg is not None and len(x_var_bkg) != len(disc_bkg):
            raise ValueError(
                f"Length of `x_var_bkg` ({len(x_var_bkg)}) and `disc_bkg` "
                f"({len(disc_bkg)}) have to be identical."
            )
        # checking that the given options are compatible
        # could also think about porting it to a class function insted of passing
        # the arguments to init e.g. `set_method`
        if working_point is None and disc_cut is None:
            raise ValueError("Either `wp` or `disc_cut` needs to be specified.")
        if fixed_eff_bin:
            if disc_cut is not None:
                raise ValueError(
                    "You cannot specify `disc_cut` when `fixed_eff_bin` is set to True."
                )
            if working_point is None:
                raise ValueError(
                    "You need to specify a working point `wp`, when `fixed_eff_bin` is "
                    "set to True."
                )
        self.x_var_sig = np.array(x_var_sig)
        self.disc_sig = np.array(disc_sig)
        self.x_var_bkg = None if x_var_bkg is None else np.array(x_var_bkg)
        self.disc_bkg = None if disc_bkg is None else np.array(disc_bkg)
        self.working_point = working_point
        self.disc_cut = disc_cut
        self.fixed_eff_bin = fixed_eff_bin
        self.key = key
        # Binning related variables
        self.n_bins = None
        self.bn_edges = None
        self.x_bin_centres = None
        self.bin_widths = None
        self.n_bins = None
        # Binned distributions
        self.bin_indices_sig = None
        self.disc_binned_sig = None
        self.bin_indices_bkg = None
        self.disc_binned_bkg = None

        self._set_bin_edges(bins)

        if disc_cut is not None:
            if working_point is not None:
                raise ValueError("You cannot specify `disc_cut` when providing `wp`.")
            if isinstance(disc_cut, (list, np.ndarray)):
                if self.n_bins != len(disc_cut):
                    raise ValueError(
                        "`disc_cut` has to be a float or has to have the same length "
                        "as number of bins."
                    )
        self._apply_binning()
        self._get_disc_cuts()
        self.inverse_cut = False

    def _set_bin_edges(self, bins):
        """Calculate bin edges, centres and width and save them as class variables.

        Parameters
        ----------
        bins : int or sequence of scalars
            If bins is an int, it defines the number of equal-width bins in the given
            range. If bins is a sequence, it defines a monotonically increasing array of
            bin edges, including the rightmost edge, allowing for non-uniform bin
            widths.
        """
        logger.debug("Calculating binning.")
        if isinstance(bins, int):
            # With this implementation, the data point with x=xmax will be added to the
            # overflow bin.
            xmin, xmax = np.amin(self.x_var_sig), np.amax(self.x_var_sig)
            if self.x_var_bkg is not None:
                xmin = min(xmin, np.amin(self.x_var_bkg))
                xmax = max(xmax, np.amax(self.x_var_bkg))
            # increasing xmax slightly to inlcude largest value due to hehavior of
            # np.digitize
            xmax *= 1 + 1e-5
            self.bin_edges = np.linspace(xmin, xmax, bins + 1)
        elif isinstance(bins, (list, np.ndarray)):
            self.bin_edges = np.array(bins)
        logger.debug("Retrieved bin edges %s}", self.bin_edges)
        # Get the bins for the histogram
        self.x_bin_centres = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
        self.bin_widths = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
        self.n_bins = self.bin_edges.size - 1
        logger.debug("N bins: %i", self.n_bins)

    def _apply_binning(self):
        """Get binned distributions for the signal and background."""
        logger.debug("Applying binning.")
        self.bin_indices_sig = np.digitize(self.x_var_sig, self.bin_edges)
        if np.all(self.bin_indices_sig == 0):
            logger.error("All your signal is in the underflow bin. Check your input.")
            # retrieve for each bin the part of self.disc_sig corresponding to this bin
            # and put them in a list
        self.disc_binned_sig = list(
            map(
                lambda x: self.disc_sig[np.where(self.bin_indices_sig == x)[0]],
                range(1, len(self.bin_edges)),
            )
        )
        if self.x_var_bkg is not None:
            self.bin_indices_bkg = np.digitize(self.x_var_bkg, self.bin_edges)
            self.disc_binned_bkg = list(
                map(
                    lambda x: self.disc_bkg[np.where(self.bin_indices_bkg == x)[0]],
                    range(1, len(self.bin_edges)),
                )
            )

    def _get_disc_cuts(self):
        """Retrieve cut values on discriminant. If `disc_cut` is not given, retrieve
        cut values from the working point.
        """
        logger.debug("Calculate discriminant cut.")
        if isinstance(self.disc_cut, float):
            self.disc_cut = [self.disc_cut] * self.n_bins
        elif isinstance(self.disc_cut, (list, np.ndarray)):
            self.disc_cut = self.disc_cut
        elif self.fixed_eff_bin:
            self.disc_cut = list(
                map(
                    lambda x: np.percentile(x, (1 - self.working_point) * 100),
                    self.disc_binned_sig,
                )
            )
        else:
            self.disc_cut = [
                np.percentile(self.disc_sig, (1 - self.working_point) * 100)
            ] * self.n_bins
        logger.debug("Discriminant cut: %.3f", self.disc_cut)

    def efficiency(self, arr: np.ndarray, cut: float):
        """Calculate efficiency and the associated error.

        Parameters
        ----------
        arr : np.ndarray
            Array with discriminants
        cut : float
            Cut value

        Returns
        -------
        float
            Efficiency
        float
            Efficiency error
        """
        if self.inverse_cut:
            eff = sum(arr < cut) / len(arr)
        else:
            eff = sum(arr > cut) / len(arr)
        eff_error = eff_err(eff, len(arr))
        return eff, eff_error

    def rejection(self, arr: np.ndarray, cut: float):
        """Calculate rejection and the associated error.

        Parameters
        ----------
        arr : np.ndarray
            Array with discriminants
        cut : float
            Cut value

        Returns
        -------
        float
            Rejection
        float
            Rejection error
        """
        if self.inverse_cut:
            rej = save_divide(len(arr), sum(arr < cut), default=np.inf)
        else:
            rej = save_divide(len(arr), sum(arr > cut), default=np.inf)
        if rej == np.inf:
            logger.warning("Your rejection is infinity -> setting it to np.nan.")
            return np.nan, np.nan
        rej_error = rej_err(rej, len(arr))
        return rej, rej_error

    @property
    def sig_eff(self):
        """Calculate signal efficiency per bin.

        Returns
        -------
        np.ndarray
            Efficiency
        np.ndarray
            Efficiency_error
        """
        logger.debug("Calculating signal efficiency.")
        eff = list(map(self.efficiency, self.disc_binned_sig, self.disc_cut))
        logger.debug("Retrieved signal efficiencies: %.2f", eff)
        return np.array(eff)[:, 0], np.array(eff)[:, 1]

    @property
    def bkg_eff(self):
        """Calculate background efficiency per bin.

        Returns
        -------
        np.ndarray
            Efficiency
        np.ndarray
            Efficiency_error
        """
        logger.debug("Calculating background efficiency.")
        eff = list(map(self.efficiency, self.disc_binned_bkg, self.disc_cut))
        logger.debug("Retrieved background efficiencies: %.2f", eff)
        return np.array(eff)[:, 0], np.array(eff)[:, 1]

    @property
    def sig_rej(self):
        """Calculate signal rejection per bin.

        Returns
        -------
        np.ndarray
            Rejection
        np.ndarray
            Rejection_error
        """
        logger.debug("Calculating signal rejection.")
        rej = list(map(self.rejection, self.disc_binned_sig, self.disc_cut))
        logger.debug("Retrieved signal rejections: %.1f", rej)
        return np.array(rej)[:, 0], np.array(rej)[:, 1]

    @property
    def bkg_rej(self):
        """Calculate background rejection per bin.

        Returns
        -------
        np.ndarray
            Rejection
        np.ndarray
            Rejection_error
        """
        logger.debug("Calculating background rejection.")
        rej = list(map(self.rejection, self.disc_binned_bkg, self.disc_cut))
        logger.debug("Retrieved background rejections: %.1f", rej)
        return np.array(rej)[:, 0], np.array(rej)[:, 1]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                np.all(self.x_var_sig == other.x_var_sig)
                and np.all(self.disc_sig == other.disc_sig)
                and np.all(self.x_var_bkg == other.x_var_bkg)
                and np.all(self.disc_bkg == other.disc_bkg)
                and np.all(self.bn_edges == other.bn_edges)
                and self.working_point == other.working_point
                and np.all(self.disc_cut == other.disc_cut)
                and self.fixed_eff_bin == other.fixed_eff_bin
                and self.key == other.key
            )
        return False

    def get(self, mode: str, inverse_cut: bool = False):
        """Wrapper around rejection and efficiency functions.

        Parameters
        ----------
        mode : str
            Can be "sig_eff", "bkg_eff", "sig_rej", "bkg_rej"
        inverse_cut : bool, optional
            Inverts the discriminant cut, which will yield the efficiency or rejection
            of the jets not passing the working point, by default False

        Returns
        -------
        np.ndarray
            Rejection or efficiency depending on `mode` value
        np.ndarray
            Rejection or efficiency error depending on `mode` value

        Raises
        ------
        ValueError
            If mode not supported
        """
        mode_options = ["sig_eff", "bkg_eff", "sig_rej", "bkg_rej"]
        self.inverse_cut = inverse_cut
        # TODO: python 3.10 switch to cases syntax
        if mode == "sig_eff":
            return self.sig_eff
        if mode == "bkg_eff":
            return self.bkg_eff
        if mode == "sig_rej":
            return self.sig_rej
        if mode == "bkg_rej":
            return self.bkg_rej
        # setting class variable again to False
        self.inverse_cut = False
        raise ValueError(
            f"The selected mode {mode} is not supported. Use one of the following: "
            f"{mode_options}."
        )

    def divide(
        self, other, mode: str, inverse: bool = False, inverse_cut: bool = False
    ):
        """Calculate ratio between two class objects.

        Parameters
        ----------
        other : var_vs_eff class
            Second var_vs_eff object to calculate ratio with
        mode : str
            Defines the mode which is used for the ratoi calculation, can be the
            following values: `sig_eff`, `bkg_eff`, `sig_rej`, `bkg_rej`
        inverse : bool
            If False the ratio is calculated `this / other`,
            if True the inverse is calculated
        inverse_cut : bool
            Inverts the discriminant cut, which will yield the efficiency or rejection
            of the jets not passing the working point, by default False

        Returns
        -------
        np.ndarray
            Ratio
        np.ndarray
            Ratio error
        np.ndarray
            Bin centres
        np.ndarray
            Bin widths

        Raises
        ------
        ValueError
            If binning is not identical between 2 objects
        """
        if not np.array_equal(self.bin_edges, other.bin_edges):
            raise ValueError("The binning of the two given objects do not match.")
        # TODO: python 3.10 switch to cases syntax
        nom, nom_err = self.get(mode, inverse_cut=inverse_cut)
        denom, denom_err = other.get(mode, inverse_cut=inverse_cut)

        ratio, ratio_err = hist_ratio(
            denom if inverse else nom,
            nom if inverse else denom,
            denom_err if inverse else nom_err,
            nom_err if inverse else denom_err,
            step=False,
        )
        return (
            ratio,
            ratio_err,
            self.x_bin_centres,
            self.bin_widths,
        )


class VarVsEffPlot(PlotBase):  # pylint: disable=too-many-instance-attributes
    """var_vs_eff plot class"""

    def __init__(self, mode, **kwargs) -> None:
        """var_vs_eff plot properties

        Parameters
        ----------
        mode : str
            Defines which quantity is plotted, the following options ar available:
            "sig_eff", "bkg_eff", "sig_rej" or "bkg_rej"
        **kwargs : kwargs
            Keyword arguments from `puma.PlotObject`

        Raises
        ------
        ValueError
            If incompatible mode given or more than 1 ratio panel requested
        """
        super().__init__(**kwargs)
        mode_options = ["sig_eff", "bkg_eff", "sig_rej", "bkg_rej"]
        if mode not in mode_options:
            raise ValueError(
                f"The selected mode {mode} is not supported. Use one of the following: "
                f"{mode_options}."
            )
        self.mode = mode
        self.plot_objects = {}
        self.add_order = []
        self.ratios_objects = {}
        self.ratio_axes = {}
        self.reference_object = None
        self.bin_edge_min = np.inf
        self.bin_edge_max = -np.inf
        self.inverse_cut = False
        if self.n_ratio_panels > 1:
            raise ValueError("Not more than one ratio panel supported.")
        self.initialise_figure(sub_plot_index=6)

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
        self.bin_edge_min = min(self.bin_edge_min, curve.bin_edges[0])
        self.bin_edge_max = max(self.bin_edge_max, curve.bin_edges[-1])

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
                "You specified a second curve %s as reference for ratio. "
                "Using it as new reference instead of %s.",
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
        logger.debug("Plotting curves with mode %s", self.mode)
        plt_handles = []
        for key in self.add_order:
            elem = self.plot_objects[key]
            y_value, y_error = elem.get(self.mode, inverse_cut=self.inverse_cut)
            error_bar = self.axis_top.errorbar(
                elem.x_bin_centres,
                y_value,
                xerr=elem.bin_widths,
                yerr=np.zeros(elem.n_bins),
                color=elem.colour,
                fmt="none",
                label=elem.label,
                alpha=elem.alpha,
                linewidth=elem.linewidth,
                **kwargs,
            )
            # set linestyle for errorbar
            error_bar[-1][0].set_linestyle(elem.linestyle)
            down_variation = y_value - y_error
            up_variation = y_value + y_error
            down_variation = np.concatenate((down_variation[:1], down_variation[:]))
            up_variation = np.concatenate((up_variation[:1], up_variation[:]))

            self.axis_top.fill_between(
                elem.bin_edges,
                down_variation,
                up_variation,
                color=elem.colour,
                alpha=0.3,
                zorder=1,
                step="pre",
                edgecolor="none",
                linestyle=elem.linestyle,
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
            (ratio, ratio_err, x_bin_centres, bin_widths,) = elem.divide(
                self.plot_objects[self.reference_object],
                mode=self.mode,
                inverse_cut=self.inverse_cut,
            )
            error_bar = self.axis_ratio_1.errorbar(
                x_bin_centres,
                ratio,
                xerr=bin_widths,
                yerr=np.zeros(elem.n_bins),
                color=elem.colour,
                fmt="none",
                alpha=elem.alpha,
                linewidth=elem.linewidth,
            )
            # set linestyle for errorbar
            error_bar[-1][0].set_linestyle(elem.linestyle)
            down_variation = ratio - ratio_err
            up_variation = ratio + ratio_err
            down_variation = np.concatenate((down_variation[:1], down_variation[:]))
            up_variation = np.concatenate((up_variation[:1], up_variation[:]))

            self.axis_ratio_1.fill_between(
                elem.bin_edges,
                down_variation,
                up_variation,
                color=elem.colour,
                alpha=0.3,
                zorder=1,
                step="pre",
                edgecolor="none",
            )

    def set_grid(self):
        """Set gtid lines."""
        self.axis_top.grid()
        self.axis_ratio_1.grid()

    def set_inverse_cut(self, inverse_cut=True):
        """Invert the discriminant cut, which will yield the efficiency or rejection
        of the jets not passing the working point.

        Parameters
        ----------
        inverse_cut : bool, optional
            Invert discriminant cut, by default True
        """
        self.inverse_cut = inverse_cut

    def draw_hline(self, y_val: float):
        """Draw hline in top plot panel.

        Parameters
        ----------
        y_val : float
            y value of the horizontal line
        """
        self.axis_top.hlines(
            y=y_val,
            xmin=self.bin_edge_min,
            xmax=self.bin_edge_max,
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
            self.bin_edge_min if self.xmin is None else self.xmin,
            self.bin_edge_max if self.xmax is None else self.xmax,
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
                self.axis_ratio_1,
                self.ylabel_ratio_1,
                align_right=False,
                labelpad=labelpad,
            )
        self.make_legend(plt_handles, ax_mpl=self.axis_top)
        if not self.atlas_tag_outside:
            self.tight_layout()
        self.plotting_done = True
        if self.apply_atlas_style is True:
            self.atlasify(use_tag=self.use_atlas_tag)
