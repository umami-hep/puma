"""Efficiency plots vs. specific variable."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from ftag.utils import calculate_efficiency_error, calculate_rejection_error

from puma.utils import logger
from puma.utils.histogram import save_divide
from puma.var_vs_var import VarVsVar, VarVsVarPlot

if TYPE_CHECKING:  # pragma: no cover
    from ftag.labels import Label


class VarVsEff(VarVsVar):  # pylint: disable=too-many-instance-attributes
    """Class for efficiency vs. variable plot."""

    def __init__(
        self,
        x_var_sig: np.ndarray,
        disc_sig: np.ndarray,
        x_var_bkg: np.ndarray = None,
        disc_bkg: np.ndarray = None,
        bins: int | list | np.ndarray = 10,
        working_point: float | list | None = None,
        fixed_bkg_rej: float | None = None,
        disc_cut: int | list | np.ndarray | None = None,
        flat_per_bin: bool = False,
        key: str | None = None,
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
        bins : int, optional
            If bins is an int, it defines the number of equal-width bins in the
            given range (10, by default). If bins is a sequence, it defines a
            monotonically increasing array of bin edges, including the
            rightmost edge, allowing for non-uniform bin widths, by default 10
        working_point : float | list | None, optional
            Working point, by default None
        fixed_bkg_rej : float | None, optional
            Instead of fixing the signal efficiency (by using working_point), fix the
            background rejection to a certain value. By default None
        disc_cut : int | list | np.ndarray | None, optional
            Cut value for discriminant, if it is a sequence it has to have the same
            length as number of bins, by default None
        flat_per_bin : bool, optional
            If True and no `disc_cut` is given the signal efficiency is held constant
            in each bin, by default False
        key : str | None, optional
            Identifier for the curve e.g. tagger, by default None
        **kwargs : kwargs
            Keyword arguments passed to `PlotLineObject`

        Raises
        ------
        ValueError
            x_var_sig and disc_sig have different lengths
        ValueError
            x_var_bkg and disc_bkg have different lengths
        ValueError
            Neither working_point nor flat_per_bin was set
        ValueError
            Both disc_cut and flat_per_bin are set
        ValueError
            working_point is not set but flat_per_bin is
        ValueError
            Using PCFT bins and flat_per_bin together
        ValueError
            Using disc_cut and working_point together
        ValueError
            disc_cut (if an array) has a different length than the number of bins given
        """
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

        # Ensure that signal/background variables are arrays
        self.x_var_sig = np.array(x_var_sig)
        self.disc_sig = np.array(disc_sig)
        self.x_var_bkg = None if x_var_bkg is None else np.array(x_var_bkg)
        self.disc_bkg = None if disc_bkg is None else np.array(disc_bkg)

        # Define attributes for the working points
        self.working_point = working_point
        self.fixed_bkg_rej = fixed_bkg_rej
        self.disc_cut = disc_cut
        self.flat_per_bin = flat_per_bin

        # Binning related variables
        self.n_bins = None
        self.bin_edges = None
        self.x_bin_centres = None
        self.bin_widths = None
        self.n_bins = None

        # Binned distributions
        self.bin_indices_sig = None
        self.disc_binned_sig = None
        self.bin_indices_bkg = None
        self.disc_binned_bkg = None

        # Kwargs
        self.kwargs = kwargs

        # Set the bin edges
        self._set_bin_edges(bins)

        # Check that either disc_cut, working_point, or fixed_bkg_rej are defined
        not_none = [
            name
            for name, val in (
                ("disc_cut", self.disc_cut),
                ("working_point", self.working_point),
                ("fixed_bkg_rej", self.fixed_bkg_rej),
            )
            if val is not None
        ]

        if len(not_none) != 1:
            raise ValueError(
                f"Exactly one of disc_cut, working_point, fixed_bkg_rej must be provided! "
                f"Got {', '.join(not_none) if not_none else 'none'}."
            )

        if self.disc_cut is not None:
            # Ensure that flat_per_bin cannot be set
            if self.flat_per_bin:
                raise ValueError(
                    "You cannot specify `disc_cut` when `flat_per_bin` is set to True."
                )

            # Ensure that disc_cut has the same length as the number of bins if it's used
            if isinstance(disc_cut, (list, np.ndarray)) and self.n_bins != len(disc_cut):
                raise ValueError(
                    "`disc_cut` has to be a float or has to have the same length as"
                    " number of bins."
                )

        elif self.working_point is not None:
            # Ensure that only a one float is given to working_point with flat_per_bin
            if self.flat_per_bin and not isinstance(working_point, float):
                raise ValueError("working_point must be a float when running with flat_per_bin!")

            # Ensure that the given working points are an array
            if isinstance(self.working_point, list):
                self.working_point = np.asarray(self.working_point)

            # Throw error if the working points are neither a float nor a list
            elif not isinstance(working_point, float):
                raise TypeError(
                    "`working_point` must either be a list or a float! "
                    f"You gave {type(self.working_point)}"
                )

        elif self.fixed_bkg_rej is not None:
            # Ensure that fixed_bkg_rej is a float:
            if not isinstance(fixed_bkg_rej, (int, float)):
                raise TypeError(
                    "`fixed_bkg_rej` must be an int or a float!"
                    f" You gave {type(self.fixed_bkg_rej)}"
                )

        # Apply the binning and get the discriminant value cuts as attributes
        self._apply_binning()
        self._get_disc_cuts()

        # Setup some needed attributes and also check them
        VarVsVar.__init__(
            self,
            x_var=self.x_bin_centres,
            y_var_mean=np.zeros_like(self.x_bin_centres),
            y_var_std=np.zeros_like(self.x_bin_centres),
            x_var_widths=2 * self.bin_widths,
            key=key,
            fill=True,
            plot_y_std=False,
            **kwargs,
        )

        # Calculate all efficiencies/rejections possible to make them easily available and storable
        self.results = {"normal": {}, "inverse": {}}

        # Iterate over inverse and normal setup
        for iter_key in self.results:
            self.inverse_cut = iter_key == "inverse"

            # Iterate over the different output modes and store them
            for iter_mode in VarVsEffPlot.mode_options:
                self.results[iter_key][iter_mode] = {}

                # We can't calculate always everything, so the ones that
                # make problems will be filtered
                try:
                    (
                        self.results[iter_key][iter_mode]["y_value"],
                        self.results[iter_key][iter_mode]["y_error"],
                    ) = getattr(self, iter_mode)

                except (TypeError, ValueError):
                    (
                        self.results[iter_key][iter_mode]["y_value"],
                        self.results[iter_key][iter_mode]["y_error"],
                    ) = (None, None)

        # Set inverse_cut back to false
        self.inverse_cut = None

    def _set_bin_edges(self, bins: int | list | np.ndarray):
        """Calculate bin edges, centres and width and save them as class variables.

        Parameters
        ----------
        bins : int | list | np.ndarray
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
        logger.debug(f"Retrieved bin edges: {self.bin_edges}")
        # Get the bins for the histogram
        self.x_bin_centres = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
        self.bin_widths = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2.0
        self.n_bins = self.bin_edges.size - 1
        logger.debug(f"N bins: {self.n_bins}")

    def _apply_binning(self):
        """Get binned distributions for the signal and background."""
        logger.debug("Applying binning.")
        self.bin_indices_sig = np.digitize(self.x_var_sig, self.bin_edges)
        if np.all(self.bin_indices_sig == 0):
            logger.error("All your signal is in the underflow bin. Check your input.")
            # retrieve for each bin the part of self.disc_sig corresponding to this bin
            # and put them in a list
        self.disc_binned_sig = [
            self.disc_sig[np.where(self.bin_indices_sig == x)[0]]
            for x in range(1, len(self.bin_edges))
        ]
        if self.x_var_bkg is not None:
            self.bin_indices_bkg = np.digitize(self.x_var_bkg, self.bin_edges)
            self.disc_binned_bkg = [
                self.disc_bkg[np.where(self.bin_indices_bkg == x)[0]]
                for x in range(1, len(self.bin_edges))
            ]

    def _get_disc_cuts(self):
        """Retrieve cut values on discriminant. If `disc_cut` is not given, retrieve
        cut values from the working point.
        """
        logger.debug("Calculate discriminant cut.")
        if isinstance(self.disc_cut, (float, int)):
            self.disc_cut = [self.disc_cut] * self.n_bins
        elif isinstance(self.disc_cut, (list, np.ndarray)):
            self.disc_cut = self.disc_cut
        elif isinstance(self.working_point, (list, np.ndarray)):
            self.disc_cut = np.column_stack((
                [np.percentile(self.disc_sig, (1 - self.working_point[0]) * 100)] * self.n_bins,
                [np.percentile(self.disc_sig, (1 - self.working_point[1]) * 100)] * self.n_bins,
            ))
        elif self.flat_per_bin:
            if isinstance(self.working_point, float):
                self.disc_cut = [
                    float(np.percentile(x, (1 - self.working_point) * 100))
                    for x in self.disc_binned_sig
                ]
            elif isinstance(self.fixed_bkg_rej, (int, float)):
                self.disc_cut = [
                    float(np.percentile(x, (1 - (1 / self.fixed_bkg_rej)) * 100))
                    for x in self.disc_binned_bkg
                ]
        elif isinstance(self.fixed_bkg_rej, (int, float)):
            self.disc_cut = [
                float(np.percentile(self.disc_bkg, (1 - (1 / self.fixed_bkg_rej)) * 100))
            ] * self.n_bins
        else:
            self.disc_cut = [
                float(np.percentile(self.disc_sig, (1 - self.working_point) * 100))
            ] * self.n_bins
        logger.debug(f"Discriminant cut: {self.disc_cut}")

    def efficiency(self, arr: np.ndarray, cut: float | np.ndarray):
        """Calculate efficiency and the associated error.

        Parameters
        ----------
        arr : np.ndarray
            Array with discriminants
        cut : float | np.ndarray
            Cut value. If you want to use PCFT, two values are provided.
            The lower and the upper cut.

        Returns
        -------
        float
            Efficiency
        float
            Efficiency error
        """
        if len(arr) == 0:
            return 0, 0

        if isinstance(cut, (int, float, np.int32, np.float32)):
            eff = sum(arr < cut) / len(arr) if self.inverse_cut else sum(arr > cut) / len(arr)

        elif isinstance(cut, np.ndarray):
            eff = sum((arr < cut[0]) & (arr > cut[1])) / len(arr)

        else:
            raise TypeError(
                f"cut parameter type {type(cut)} is not supported! Must be float or np.ndarray"
            )

        eff_error = calculate_efficiency_error(eff, len(arr))
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

        elif isinstance(cut, (int, float, np.int32, np.float32)):
            rej = save_divide(len(arr), sum(arr > cut), default=np.inf)

        elif isinstance(cut, np.ndarray):
            rej = save_divide(len(arr), sum((arr < cut[0]) & (arr > cut[1])), default=np.inf)

        else:
            raise TypeError(
                f"`cut` parameter type {type(cut)} is not supported! Must be float or np.ndarray!"
            )

        if rej == np.inf:
            logger.warning("Your rejection is infinity -> setting it to np.nan.")
            return np.nan, np.nan

        rej_error = calculate_rejection_error(rej, len(arr))
        return rej, rej_error

    @property
    def args_to_store(self) -> dict[str, Any]:
        """Returns the arguments that need to be stored/loaded.

        Returns
        -------
        dict[str, Any]
            Dict with the arguments
        """
        # Copy the kwargs to remove safely stuff
        extra_kwargs = dict(getattr(self, "kwargs", {}))

        # Create the dict with the args to store/load
        return {
            "results": self.results,
            "working_point": self.working_point,
            "fixed_bkg_rej": self.fixed_bkg_rej,
            "disc_cut": self.disc_cut,
            "flat_per_bin": self.flat_per_bin,
            "x_var": self.x_var,
            "y_var_mean": self.y_var_mean,
            "y_var_std": self.y_var_std,
            "x_var_widths": self.x_var_widths,
            "key": self.key,
            "fill": self.fill,
            "plot_y_std": self.plot_y_std,
            "ratio_group": self.ratio_group,
            **extra_kwargs,
        }

    @property
    def bkg_eff_sig_err(self):
        """Calculate signal efficiency per bin, assuming a flat background per
        bin. This results in returning the signal efficiency per bin, but the
        background error per bin.
        """
        logger.debug("Calculating signal efficiency.")
        eff = np.array(list(map(self.efficiency, self.disc_binned_bkg, self.disc_cut)))[:, 0]
        err = np.array(list(map(self.efficiency, self.disc_binned_sig, self.disc_cut)))[:, 1]
        logger.debug(f"Retrieved signal efficiencies: {eff}")
        return eff, err

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
        logger.debug(f"Retrieved signal efficiencies: {eff}")
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
        logger.debug(f"Retrieved background efficiencies: {eff}")
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
        logger.debug(f"Retrieved signal rejections: {rej}")
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
        logger.debug(f"Retrieved background rejections: {rej}")
        return np.array(rej)[:, 0], np.array(rej)[:, 1]

    def get(self, mode: str, inverse_cut: bool = False):
        """Wrapper around rejection and efficiency functions.

        Parameters
        ----------
        mode : str
            Can be "sig_eff", "bkg_eff", "sig_rej", "bkg_rej", or
            "bkg_eff_sig_err"
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
        if mode in VarVsEffPlot.mode_options:
            name = "normal" if inverse_cut is False else "inverse"
            return (
                self.results[name][mode]["y_value"],
                self.results[name][mode]["y_error"],
            )
        raise ValueError(
            f"The selected mode {mode} is not supported. Use one of the following:"
            f" {VarVsEffPlot.mode_options}."
        )


class VarVsEffPlot(VarVsVarPlot):  # pylint: disable=too-many-instance-attributes
    """var_vs_eff plot class."""

    mode_options: ClassVar[list[str]] = [
        "sig_eff",
        "bkg_eff",
        "sig_rej",
        "bkg_rej",
        "bkg_eff_sig_err",
    ]

    def __init__(self, mode, grid: bool = False, **kwargs) -> None:
        """var_vs_eff plot properties.

        Parameters
        ----------
        mode : str
            Defines which quantity is plotted, the following options ar available:
                sig_eff - Plots signal efficiency vs. variable, with statistical error
                    on N signal per bin
                bkg_eff - Plots background efficiency vs. variable, with statistical
                    error on N background per bin
                sig_rej - Plots signal rejection vs. variable, with statistical error
                    on N signal per bin
                bkg_rej - Plots background rejection vs. variable, with statistical
                    error on N background per bin
                bkg_eff_sig_err - Plots background efficiency vs. variable, with
                    statistical error on N signal per bin.
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
        if mode not in self.mode_options:
            raise ValueError(
                f"The selected mode {mode} is not supported. Use one of the following: "
                f"{self.mode_options}."
            )
        self.mode = mode

    def _setup_curves(self):
        for key in self.add_order:
            elem = self.plot_objects[key]

            # Get the results dict from the object
            results_dict = elem.results

            # Check if the "normal" values or the inverse values are needed
            type_key = "normal" if self.inverse_cut is False else "inverse"

            # Get the y_values and errors
            elem.y_var_mean = results_dict[type_key][self.mode]["y_value"]
            elem.y_var_std = results_dict[type_key][self.mode]["y_error"]

    def apply_modified_atlas_second_tag(
        self,
        signal: Label | None = None,
        background: Label | None = None,
        working_point: float | list | None = None,
        fixed_bkg_rej: float | None = None,
        disc_cut: float | None = None,
        flat_per_bin: bool = False,
    ):
        """Modifies the atlas_second_tag to include info on the type of p-eff plot.

        Parameters
        ----------
        signal : Label | None, optional
            Signal flavour that is used. By default None
        background : Label | None, optional
            Background flavour that is used. By default None
        working_point : float | list | None, optional
            Working point that was used. When PCFT was used, this is a list of length 2.
            If disc_cut/fixed_bkg_rej is used, this needs to be None, by default None
        fixed_bkg_rej : float | None, optional
            The background rejection to which it was fixed. Must be None when working_point
            or disc_cut is used. By default None
        disc_cut : float | None, optional
            Discriminant cut that was used. If working_point/fixed_bkg_rej is used,
            this must be None. By default None
        flat_per_bin : bool, optional
            If flat_per_bin was used, by default False
        """
        if working_point:
            if isinstance(working_point, list):
                mid_str = (
                    f"{int(round(working_point[0] * 100, 0))}% - "
                    f"{int(round(working_point[1] * 100, 0))}% " + signal.eff_str
                )

            else:
                mid_str = f"{int(round(working_point * 100, 0))}% " + signal.eff_str
        elif disc_cut:
            mid_str = rf"$D_{{{signal.name.rstrip('jets')}}}$ > {disc_cut}"
        elif fixed_bkg_rej:
            mid_str = f"{int(round(fixed_bkg_rej, 0))} " + background.rej_str
        tag = f"Flat {mid_str} per bin" if flat_per_bin else f"{mid_str}"
        if self.atlas_second_tag:
            self.atlas_second_tag = f"{self.atlas_second_tag}\n{tag}"
        else:
            self.atlas_second_tag = tag

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
        logger.debug(f"Plotting curves with mode {self.mode}")
        self._setup_curves()
        return super().plot(**kwargs)
