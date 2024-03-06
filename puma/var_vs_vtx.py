"""Aux task efficiency plots vs. specific variable."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from puma.metrics import eff_err
from puma.utils import logger
from puma.utils.histogram import save_divide
from puma.var_vs_var import VarVsVar, VarVsVarPlot


class VarVsVtx(VarVsVar):  # pylint: disable=too-many-instance-attributes
    """var_vs_vtx class storing info about vertexing performance."""

    def __init__(
        self,
        x_var: np.ndarray,
        n_match: np.ndarray,
        n_true: np.ndarray,
        n_reco: np.ndarray,
        bins=10,
        key: str | None = None,
        **kwargs,
    ) -> None:
        """Initialise properties of roc curve object.

        Parameters
        ----------
        x_var : np.ndarray
            Values for x-axis variable for signal
        n_match : np.ndarray
            Values for number of correctly identified objects (where truth and
            reco match)
        n_true : np.ndarray
            Values for true number of objects
        n_reco : np.ndarray
            Values for reconstructed number of objects
        bins : int or sequence of scalars, optional
            If bins is an int, it defines the number of equal-width bins in the
            given range (10, by default). If bins is a sequence, it defines a
            monotonically increasing array of bin edges, including the
            rightmost edge, allowing for non-uniform bin widths, by default 10
        key : str, optional
            Identifier for the curve e.g. tagger, by default None
        **kwargs : kwargs
            Keyword arguments passed to `PlotLineObject`

        Raises
        ------
        ValueError
            If provided options are not compatible with each other
        """
        if len(x_var) != len(n_match):
            raise ValueError(
                f"Length of `x_var` ({len(x_var)}) and `n_match` "
                f"({len(n_match)}) have to be identical."
            )
        if len(x_var) != len(n_true):
            raise ValueError(
                f"Length of `x_var` ({len(x_var)}) and `n_true` "
                f"({len(n_true)}) have to be identical."
            )
        if len(x_var) != len(n_reco):
            raise ValueError(
                f"Length of `x_var` ({len(x_var)}) and `n_reco` "
                f"({len(n_reco)}) have to be identical."
            )

        self.x_var = np.array(x_var)
        self.n_match = np.array(n_match)
        self.n_true = np.array(n_true)
        self.n_reco = np.array(n_reco)
        # Binning related variables
        self.n_bins = None
        self.bin_edges = None
        self.x_bin_centres = None
        self.bin_widths = None
        # Binned distributions
        self.bin_indices = None
        self.metric_binned = None

        self._set_bin_edges(bins)
        self._apply_binning()

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
            xmin, xmax = np.amin(self.x_var), np.amax(self.x_var)
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
        """Get binned distributions for number of matches, truth and reco objects."""
        logger.debug("Applying binning.")
        self.bin_indices = np.digitize(self.x_var, self.bin_edges)
        self.match_binned = [
            self.n_match[np.where(self.bin_indices == x)[0]] for x in range(1, len(self.bin_edges))
        ]
        self.true_binned = [
            self.n_true[np.where(self.bin_indices == x)[0]] for x in range(1, len(self.bin_edges))
        ]
        self.reco_binned = [
            self.n_reco[np.where(self.bin_indices == x)[0]] for x in range(1, len(self.bin_edges))
        ]

    def get_performance_ratio(self, num: np.ndarray, denom: np.ndarray):
        """Calculate performance ratio for vertexing task. Either n_matched/n_true
        (efficiency) or n_matched/n_reco (purity).

        Parameters
        ----------
        arr : np.ndarray
            Array with discriminants
        cut : float
            Cut value

        Returns
        -------
        float
            Performance ratio
        float
            Performance ratio error
        """
        pm = save_divide(np.sum(num), np.sum(denom), default=np.inf)
        if pm == np.inf:
            logger.warning("Your vertexing performance ratio is infinity -> setting it to np.nan.")
            return np.nan, np.nan
        if pm == 0:
            logger.warning("Your vertexing performance ratio is zero -> setting error to zero.")
            return 0.0, 0.0
        pm_error = eff_err(pm, len(num))
        return pm, pm_error

    @property
    def efficiency(self):
        """Calculate vertexing efficiency per bin. Defined as number of reconstructed
        vertices matched to truth divided by number of total true vertices.

        Returns
        -------
        np.ndarray
            Efficiency
        np.ndarray
            Efficiency error
        """
        logger.debug("Calculating vertexing efficiency.")
        eff = list(
            map(
                self.get_performance_ratio,
                self.match_binned,
                self.true_binned,
            )
        )
        logger.debug("Retrieved vertexing efficiencies: %s", eff)
        return np.array(eff)[:, 0], np.array(eff)[:, 1]

    @property
    def purity(self):
        """Calculate vertexing purity per bin. Defined as number of reconstructed
        vertices matched to truth divided by number of total reconstructed vertices.

        Returns
        -------
        np.ndarray
            Purity
        np.ndarray
            Purity error
        """
        logger.debug("Calculating vertexing purity.")
        purity = list(
            map(
                self.get_performance_ratio,
                self.match_binned,
                self.reco_binned,
            )
        )
        logger.debug("Retrieved vertexing purity: %s", purity)
        return np.array(purity)[:, 0], np.array(purity)[:, 1]

    @property
    def fakes(self):
        """Calculate vertexing fake rate per bin. Defined as total number
        of events with reconstructed vertices where vertices are not expected.

        Returns
        -------
        np.ndarray
            Fake rate
        np.ndarray
            Fake rate error
        """
        logger.debug("Calculating vertexing fake rate.")
        total_reco = list(
            map(
                self.get_performance_ratio,
                [np.where(reco_bin > 0, 1, 0) for reco_bin in self.reco_binned],
                list(map(np.ones_like, self.reco_binned)),
            )
        )
        logger.debug("Retrieved vertexing fake rate: %s", total_reco)
        return np.array(total_reco)[:, 0], np.array(total_reco)[:, 1]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                np.all(self.x_var == other.x_var)
                and np.all(self.n_match == other.n_match)
                and np.all(self.n_true == other.n_true)
                and np.all(self.n_reco == other.n_reco)
                and np.all(self.bin_edges == other.bin_edges)
                and self.key == other.key
            )
        return False

    def get(self, mode: str):
        """Wrapper around rejection and efficiency functions.

        Parameters
        ----------
        mode : str
            Can be "efficiency", "purity" or "fakes"

        Returns
        -------
        np.ndarray
            Efficiency, purity or fake rate depending on `mode` value
        np.ndarray
            Efficiency, purity or fake rate error depending on `mode` value

        Raises
        ------
        ValueError
            If mode not supported
        """
        if mode == "efficiency":
            return self.efficiency
        if mode == "purity":
            return self.purity
        if mode == "fakes":
            return self.fakes
        raise ValueError(
            f"The selected mode {mode} is not supported. Use one of the following:"
            f" {VarVsVtxPlot.mode_options}."
        )


class VarVsVtxPlot(VarVsVarPlot):  # pylint: disable=too-many-instance-attributes
    mode_options: ClassVar[list[str]] = [
        "efficiency",
        "purity",
        "fakes",
    ]

    def __init__(self, mode, grid: bool = False, **kwargs) -> None:
        """var_vs_vtx plot properties.

        Parameters
        ----------
        mode : str
            Defines which quantity is plotted, the following options ar available:
                efficiency - Plots efficiency vs. variable for jets where vertices are
                expected
                purity - Plots purity vs. variable for jets where vertices are expected
                fakes - Plots fake rate vs. variable for jets where vertices are not
                expected
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
            y_value, y_error = elem.get(self.mode)
            elem.y_var_mean = y_value
            elem.y_var_std = y_error

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
        logger.debug("Plotting curves with mode %s", self.mode)
        self._setup_curves()
        return super().plot(**kwargs)
