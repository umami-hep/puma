"""Tools for metrics module."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from puma.utils import logger
from puma.utils.histogram import hist_w_unc, save_divide


def weighted_percentile(
    data: np.ndarray,
    perc: np.ndarray,
    weights: np.ndarray = None,
):
    """Calculate weighted percentile.

    Implementation according to https://stackoverflow.com/a/29677616/11509698
    (https://en.wikipedia.org/wiki/Percentile#The_weighted_percentile_method)

    Parameters
    ----------
    data : np.ndarray
        Data array
    perc : np.ndarray
        Percentile array
    weights : np.ndarray
        Weights array, by default None

    Returns
    -------
    np.ndarray
        Weighted percentile array
    """
    if weights is None:
        weights = np.ones_like(data)
    dtype = np.float64 if np.sum(weights) > 1000000 else np.float32
    ix = np.argsort(data)
    data = data[ix]  # sort data
    weights = weights[ix]  # sort weights
    cdf = np.cumsum(weights, dtype=dtype) - 0.5 * weights
    cdf -= cdf[0]
    cdf /= cdf[-1]
    return np.interp(perc, cdf, data)


def calc_eff(
    sig_disc: np.ndarray,
    bkg_disc: np.ndarray,
    target_eff: float | list | np.ndarray,
    return_cuts: bool = False,
    sig_weights: np.ndarray = None,
    bkg_weights: np.ndarray = None,
):
    """Calculate efficiency.

    Parameters
    ----------
    sig_disc : np.ndarray
        Signal discriminant
    bkg_disc : np.ndarray
        Background discriminant
    target_eff : float or list or np.ndarray
        Working point which is used for discriminant calculation
    return_cuts : bool
        Specifies if the cut values corresponding to the provided WPs are returned.
        If target_eff is a float, only one cut value will be returned. If target_eff
        is an array, target_eff is an array as well.
    sig_weights : np.ndarray
        Weights for signal events
    bkg_weights : np.ndarray
        Weights for background events

    Returns
    -------
    eff : float or np.ndarray
        Efficiency.
        Return float if target_eff is a float, else np.ndarray
    cutvalue : float or np.ndarray
        Cutvalue if return_cuts is True.
        Return float if target_eff is a float, else np.ndarray
    """
    # float | np.ndarray for both target_eff and the returned values
    return_float = False
    if isinstance(target_eff, float):
        return_float = True

    target_eff = np.asarray([target_eff]).flatten()

    cutvalue = weighted_percentile(sig_disc, 1.0 - target_eff, weights=sig_weights)
    sorted_args = np.argsort(1 - target_eff)  # need to sort the cutvalues to get the correct order
    hist, _ = np.histogram(bkg_disc, (-np.inf, *cutvalue[sorted_args], np.inf), weights=bkg_weights)
    eff = hist[::-1].cumsum()[-2::-1] / hist.sum()
    eff = eff[sorted_args]

    if return_float:
        eff = eff[0]
        cutvalue = cutvalue[0]

    if return_cuts:
        return eff, cutvalue
    return eff


def calc_rej(
    sig_disc: np.ndarray,
    bkg_disc: np.ndarray,
    target_eff,
    return_cuts: bool = False,
    sig_weights: np.ndarray = None,
    bkg_weights: np.ndarray = None,
    smooth: bool = False,
):
    """Calculate efficiency.

    Parameters
    ----------
    sig_disc : np.ndarray
        Signal discriminant
    bkg_disc : np.ndarray
        Background discriminant
    target_eff : float or list
        Working point which is used for discriminant calculation
    return_cuts : bool
        Specifies if the cut values corresponding to the provided WPs are returned.
        If target_eff is a float, only one cut value will be returned. If target_eff
        is an array, target_eff is an array as well.
    sig_weights : np.ndarray
        Weights for signal events, by default None
    bkg_weights : np.ndarray
        Weights for background events, by default None

    Returns
    -------
    rej : float or np.ndarray
        Rejection.
        If target_eff is a float, a float is returned if it's a list a np.ndarray
    cut_value : float or np.ndarray
        Cutvalue if return_cuts is True.
        If target_eff is a float, a float is returned if it's a list a np.ndarray
    """
    # float | np.ndarray for both target_eff and the returned values
    eff = calc_eff(
        sig_disc=sig_disc,
        bkg_disc=bkg_disc,
        target_eff=target_eff,
        return_cuts=return_cuts,
        sig_weights=sig_weights,
        bkg_weights=bkg_weights,
    )
    rej = save_divide(1, eff[0] if return_cuts else eff, np.inf)
    if smooth:
        rej = gaussian_filter1d(rej, sigma=1, radius=2, mode="nearest")
    if return_cuts:
        return rej, eff[1]
    return rej


def eff_err(
    arr: np.ndarray,
    n_counts: int,
    suppress_zero_divison_error: bool = False,
    norm: bool = False,
) -> np.ndarray:
    """Calculate statistical efficiency uncertainty.

    Parameters
    ----------
    arr : numpy.array
        Efficiency values
    n_counts : int
        Number of used statistics to calculate efficiency
    suppress_zero_divison_error : bool
        Not raising Error for zero division
    norm : bool, optional
        If True, normed (relative) error is being calculated, by default False

    Returns
    -------
    numpy.array
        Efficiency uncertainties

    Raises
    ------
    ValueError
        If n_counts <=0

    Notes
    -----
    This method uses binomial errors as described in section 2.2 of
    https://inspirehep.net/files/57287ac8e45a976ab423f3dd456af694
    """
    logger.debug("Calculating efficiency error.")
    logger.debug("arr: %s", arr)
    logger.debug("n_counts: %i", n_counts)
    logger.debug("suppress_zero_divison_error: %s", suppress_zero_divison_error)
    logger.debug("norm: %s", norm)
    if np.any(n_counts <= 0) and not suppress_zero_divison_error:
        raise ValueError(f"You passed as argument `N` {n_counts} but it has to be larger 0.")
    if norm:
        return np.sqrt(arr * (1 - arr) / n_counts) / arr
    return np.sqrt(arr * (1 - arr) / n_counts)


def rej_err(
    arr: np.ndarray,
    n_counts: int,
    norm: bool = False,
) -> np.ndarray:
    """Calculate the rejection uncertainties.

    Parameters
    ----------
    arr : numpy.array
        Rejection values
    n_counts : int
        Number of used statistics to calculate rejection
    norm : bool, optional
        If True, normed (relative) error is being calculated, by default False

    Returns
    -------
    numpy.array
        Rejection uncertainties

    Raises
    ------
    ValueError
        If n_counts <=0
    ValueError
        If any rejection value is 0

    Notes
    -----
    Special case of `eff_err()`
    """
    logger.debug("Calculating rejection error.")
    logger.debug("arr: %s", arr)
    logger.debug("n_counts: %i", n_counts)
    logger.debug("norm: %s", norm)
    if np.any(n_counts <= 0):
        raise ValueError(f"You passed as argument `n_counts` {n_counts} but it has to be larger 0.")
    if np.any(arr == 0):
        raise ValueError("One rejection value is 0, cannot calculate error.")
    if norm:
        return np.power(arr, 2) * eff_err(1 / arr, n_counts) / arr
    return np.power(arr, 2) * eff_err(1 / arr, n_counts)


def calc_separation(
    values_a: np.ndarray,
    values_b: np.ndarray,
    bins: int = 100,
    bins_range: tuple | None = None,
    return_hist: bool = False,
) -> float:
    """Calculates the separation of two distributions.

    Parameters
    ----------
    values_a : np.ndarray
        First distribution
    values_b : np.ndarray
        Second distribution
    bins : int, optional
        Number of bins used for the histograms (common binning over whole range if
        `bins_range` is not defined), by default 100
    bins_range : tuple, optional
        Lower and upper limit for binning. If not provided, the whole range of the
        joined distribution is used, by default None
    return_hist : bool, optional
        Option to also return the hist_a, hist_b and bin_edges arrays

    Returns
    -------
    float
        separation value
    float
        separation uncertainty
    numpy.ndarray
        Bins height of histogram a (only returned if `return_hist` is True)
    numpy.ndarray
        Bins height of histogram b (only returned if `return_hist` is True)
    numpy.ndarray
        Bin edges of the two histograms (only returned if `return_hist` is True)
    """
    _, bin_edges = np.histogram(np.hstack([values_a, values_b]), bins=bins, range=bins_range)

    _, hist_a, unc_a, _ = hist_w_unc(values_a, bin_edges)
    _, hist_b, unc_b, _ = hist_w_unc(values_b, bin_edges)

    separation = 0.5 * np.sum(
        save_divide(
            numerator=(hist_a - hist_b) ** 2,
            denominator=(hist_a + hist_b),
            default=0,
        )
    )

    # helper variable for error calculation below
    ab_diff_over_sum = save_divide(hist_a - hist_b, hist_a + hist_b, default=0)

    separation_uncertainty = 0.5 * np.sqrt(
        np.sum(
            (unc_a * (2 * ab_diff_over_sum - ab_diff_over_sum**2)) ** 2
            + (unc_b * (-2 * ab_diff_over_sum - ab_diff_over_sum**2)) ** 2
        )
    )

    if return_hist:
        return separation, separation_uncertainty, hist_a, hist_b, bin_edges
    return separation, separation_uncertainty
