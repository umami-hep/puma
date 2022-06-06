"""Tools for metrics module."""
import numpy as np

from puma.utils import logger
from puma.utils.histogram import hist_w_unc, save_divide


def calc_eff(
    sig_disc: np.ndarray,
    bkg_disc: np.ndarray,
    target_eff,
    return_cuts: bool = False,
):
    """Calculate efficiency

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

    Returns
    -------
    eff : float or np.ndarray
        Efficiency.
        If target_eff is a float, a float is returned if it's a list a np.ndarray
    cutvalue : float or np.ndarray
        Cutvalue if return_cuts is True.
        If target_eff is a float, a float is returned if it's a list a np.ndarray
    """
    # TODO: with python 3.10 using type union operator
    # float | np.ndarray for both target_eff and the returned values
    if isinstance(target_eff, float):
        cutvalue = np.percentile(sig_disc, 100.0 * (1.0 - target_eff))
        eff = save_divide(len(bkg_disc[bkg_disc > cutvalue]), len(bkg_disc), 0)

        if return_cuts:
            return eff, cutvalue
        return eff

    eff = np.zeros(len(target_eff))
    cutvalue = np.zeros(len(target_eff))
    for i, t_eff in enumerate(target_eff):
        cutvalue[i] = np.percentile(sig_disc, 100.0 * (1.0 - t_eff))
        eff[i] = save_divide(len(bkg_disc[bkg_disc > cutvalue[i]]), len(bkg_disc), 0)
    if return_cuts:
        return eff, cutvalue
    return eff


def calc_rej(
    sig_disc: np.ndarray,
    bkg_disc: np.ndarray,
    target_eff,
    return_cuts: bool = False,
):
    """Calculate efficiency

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

    Returns
    -------
    rej : float or np.ndarray
        Rejection.
        If target_eff is a float, a float is returned if it's a list a np.ndarray
    cut_value : float or np.ndarray
        Cutvalue if return_cuts is True.
        If target_eff is a float, a float is returned if it's a list a np.ndarray
    """
    # TODO: with python 3.10 using type union operator
    # float | np.ndarray for both target_eff and the returned values
    eff = calc_eff(
        sig_disc=sig_disc,
        bkg_disc=bkg_disc,
        target_eff=target_eff,
        return_cuts=return_cuts,
    )
    rej = save_divide(1, eff[0] if return_cuts else eff, np.inf)

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
    # TODO: suppress_zero_divison_error should not be necessary, but functions calling
    # eff_err seem to need this functionality - should be deprecated though.
    if np.any(n_counts <= 0) and not suppress_zero_divison_error:
        raise ValueError(
            f"You passed as argument `N` {n_counts} but it has to be larger 0."
        )
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
        raise ValueError(
            f"You passed as argument `n_counts` {n_counts} but it has to be larger 0."
        )
    if np.any(arr == 0):
        raise ValueError("One rejection value is 0, cannot calculate error.")
    if norm:
        return np.power(arr, 2) * eff_err(1 / arr, n_counts) / arr
    return np.power(arr, 2) * eff_err(1 / arr, n_counts)


def calc_separation(
    values_a: np.ndarray,
    values_b: np.ndarray,
    bins: int = 100,
    bins_range: tuple = None,
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

    _, bin_edges = np.histogram(
        np.hstack([values_a, values_b]), bins=bins, range=bins_range
    )

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
