"""Tools for metrics module."""
import numpy as np

from puma.utils import logger
from puma.utils.histogram import save_divide


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
        signal discriminant
    bkg_disc : np.ndarray
        background discriminant
    target_eff : float or list
         WP which is used for discriminant calculation
    return_cuts : bool
        Specifies if the cut values corresponding to the provided WPs are returned.
        If target_eff is a float, only one cut value will be returned. If target_eff
        is an array, target_eff is an array as well.

    Returns
    -------
    float or np.ndarray
        efficiency
        if target_eff is a float, a float is returned if it's a list a np.ndarray
    float or np.ndarray
        cutvalue if return_cuts is True
        if target_eff is a float, a float is returned if it's a list a np.ndarray
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
        signal discriminant
    bkg_disc : np.ndarray
        background discriminant
    target_eff : float or list
         WP which is used for discriminant calculation
    return_cuts : bool
        Specifies if the cut values corresponding to the provided WPs are returned.
        If target_eff is a float, only one cut value will be returned. If target_eff
        is an array, target_eff is an array as well.

    Returns
    -------
    float or np.ndarray
        rejection
        if target_eff is a float, a float is returned if it's a list a np.ndarray
    float or np.ndarray
        cutvalue if return_cuts is True
        if target_eff is a float, a float is returned if it's a list a np.ndarray
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
        efficiency values
    n_counts : int
        number of used statistics to calculate efficiency
    suppress_zero_divison_error : bool
        not raising Error for zero division
    norm : bool, optional
        if True, normed (relative) error is being calculated, by default False

    Returns
    -------
    numpy.array
        efficiency uncertainties

    Raises
    ------
    ValueError
        if n_counts <=0

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
        rejection values
    n_counts : int
        number of used statistics to calculate rejection
    norm : bool, optional
        if True, normed (relative) error is being calculated, by default False

    Returns
    -------
    numpy.array
        rejection uncertainties

    Raises
    ------
    ValueError
        if n_counts <=0
    ValueError
        if any rejection value is 0

    Notes
    -----
    special case of `eff_err()`
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
