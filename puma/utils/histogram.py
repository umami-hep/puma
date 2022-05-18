"""Helper function for histogram handling."""
import numpy as np


def save_divide(
    numerator,
    denominator,
    default: float = 1.0,
):
    """
    Division using numpy divide function returning default value in cases where
    denominator is 0.

    Parameters
    ----------
    numerator: array_like, int, float
        Numerator in the ratio calculation.
    denominator: array_like, int, float
        Denominator in the ratio calculation.
    default: float
        Default value which is returned if denominator is 0.

    Returns
    -------
    ratio: array_like, float
    """
    if isinstance(numerator, (int, float, np.number)) and isinstance(
        denominator, (int, float, np.number)
    ):
        output_shape = 1
    else:
        try:
            output_shape = denominator.shape
        except AttributeError:
            output_shape = numerator.shape

    ratio = np.divide(
        numerator,
        denominator,
        out=np.ones(
            output_shape,
            dtype=float,
        )
        * default,
        where=(denominator != 0),
    )
    if output_shape == 1:
        return float(ratio)
    return ratio


def hist_w_unc(
    arr,
    bins,
    bins_range=None,
    normed: bool = True,
):
    """
    Computes histogram and the associated statistical uncertainty.

    Parameters
    ----------
    arr : array_like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars or str
        bins parameter from np.histogram
    bins_range : tuple, optional
        range parameter from np.histogram
    normed : bool, optional
        If True (default) the calculated histogram is normalised to an integral
        of 1.

    Returns
    -------
    bin_edges : array of dtype float
        Return the bin edges (length(hist)+1)
    hist : numpy array
        The values of the histogram. If normed is true (default), returns the
        normed counts per bin
    unc : numpy array
        Statistical uncertainty per bin.
        If normed is true (default), returns the normed values.
    band : numpy array
        lower uncertainty band location: hist - unc
        If normed is true (default), returns the normed values.
    """
    arr_length = len(arr)

    # Calculate the counts and the bin edges
    counts, bin_edges = np.histogram(arr, bins=bins, range=bins_range)

    unc = save_divide(np.sqrt(counts), arr_length, 0) if normed else np.sqrt(counts)
    band = save_divide(counts, arr_length, 0) - unc if normed else counts - unc
    hist = save_divide(counts, arr_length, 0) if normed else counts

    return bin_edges, hist, unc, band


def hist_ratio(
    numerator,
    denominator,
    numerator_unc,
    denominator_unc,
    step: bool = True,
):
    """
    This method calculates the ratio of the given bincounts and
    returns the input for a step function that plots the ratio.

    Parameters
    ----------
    numerator : array_like
        Numerator in the ratio calculation.
    denominator : array_like
        Denominator in the ratio calculation.
    numerator_unc : array_like
        Uncertainty of the numerator.
    denominator_unc : array_like
        Uncertainty of the denominator.
    step : bool
        if True duplicates first bin to match with step plotting function,
        by default True


    Returns
    -------
    step_ratio : array_like
        Ratio returning 1 in case the denominator is 0.
    step_ratio_unc : array_like
        Stat. uncertainty of the step_ratio

    Raises
    ------
    AssertionError
        If inputs don't have the same shape.

    """
    numerator, denominator, numerator_unc, denominator_unc = (
        np.array(numerator),
        np.array(denominator),
        np.array(numerator_unc),
        np.array(denominator_unc),
    )
    if numerator.shape != denominator.shape:
        raise AssertionError("Numerator and denominator don't have the same legth")
    if numerator.shape != numerator_unc.shape:
        raise AssertionError("Numerator and numerator_unc don't have the same legth")
    if denominator.shape != denominator_unc.shape:
        raise (
            AssertionError("Denominator and denominator_unc don't have the same legth")
        )
    step_ratio = save_divide(numerator, denominator, 1 if step else np.inf)

    # Calculate rel uncertainties
    numerator_rel_unc = save_divide(
        numerator_unc, numerator, default=0 if step else np.inf
    )
    denominator_rel_unc = save_divide(
        denominator_unc, denominator, default=0 if step else np.inf
    )

    # Calculate rel uncertainty
    step_rel_unc = np.sqrt(numerator_rel_unc**2 + denominator_rel_unc**2)

    # Calculate final uncertainty
    step_unc = step_ratio * step_rel_unc

    if step:
        # Add an extra bin in the beginning to have the same binning as the input
        # Otherwise, the ratio will not be exactly above each other (due to step)
        step_ratio = np.append(np.array([step_ratio[0]]), step_ratio)
        step_unc = np.append(np.array([step_rel_unc[0]]), step_rel_unc) * step_ratio

    return step_ratio, step_unc
