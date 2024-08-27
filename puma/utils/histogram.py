"""Helper function for histogram handling."""

from __future__ import annotations

import numpy as np

from puma.utils.logging import logger


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
    filled: bool = False,
    bins_range=None,
    normed: bool = True,
    weights: np.ndarray = None,
    bin_edges: np.ndarray = None,
    sum_squared_weights: np.ndarray = None,
    underoverflow: bool = False,
):
    """
    Computes histogram and the associated statistical uncertainty.

    Parameters
    ----------
    arr : array_like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars or str
        `bins` parameter from `np.histogram`
    bins_range : tuple, optional
        `range` parameter from `np.histogram`. This is ignored if `bins` is array like,
        because then the entries of `bins` are used as bin edges.
    normed : bool, optional
        If True (default) the calculated histogram is normalised to an integral
        of 1.
    weights : np.ndarray, optional
        Weights for the input data. Has to be an array of same length as the input
        data with a weight for each entry. If not specified, weight 1 will be given
        to each entry. The uncertainty of bins with weighted entries is
        sqrt(sum_i{w_i^2}) where w_i are the weights of the entries in this bin.
        By default None.
    underoverflow : bool, optional
        Option to include under- and overflow values in outermost bins.

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
    if weights is None:
        weights = np.ones(len(arr))

    # Check if there are nan values in the input values
    nan_mask = np.isnan(arr)
    if np.sum(nan_mask) > 0:
        logger.warning("Histogram values contain %i nan values!", np.sum(nan_mask))
        # Remove nan values
        arr = arr[~nan_mask]
        weights = weights[~nan_mask]

    # Check if there are inf values in the input values
    inf_mask = np.isinf(arr)
    if np.sum(inf_mask) > 0:
        logger.warning("Histogram values contain %i +-inf values!", np.sum(inf_mask))

    # If the histogram is not already filled we need to produce the histogram counts
    # and bin edges
    if not filled:
        # Calculate the counts and the bin edges
        counts, bin_edges = np.histogram(arr, bins=bins, range=bins_range, weights=weights)

        # calculate the uncertainty with sum of squared weights (per bin, so we use
        # np.histogram again here)
        unc = np.sqrt(np.histogram(arr, bins=bins, range=bins_range, weights=weights**2)[0])

        if underoverflow:
            # add two dummy bins (from outermost bins to +-infinity)
            bins_with_overunderflow = np.hstack([
                np.array([-np.inf]),
                bin_edges,
                np.array([np.inf]),
            ])
            # recalculate the histogram with this adjusted binning
            counts, _ = np.histogram(arr, bins=bins_with_overunderflow, weights=weights)
            counts[1] += counts[0]  # add underflow values to underflow bin
            counts[-2] += counts[-1]  # add overflow values to overflow bin
            counts = counts[1:-1]  # remove dummy bins

            # calculate the sum of squared weights
            sum_squared_weights = np.histogram(
                arr, bins=bins_with_overunderflow, weights=weights**2
            )[0]

            # add sum of squared weights from under/overflow values
            # to under/overflow bin
            sum_squared_weights[1] += sum_squared_weights[0]
            sum_squared_weights[-2] += sum_squared_weights[-1]
            # remove dummy bins
            sum_squared_weights = sum_squared_weights[1:-1]

            # uncertainty is sqrt(sum_squared_weights)
            unc = np.sqrt(sum_squared_weights)

        if normed:
            sum_of_weights = float(np.sum(weights))
            counts = save_divide(counts, sum_of_weights, 0)
            unc = save_divide(unc, sum_of_weights, 0)

    # If the histogram is already filled then the uncertainty is computed
    # differently
    else:
        if sum_squared_weights is not None:
            sum_squared_weights = np.array(sum_squared_weights)[~nan_mask]
            unc = np.sqrt(sum_squared_weights)
        else:
            unc = np.sqrt(arr)  # treat arr as bin heights (counts)

        counts = arr

        if normed:
            counts_sum = float(np.sum(counts))
            counts = save_divide(counts, counts_sum, 0)
            unc = save_divide(unc, counts_sum, 0)

    # regardless of if the histogram is filled
    band = counts - unc
    hist = counts

    return bin_edges, hist, unc, band


def hist_ratio(
    numerator,
    denominator,
    numerator_unc,
    step: bool = True,
    method: str = "divide",
):
    """
    Calculate the ratio of the given bincounts and
    returns the input for a step function that plots the ratio.

    Parameters
    ----------
    numerator : array_like
        Numerator in the ratio calculation.
    denominator : array_like
        Denominator in the ratio calculation.
    numerator_unc : array_like
        Uncertainty of the numerator.
    step : bool
        if True duplicates first bin to match with step plotting function,
        by default True
    method : str
        Selects the method by which the ratio should be calculated.
        "divide" calculates the ratio as the division of the numerator by the denominator.
        "root_square_diff" calculates the Root Square Difference
        between the numerator and the denominator.
        by default "divide"


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
    numerator, denominator, numerator_unc = (
        np.array(numerator),
        np.array(denominator),
        np.array(numerator_unc),
    )
    if numerator.shape != denominator.shape:
        raise AssertionError("Numerator and denominator don't have the same legth")
    if numerator.shape != numerator_unc.shape:
        raise AssertionError("Numerator and numerator_unc don't have the same legth")

    if method == "divide":
        step_ratio = save_divide(numerator, denominator, 1 if step else np.inf)
        # Calculate ratio uncertainty
        step_unc = save_divide(numerator_unc, denominator, default=0 if step else np.inf)
    elif method == "root_square_diff":
        step_ratio = np.multiply(
            np.sign(numerator - denominator), np.sqrt(np.abs(numerator**2 - denominator**2))
        )
        # Calculate ratio uncertainty
        step_unc = np.zeros_like(step_ratio)
        step_unc = np.divide(
            np.multiply(numerator, numerator_unc),
            np.sqrt(np.abs(numerator**2 - denominator**2)),
            where=(numerator - denominator != 0),
        )
    else:
        raise ValueError("'method' can only be 'divide' or 'root_square_diff'.")

    if step:
        # Add an extra bin in the beginning to have the same binning as the input
        # Otherwise, the ratio will not be exactly above each other (due to step)
        step_ratio = np.append(np.array([step_ratio[0]]), step_ratio)
        step_unc = np.append(np.array([step_unc[0]]), step_unc)

    return step_ratio, step_unc
