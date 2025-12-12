"""Helper function for histogram handling."""

from __future__ import annotations

import numpy as np

from puma.utils.logger import logger


def save_divide(
    numerator: np.ndarray | float,
    denominator: np.ndarray | float,
    default: float = 1.0,
) -> np.ndarray:
    """
    Division using numpy divide function returning default value in cases where
    denominator is 0.

    Parameters
    ----------
    numerator: np.ndarray | float
        Numerator in the ratio calculation.
    denominator: np.ndarray | float
        Denominator in the ratio calculation.
    default: float, optional
        Default value which is returned if denominator is 0. By default 1.

    Returns
    -------
    ratio : np.ndarray
        Ratio of the numerator and denominator.
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
    arr: np.ndarray,
    bins: np.ndarray | int,
    filled: bool = False,
    bins_range: tuple | None = None,
    normed: bool = True,
    weights: np.ndarray = None,
    bin_edges: np.ndarray = None,
    sum_squared_weights: np.ndarray = None,
    underoverflow: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a 1D histogram together with per-bin statistical uncertainties.

    The function supports both raw input data (``filled=False``) and
    pre-computed bin contents (``filled=True``). Uncertainties follow the
    standard definition:
    ``sqrt(sum_i w_i^2)`` for weighted data, or ``sqrt(N)`` for unweighted
    counts. Optional inclusion of underflow and overflow contributions is
    supported for unfilled histograms.

    Parameters
    ----------
    arr : np.ndarray
        Input values. For ``filled=False`` these are the data points from
        which the histogram is constructed. For ``filled=True`` this array
        is interpreted as an array of bin counts.
    bins : np.ndarray | int
        Bin specification as accepted by ``np.histogram``. If array-like,
        the values are interpreted as explicit bin edges; otherwise the
        number of bins.
    filled : bool, optional
        If False (default), ``arr`` is treated as event-level input and a
        histogram is computed. If True, ``arr`` is assumed to contain
        pre-computed bin contents.
    bins_range : tuple | None, optional
        Range passed to ``np.histogram`` when ``bins`` is an integer.
        Ignored when ``bins`` is array-like.
    normed : bool, optional
        If True (default), the histogram and uncertainties are normalized
        such that the total integral equals 1.
    weights : np.ndarray, optional
        Per-event weights used when ``filled=False``. Must match the shape
        of ``arr``. If None, all entries receive weight 1.
        Uncertainties are computed as ``sqrt(sum_i w_i^2)`` per bin.
    bin_edges : np.ndarray, optional
        Explicit bin edges to return when ``filled=True``. Ignored for
        ``filled=False``. If None and ``filled=True``, the caller is
        expected to supply this value.
    sum_squared_weights : np.ndarray, optional
        Pre-computed sum of squared weights per bin when ``filled=True``.
        If provided, uncertainties are computed as ``sqrt(sum_squared_weights)``.
        If None, uncertainties fall back to ``sqrt(arr)``.
    underoverflow : bool, optional
        If True, underflow and overflow contributions are added to the
        outermost bins when ``filled=False``. Default is False.

    Returns
    -------
    bin_edges : np.ndarray
        Array of bin edges of length ``len(hist) + 1``.
    hist : np.ndarray
        Histogram bin contents. If ``normed=True``, returns normalized bin
        contents.
    unc : np.ndarray
        Statistical uncertainty per bin. If ``normed=True``, uncertainties
        are normalized consistently with ``hist``.
    band : np.ndarray
        Lower edge of the uncertainty band, ``hist - unc``. If
        ``normed=True``, the normalized values are returned.

    Notes
    -----
    * NaN values in the input are removed and reported via logging.
    * Infinite values are reported but retained.
    * For weighted histograms, uncertainties always follow the standard
      definition using squared weights.
    * When ``filled=True``, no histogramming of input values is performed;
      ``arr`` is interpreted directly as bin contents.
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
    numerator: np.ndarray,
    denominator: np.ndarray,
    numerator_unc: np.ndarray,
    step: bool = True,
    method: str = "divide",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the ratio of the given bincounts and
    returns the input for a step function that plots the ratio.

    Parameters
    ----------
    numerator : np.ndarray
        Numerator in the ratio calculation.
    denominator : np.ndarray
        Denominator in the ratio calculation.
    numerator_unc : np.ndarray
        Uncertainty of the numerator.
    step : bool, optional
        if True duplicates first bin to match with step plotting function,
        by default True
    method : str, optional
        Selects the method by which the ratio should be calculated.
        "divide" calculates the ratio as the division of the numerator by the denominator.
        "root_square_diff" calculates the Root Square Difference between the numerator and the
        denominator.
        "subtract" calculates the difference between the numerator and the denominator
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
    ValueError
        If the method is netiher "divide" nor "root_square_diff"
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
    elif method == "subtract":
        step_ratio = np.subtract(numerator, denominator)
        step_unc = numerator_unc
    else:
        raise ValueError("'method' can only be 'divide' or 'root_square_diff'.")

    if step:
        # Add an extra bin in the beginning to have the same binning as the input
        # Otherwise, the ratio will not be exactly above each other (due to step)
        step_ratio = np.append(np.array([step_ratio[0]]), step_ratio)
        step_unc = np.append(np.array([step_unc[0]]), step_unc)

    return step_ratio, step_unc
