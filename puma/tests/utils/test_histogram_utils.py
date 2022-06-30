#!/usr/bin/env python

"""
Unit test script for the functions in utils/histogram.py
"""

# import os
# import tempfile
import unittest

import numpy as np

# from puma import Histogram, HistogramPlot
from puma.utils import logger, set_log_level
from puma.utils.histogram import hist_w_unc

# from puma.utils.histogram import hist_ratio, hist_w_unc

set_log_level(logger, "DEBUG")


class histogram_utils_TestCase(unittest.TestCase):
    """Test class for the puma.utils.histogram functions."""

    def test_histogram_unweighted_not_normalised(self):
        """Test if unweighted histogram returns the expected counts (not normalised)."""

        values = np.array([0, 1, 1, 3])
        n_bins = 3

        exp_bin_edges = np.array([0, 1, 2, 3])
        exp_hist = np.array([1, 2, 1])
        exp_unc = np.sqrt(np.array([1, 2, 1]))
        exp_band = exp_hist - exp_unc

        bin_edges, hist, unc, band = hist_w_unc(values, bins=n_bins, normed=False)

        np.testing.assert_array_almost_equal(exp_bin_edges, bin_edges)
        np.testing.assert_array_almost_equal(exp_hist, hist)
        np.testing.assert_array_almost_equal(exp_unc, unc)
        np.testing.assert_array_almost_equal(exp_band, band)

    def test_histogram_unweighted_normalised(self):
        """Test if unweighted histogram returns the expected counts (normalised)."""

        values = np.array([0, 1, 1, 3])
        n_bins = 3

        exp_bin_edges = np.array([0, 1, 2, 3])
        exp_hist = np.array([1, 2, 1]) / len(values)
        exp_unc = np.sqrt(np.array([1, 2, 1])) / len(values)
        exp_band = exp_hist - exp_unc

        bin_edges, hist, unc, band = hist_w_unc(values, bins=n_bins, normed=True)

        np.testing.assert_array_almost_equal(exp_bin_edges, bin_edges)
        np.testing.assert_array_almost_equal(exp_hist, hist)
        np.testing.assert_array_almost_equal(exp_unc, unc)
        np.testing.assert_array_almost_equal(exp_band, band)

    def test_histogram_weighted_normalised(self):
        """Test if weighted histogram returns the expected counts (normalised)."""

        values = np.array([0, 1, 1, 3])
        weights = np.array([1, 2, 1, 1])
        n_bins = 3

        exp_bin_edges = np.array([0, 1, 2, 3])
        # 3 counts in second bin due to weights
        exp_hist = np.array([1, 3, 1]) / len(values)
        exp_unc = np.sqrt(np.array([1, 2**2 + 1, 1])) / len(values)
        exp_band = exp_hist - exp_unc

        bin_edges, hist, unc, band = hist_w_unc(
            values, weights=weights, bins=n_bins, normed=True
        )

        np.testing.assert_array_almost_equal(exp_bin_edges, bin_edges)
        np.testing.assert_array_almost_equal(exp_hist, hist)
        np.testing.assert_array_almost_equal(exp_unc, unc)
        np.testing.assert_array_almost_equal(exp_band, band)

    # TODO: Add unit tests for:
    # 2. hist_w_unc for weighted calculation
    # test range/bins cases
    # 3. hist_ratio
    # 4. test negative weights
    # 4. Check what happens in hist_w_unc if `bins` (array with bin_edges) AND
    #    `bins_range` is specified (in this case bins_range should be ignored, since
    #    we just use the np.histogram function and hand the parameters to that function)
