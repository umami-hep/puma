"""Unit test script for the functions in utils/histogram.py."""

from __future__ import annotations

import unittest

import numpy as np
from testfixtures import LogCapture

from puma.utils import logger, set_log_level
from puma.utils.histogram import hist_ratio, hist_w_unc, save_divide

set_log_level(logger, "DEBUG")


class HistWUncTestCase(unittest.TestCase):
    """Test case for hist_w_unc function."""

    def setUp(self):
        self.input = np.array([0, 1, 1, 3])
        self.weights = np.array([1, 2, 1, 1])
        self.n_bins = 3
        self.bin_edges = np.array([0, 1, 2, 3])

        self.hist = np.array([1, 2, 1])
        self.unc = np.sqrt(np.array([1, 2, 1]))
        self.band = self.hist - self.unc

        self.hist_normed = np.array([1, 2, 1]) / len(self.input)
        self.unc_normed = np.sqrt(np.array([1, 2, 1])) / len(self.input)
        self.band_normed = self.hist_normed - self.unc_normed

        # --- weighted cases ---
        # 3 counts in second bin due to weights
        self.hist_weighted_normed = np.array([1, 3, 1]) / np.sum(self.weights)
        # use sqrt(sum of squared weights) for error calculation
        self.unc_weighted_normed = np.sqrt(np.array([1, 2**2 + 1, 1])) / np.sum(self.weights)
        self.band_weighted_normed = self.hist_weighted_normed - self.unc_weighted_normed

        self.hist_weighted = np.array([1, 3, 1])
        # use sqrt(sum of squared weights) for error calculation
        self.unc_weighted = np.sqrt(np.array([1, 2**2 + 1, 1]))
        self.band_weighted = self.hist_weighted - self.unc_weighted

    def test_under_overflow_values(self):
        """Test behaviour for under- and overflow values."""
        values_with_inf = np.array([-1, 1, 2, 100, np.inf])

        with self.subTest("Under/overflow values without under/overflow bins."):
            bins, hist, _, _ = hist_w_unc(values_with_inf, bins=5, bins_range=(0, 5))
            np.testing.assert_almost_equal(bins, np.linspace(0, 5, 6))
            # in this case only 40% of the values are shown in the plot
            np.testing.assert_almost_equal(hist, np.array([0, 0.2, 0.2, 0, 0]))

        with self.subTest("Under/overflow values with under/overflow bins."):
            bins, hist, _, _ = hist_w_unc(
                values_with_inf, bins=5, bins_range=(0, 5), underoverflow=True
            )
            np.testing.assert_almost_equal(bins, np.linspace(0, 5, 6))
            np.testing.assert_almost_equal(hist, np.array([0.2, 0.2, 0.2, 0, 0.4]))

    def test_hist_w_unc_zero_case(self):
        """Test what happens if empty array is provided as input."""
        bins, hist, unc, band = hist_w_unc(
            arr=[],
            bins=[],
        )

        np.testing.assert_almost_equal(bins, [])
        np.testing.assert_almost_equal(hist, [])
        np.testing.assert_almost_equal(unc, [])
        np.testing.assert_almost_equal(band, [])

    def test_hist_w_unc_normed(self):
        """Test normalised case."""
        bins, hist, unc, band = hist_w_unc(
            arr=self.input,
            bins=self.bin_edges,
        )

        np.testing.assert_almost_equal(bins, self.bin_edges)
        np.testing.assert_almost_equal(hist, self.hist_normed)
        np.testing.assert_almost_equal(unc, self.unc_normed)
        np.testing.assert_almost_equal(band, self.band_normed)

    def test_hist_w_unc_not_normed(self):
        """Test not normalised case."""
        bins, hist, unc, band = hist_w_unc(
            arr=self.input,
            bins=self.bin_edges,
            normed=False,
        )

        np.testing.assert_almost_equal(bins, self.bin_edges)
        np.testing.assert_almost_equal(hist, self.hist)
        np.testing.assert_almost_equal(unc, self.unc)
        np.testing.assert_almost_equal(band, self.band)

    def test_histogram_weighted_normalised(self):
        """Test weighted histogram (normalised)."""
        bin_edges, hist, unc, band = hist_w_unc(
            self.input, weights=self.weights, bins=self.n_bins, normed=True
        )

        np.testing.assert_array_almost_equal(self.bin_edges, bin_edges)
        np.testing.assert_array_almost_equal(self.hist_weighted_normed, hist)
        np.testing.assert_array_almost_equal(self.unc_weighted_normed, unc)
        np.testing.assert_array_almost_equal(self.band_weighted_normed, band)

    def test_histogram_weighted_not_normalised(self):
        """Test weighted histogram (not normalised)."""
        bin_edges, hist, unc, band = hist_w_unc(
            self.input, weights=self.weights, bins=self.n_bins, normed=False
        )

        np.testing.assert_array_almost_equal(self.bin_edges, bin_edges)
        np.testing.assert_array_almost_equal(self.hist_weighted, hist)
        np.testing.assert_array_almost_equal(self.unc_weighted, unc)
        np.testing.assert_array_almost_equal(self.band_weighted, band)

    def test_range_argument_ignored(self):
        """Test if the hist_range argument is ignored when bin_edges are provided."""
        bins_range = (1, 2)

        bin_edges, hist, _, _ = hist_w_unc(
            self.input,
            bins=self.bin_edges,
            bins_range=bins_range,
            normed=False,
        )

        # check if we end up with the same bin edges anyway
        np.testing.assert_array_almost_equal(self.bin_edges, bin_edges)
        np.testing.assert_array_almost_equal(self.hist, hist)

    def test_range_argument(self):
        """Test if the hist_range argument is used when bins is an integer."""
        # we test with range from 0 to 2, with 3 bins -> [0, 0.66, 1.33, 2] exp. bins
        bins_range = (0, 2)
        bins_exp = np.array([0, 2 / 3, 1 + 1 / 3, 2])
        hist_exp = np.array([1, 2, 0])

        bin_edges, hist, _, _ = hist_w_unc(
            self.input,
            bins=self.n_bins,
            bins_range=bins_range,
            normed=False,
        )

        # check if we end up with the same bin edges anyway
        np.testing.assert_array_almost_equal(bins_exp, bin_edges)
        np.testing.assert_array_almost_equal(hist_exp, hist)

    def test_negative_weights(self):
        """Test if negative weights are properly handled."""
        values = np.array([0, 1, 2, 2, 3])
        weights = np.array([1, -1, 3, -2, 1])

        hist_exp = np.array([1, -1, 2])
        # uncertainties are the sqrt(sum of squared weights)
        unc_exp = np.sqrt(np.array([1, (-1) ** 2, 3**2 + (-2) ** 2 + 1]))

        _, hist, unc, _ = hist_w_unc(values, weights=weights, bins=3, normed=False)
        np.testing.assert_array_almost_equal(hist_exp, hist)
        np.testing.assert_array_almost_equal(unc_exp, unc)

    def test_inf_treatment(self):
        """Test if infinity values are treated as expected."""
        values_with_infs = np.array([1, 2, 3, -np.inf, +np.inf, +np.inf])

        with self.subTest("Test if the warning for number of inf values is raised in hist_w_unc"):  # noqa: SIM117
            with LogCapture("puma") as log:
                _ = hist_w_unc(values_with_infs, bins=np.linspace(0, 3, 3))
                log.check((
                    "puma",
                    "WARNING",
                    "Histogram values contain 3 +-inf values!",
                ))
        with self.subTest(
            "Test if error is raised if inf values are in input but no range is defined"
        ), self.assertRaises(ValueError):
            hist_w_unc(values_with_infs, bins=10)

    def test_nan_check(self):
        """Test if the warning with number of nan values is raised in hist_w_unc."""
        values_with_nans = np.array([1, 2, 3, np.nan, np.nan])

        with LogCapture("puma") as log:
            _ = hist_w_unc(values_with_nans, bins=4)
            log.check((
                "puma",
                "WARNING",
                "Histogram values contain 2 nan values!",
            ))


class SaveDivideTestCase(unittest.TestCase):
    """Test case for save_divide function."""

    def test_zero_case(self):
        """Test zero case."""
        steps = save_divide(np.zeros(2), np.zeros(2))
        np.testing.assert_equal(steps, np.ones(2))

    def test_ones_case(self):
        """Test where all inputs are 1s."""
        steps = save_divide(np.ones(2), np.ones(2))
        np.testing.assert_equal(steps, np.ones(2))

    def test_half_case(self):
        """Test scenario where result is 1/2."""
        steps = save_divide(np.ones(2), 2 * np.ones(2))
        np.testing.assert_equal(steps, 0.5 * np.ones(2))

    def test_denominator_float(self):
        """Test scenario where denominator is a float."""
        steps = save_divide(np.ones(2), 2)
        np.testing.assert_equal(steps, 0.5 * np.ones(2))

    def test_numerator_float(self):
        """Test scenario where numerator is a float."""
        steps = save_divide(1, np.ones(2) * 2)
        np.testing.assert_equal(steps, 0.5 * np.ones(2))


class HistRatioTestCase(unittest.TestCase):
    """Test case for hist_ratio function."""

    def setUp(self):
        self.numerator = np.array([5, 3, 2, 5, 6, 2])
        self.denominator = np.array([3, 6, 2, 7, 10, 12])
        self.numerator_unc = np.array([0.5, 1, 0.3, 0.2, 0.5, 0.3])
        self.denominator_unc = np.array([1, 0.3, 2, 1, 5, 3])
        self.step = np.array([1.6666667, 1.6666667, 0.5, 1, 0.7142857, 0.6, 0.1666667])
        self.step_unc = np.array([
            0.16666667,
            0.16666667,
            0.16666667,
            0.15,
            0.02857143,
            0.05,
            0.025,
        ])
        self.step_rsd = np.array([4, 4, -5.1961524, 0, -4.8989795, -8, -11.8321596])
        self.step_rsd_unc = np.array([
            0.625,
            0.625,
            0.5773503,
            0,
            0.2041241,
            0.375,
            0.0507093,
        ])

    def test_hist_ratio(self):
        """Test if ratio is correctly calculated."""
        step, step_unc = hist_ratio(
            numerator=self.numerator,
            denominator=self.denominator,
            numerator_unc=self.numerator_unc,
        )

        np.testing.assert_almost_equal(step, self.step)
        np.testing.assert_almost_equal(step_unc, self.step_unc)

    def test_hist_rsd(self):
        step_rsd, step_rsd_unc = hist_ratio(
            numerator=self.numerator,
            denominator=self.denominator,
            numerator_unc=self.numerator_unc,
            method="root_square_diff",
        )

        np.testing.assert_almost_equal(step_rsd, self.step_rsd)
        np.testing.assert_almost_equal(step_rsd_unc, self.step_rsd_unc)

    def test_hist_not_same_length_numerator_denominator(self):
        """Test case where denominator and numerator have not the same length."""
        with self.assertRaises(AssertionError):
            _, _ = hist_ratio(
                numerator=np.ones(2),
                denominator=np.ones(3),
                numerator_unc=np.ones(3),
            )

    def test_hist_not_same_length_numerator_and_unc(self):
        """Test case where numerator and uncertainty have not the same length."""
        with self.assertRaises(AssertionError):
            _, _ = hist_ratio(
                numerator=np.ones(3),
                denominator=np.ones(3),
                numerator_unc=np.ones(2),
            )
