"""Unit test script for the functions in metrics.py."""

from __future__ import annotations

import unittest

import numpy as np

from puma.metrics import calc_eff, calc_rej, calc_separation, eff_err, rej_err
from puma.utils import logger, set_log_level

set_log_level(logger, "DEBUG")


class SeparationTestCase(unittest.TestCase):
    """Test class for the puma.metrics.calc_separation."""

    def setUp(self):
        """Define a default (seeded) random number generator for all tests."""
        self.rng = np.random.default_rng(42)

    def test_equal_datasets(self):
        """Separation of two equal datasets (=0)."""
        values_a = np.array([1, 1, 2, 2])
        self.assertEqual(0, calc_separation(values_a, values_a)[0])

    def test_completely_separated(self):
        """Separation of two completely separated distributions."""
        values_a = np.array([0.1, 0.5, 0.8, 1])
        values_b = np.array([1.1, 1.5, 1.8, 2])
        self.assertAlmostEqual(1, calc_separation(values_a, values_b)[0])

    def test_completely_separated_bad_binning(self):
        """Separation of two completely separated distributions if the binning
        if chosen such that they share one bin
        .
        """
        values_a = np.array([0.1, 0.5, 0.8, 1])
        values_b = np.array([1.1, 1.5, 1.8, 2])
        self.assertNotAlmostEqual(1, calc_separation(values_a, values_b, bins=3)[0])

    def test_half_separated(self):
        """Separation of 0.5."""
        values_a = np.array([0, 1])
        values_b = np.array([1, 2])
        self.assertAlmostEqual(0.5, calc_separation(values_a, values_b, bins=3)[0])

    def test_return_bins(self):
        """Test if bins are correctly returned."""
        values_a = np.array([0, 1])
        values_b = np.array([1, 2])

        _, _, hist_a, hist_b, bin_edges = calc_separation(
            values_a,
            values_b,
            bins=3,
            return_hist=True,
        )
        # Check for correct values in hist_a, hist_b and bin_edges
        np.testing.assert_array_equal(np.array([0.5, 0.5, 0]), hist_a)
        np.testing.assert_array_equal(np.array([0, 0.5, 0.5]), hist_b)
        np.testing.assert_array_equal(np.array([0, 2 / 3, 4 / 3, 2]), bin_edges)

    def test_bins_range(self):
        """Test if bins_range is properly treated."""
        values_a = np.array([0, 1])
        values_b = np.array([1, 2])

        _, _, hist_a, _, bin_edges = calc_separation(
            values_a,
            values_b,
            bins=4,
            bins_range=(0, 4),  # this should result in bin edges [0, 1, 2, 3, 4]
            return_hist=True,
        )
        # Check for correct values in hist_a and bin_edges
        np.testing.assert_array_equal(np.array([0.5, 0.5, 0, 0]), hist_a)
        np.testing.assert_array_equal(np.array([0, 1, 2, 3, 4]), bin_edges)


class CalcEffAndRejTestCase(unittest.TestCase):
    """Test class for the puma.metrics functions."""

    def setUp(self):
        rng = np.random.default_rng(seed=42)
        self.disc_sig = rng.normal(loc=3, size=100_000)
        self.disc_bkg = rng.normal(loc=0, size=100_000)

    def test_float_target(self):
        """Test efficiency and cut value calculation for one target value."""
        # we target a signal efficiency of 0.841345, which is the integral of a gaussian
        # from μ-1o to infinity
        # -->   the cut_value should be at 2, since the signal is a normal distribution
        #       with mean 3
        # https://www.wolframalpha.com/input?i=integrate+1%2Fsqrt%282+pi%29+*+exp%28-0.5*%28x-3%29**2%29+from+2+to+oo
        # -->   For the bkg efficiency this means that we integrate a normal distr.
        #       from μ+2o to infinity --> expect a value of 0.0227501
        # https://www.wolframalpha.com/input?i=integrate+1%2Fsqrt%282+pi%29+*+exp%28-0.5*x**2%29+from+2+to+oo
        bkg_eff, cut = calc_eff(self.disc_sig, self.disc_bkg, target_eff=0.841345, return_cuts=True)
        # the values here differ slightly from the values of the analytical integral,
        # since we use random numbers
        self.assertAlmostEqual(cut, 1.9956997)
        self.assertAlmostEqual(bkg_eff, 0.02367)

        # same for rejection, just use rej = 1 / eff
        bkg_rej, cut = calc_rej(self.disc_sig, self.disc_bkg, target_eff=0.841345, return_cuts=True)
        self.assertAlmostEqual(cut, 1.9956997)
        self.assertAlmostEqual(bkg_rej, 1 / 0.02367)

    def test_array_target(self):
        """Test efficiency and cut value calculation for list of target efficiencies."""
        # explanation is the same as above, now also cut the signal in the middle
        # --> target sig.efficiency 0.841345 and 0.5 --> cut at 2 and 3
        bkg_eff, cut = calc_eff(
            self.disc_sig, self.disc_bkg, target_eff=[0.841345, 0.5], return_cuts=True
        )
        # the values here differ slightly from the values of the analytical integral,
        # since we use random numbers
        np.testing.assert_array_almost_equal(cut, np.array([1.9956997, 2.990996]))
        np.testing.assert_array_almost_equal(bkg_eff, np.array([0.02367, 0.00144]))

        # same for rejection, just use rej = 1 / eff
        bkg_rej, cut = calc_rej(
            self.disc_sig, self.disc_bkg, target_eff=[0.841345, 0.5], return_cuts=True
        )
        np.testing.assert_array_almost_equal(cut, np.array([1.9956997, 2.990996]))
        np.testing.assert_array_almost_equal(bkg_rej, 1 / np.array([0.02367, 0.00144]))


class EffErrTestCase(unittest.TestCase):
    """Test class for the puma.metrics functions."""

    def test_zero_n_case(self):
        """Test eff_err function."""
        with self.assertRaises(ValueError):
            eff_err(0, 0)

    def test_negative_n_case(self):
        """Test eff_err function."""
        with self.assertRaises(ValueError):
            eff_err(0, -1)

    def test_one_case(self):
        """Test eff_err function."""
        self.assertEqual(eff_err(1, 1), 0)

    def test_example_case(self):
        """Test eff_err function."""
        x_eff = np.array([0.25, 0.5, 0.75])
        error_eff = np.array([0.043301, 0.05, 0.043301])
        np.testing.assert_array_almost_equal(eff_err(x_eff, 100), error_eff)


class RejErrTestCase(unittest.TestCase):
    """Test class for the puma.metrics functions."""

    def setUp(self):
        self.array = np.array([[1, 2, 3], [4, 5, 6]])

    def test_zero_n_case(self):
        """Test rej_err function."""
        with self.assertRaises(ValueError):
            rej_err(0, 0)

    def test_negative_n_case(self):
        """Test rej_err function."""
        with self.assertRaises(ValueError):
            rej_err(0, -1)

    def test_one_case(self):
        """Test rej_err function."""
        self.assertEqual(rej_err(1, 1), 0)

    def test_zero_x_case(self):
        """Test rej_err function."""
        with self.assertRaises(ValueError):
            rej_err(np.array([0, 1, 2]), 3)

    def test_example_case(self):
        """Test rej_err function."""
        x_rej = np.array([20, 50, 100])
        error_rej = np.array([8.717798, 35.0, 99.498744])
        np.testing.assert_array_almost_equal(rej_err(x_rej, 100), error_rej)
