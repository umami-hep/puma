#!/usr/bin/env python

"""
Unit test script for the functions in metrics.py
"""

import unittest

import numpy as np

from puma.metrics import calc_separation, eff_err, rej_err
from puma.utils import logger, set_log_level

set_log_level(logger, "DEBUG")


class separation_TestCase(unittest.TestCase):
    """Test class for the puma.metrics.calc_separation."""

    def setUp(self):
        """Define a default (seeded) random number generator for all tests"""
        self.rng = np.random.default_rng(42)

    def test_equal_datasets(self):
        """Separation of two equal datasets (=0)"""
        values_a = np.array([1, 1, 2, 2])
        self.assertEqual(0, calc_separation(values_a, values_a)[0])

    def test_completely_separated(self):
        """Separation of two completely separated distributions"""
        values_a = np.array([0.1, 0.5, 0.8, 1])
        values_b = np.array([1.1, 1.5, 1.8, 2])
        self.assertAlmostEqual(1, calc_separation(values_a, values_b)[0])

    def test_completely_separated_bad_binning(self):
        """Separation of two completely separated distributions if the binning
        if chosen such that they share one bin"""
        values_a = np.array([0.1, 0.5, 0.8, 1])
        values_b = np.array([1.1, 1.5, 1.8, 2])
        self.assertNotAlmostEqual(1, calc_separation(values_a, values_b, bins=3)[0])

    def test_half_separated(self):
        """Separation of 0.5"""
        values_a = np.array([0, 1])
        values_b = np.array([1, 2])
        self.assertAlmostEqual(0.5, calc_separation(values_a, values_b, bins=3)[0])


class eff_err_TestCase(unittest.TestCase):
    """Test class for the puma.metrics functions."""

    def test_zero_N_case(self):
        """Test eff_err function."""
        with self.assertRaises(ValueError):
            eff_err(0, 0)

    def test_negative_N_case(self):
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


class rej_err_TestCase(unittest.TestCase):
    """Test class for the puma.metrics functions."""

    def setUp(self):
        self.array = np.array([[1, 2, 3], [4, 5, 6]])

    def test_zero_N_case(self):
        """Test rej_err function."""
        with self.assertRaises(ValueError):
            rej_err(0, 0)

    def test_negative_N_case(self):
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
