#!/usr/bin/env python

"""
Unit test script for the functions in metrics.py
"""

import unittest

import numpy as np

from puma.metrics import calc_separation
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
