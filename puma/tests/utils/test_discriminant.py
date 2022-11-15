#!/usr/bin/env python
"""
Unit test script for the functions in utils/discriminant.py
"""

import unittest

import numpy as np

from puma.utils import calc_disc_b, calc_disc_c, logger, set_log_level

set_log_level(logger, "DEBUG")


class BTaggingDiscTestCase(unittest.TestCase):
    """Tes case for calc_disc_b function."""

    def test_wrong_length(self):
        """Check all input arrays have same size."""
        with self.assertRaises(ValueError):
            calc_disc_b(np.ones(10), np.ones(10), np.ones(5), 0.3)

    def test_empty_input(self):
        """Check all input arrays have same size."""
        discs = calc_disc_b(np.ones(0), np.ones(0), np.ones(0), 0.3)
        np.testing.assert_almost_equal(discs, np.array([]))

    def test_ones(self):
        """Check all input arrays have same size."""
        discs = calc_disc_b(np.ones(10), np.ones(10), np.ones(10), 0)
        np.testing.assert_almost_equal(discs, np.zeros(10))


class CTaggingDiscTestCase(unittest.TestCase):
    """Tes case for calc_disc_c function."""

    def test_wrong_length(self):
        """Check all input arrays have same size."""
        with self.assertRaises(ValueError):
            calc_disc_c(np.ones(10), np.ones(10), np.ones(5), 0.3)

    def test_empty_input(self):
        """Check all input arrays have same size."""
        discs = calc_disc_c(np.ones(0), np.ones(0), np.ones(0), 0.3)
        np.testing.assert_almost_equal(discs, np.array([]))

    def test_ones(self):
        """Check all input arrays have same size."""
        discs = calc_disc_c(np.ones(10), np.ones(10), np.ones(10), 0)
        np.testing.assert_almost_equal(discs, np.zeros(10))
