#!/usr/bin/env python

"""
Unit test script for the functions in plot_base.py
"""

import unittest

from puma import PlotObject
from puma.utils import logger, set_log_level

set_log_level(logger, "DEBUG")


class PlotObjectTestCase(unittest.TestCase):
    """Test class for the puma.PlotObject dataclass."""

    def test_only_one_input_figsize(self):
        """Test only one input figsize."""
        with self.assertRaises(ValueError):
            PlotObject(figsize=1)

    def test_only_tuple_three_inputs_figsize(self):
        """Test only tuple three inputs figsize."""
        with self.assertRaises(ValueError):
            PlotObject(figsize=(1, 2, 3))

    def test_tuple_input_figsize(self):
        """Test tuple input figsize."""
        figsize = PlotObject(figsize=(1, 2)).figsize
        self.assertEqual(figsize, (1, 2))

    def test_list_input_figsize(self):
        """Test list input figsize."""
        figsize = PlotObject(figsize=[1, 2]).figsize
        self.assertEqual(figsize, (1, 2))

    def test_list_input_wrong_len_figsize(self):
        """Test list input wrong len figsize."""
        with self.assertRaises(ValueError):
            PlotObject(figsize=[1, 2, 3])

    def test_wrong_n_ratio_panels(self):
        """Test wrong n ratio panels."""
        with self.assertRaises(ValueError):
            PlotObject(n_ratio_panels=5)
