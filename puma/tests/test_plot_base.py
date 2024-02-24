"""Unit test script for the functions in plot_base.py."""

from __future__ import annotations

import unittest

from puma import PlotBase, PlotObject
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

    def test_wrong_ymin_ratio(self):
        """Test wrong ymin_ratio."""
        with self.assertRaises(ValueError):
            PlotObject(n_ratio_panels=2, ymin_ratio=[0])

    def test_wrong_ymax_ratio(self):
        """Test wrong ymax_ratio."""
        with self.assertRaises(ValueError):
            PlotObject(n_ratio_panels=2, ymax_ratio=[0])

    def test_set_ratio_label_invalid_ratio_panel(self):
        plot_object = PlotBase(n_ratio_panels=2)
        plot_object.set_ratio_label(ratio_panel=1, label="Label")
        plot_object.set_ratio_label(ratio_panel=2, label="Label")
        with self.assertRaises(ValueError):
            plot_object.set_ratio_label(ratio_panel=3, label="Label")


class PlotBaseTestCase(unittest.TestCase):
    """Test class for the puma.PlotBase class."""

    def test_ymin_ymax_ratio(self):
        """Test correct ymax/ymin_ratio."""
        plot_object = PlotBase(n_ratio_panels=2, ymin_ratio=[0, 1], ymax_ratio=[1, 2])
        plot_object.initialise_figure()
        plot_object.set_y_lim()
        for i in range(2):
            ymin, ymax = plot_object.ratio_axes[i].get_ylim()
            self.assertEqual((ymin, ymax), (i, i + 1))
