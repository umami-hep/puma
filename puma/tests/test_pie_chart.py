#!/usr/bin/env python

"""
Unit test script for the functions in histogram.py
"""

import os
import tempfile
import unittest

from matplotlib.testing.compare import compare_images

from puma import PiePlot
from puma.utils import logger, set_log_level

set_log_level(logger, "DEBUG")


class pie_plot_TestCase(unittest.TestCase):
    """Test class for puma.PiePlot"""

    def setUp(self):

        # Set up directories for comparison plots
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        self.expected_plots_dir = os.path.join(
            os.path.dirname(__file__), "expected_plots"
        )

    def test_plot_pie_chart_default_style(self):
        """check if pie chart is plotted correctly (using default style)"""
        pie_plot = PiePlot(
            wedge_sizes=[20, 40, 30, 10],
            labels=["light-flavour jets", "c-jets", "b-jets", "tau-jets"],
        )
        plotname = "test_pie_chart_default_style.png"
        pie_plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # pie_plot.savefig(f"{self.expected_plots_dir}/{plotname}")

        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )
