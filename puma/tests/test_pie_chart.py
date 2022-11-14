#!/usr/bin/env python

"""
Unit test script for the functions in histogram.py
"""

import os
import tempfile
import unittest

from matplotlib.testing.compare import compare_images

from puma import PiePlot
from puma.utils import get_good_colours, logger, set_log_level

set_log_level(logger, "DEBUG")


class PiePlotTestCase(unittest.TestCase):
    """Test class for puma.PiePlot"""

    def setUp(self):

        # Set up directories for comparison plots
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
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

    def test_plot_pie_chart_custom_style(self):
        """check if pie chart is plotted correctly (using default style)"""
        pie_plot = PiePlot(
            wedge_sizes=[20, 40, 30, 10],
            labels=["light-flavour jets", "c-jets", "b-jets", "tau-jets"],
            draw_legend=True,
            colours=get_good_colours()[:4],
            # have a look at the possible kwargs for matplotlib.pyplot.pie here:
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html
            mpl_pie_kwargs={
                "explode": [0, 0, 0, 0.1],
                "shadow": False,
                "startangle": 90,
                "textprops": {"fontsize": 10},
                "radius": 1,
                "wedgeprops": dict(width=0.4, edgecolor="w"),
                "pctdistance": 0.4,
            },
            # kwargs passed to puma.PlotObject
            atlas_second_tag=(
                "Unit test plot to test if the custom\nstyling of the pie plot"
            ),
            figsize=(5.5, 3.5),
            y_scale=1.3,
        )
        plotname = "test_pie_chart_custom_style.png"
        pie_plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        pie_plot.savefig(f"{self.expected_plots_dir}/{plotname}")

        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )
