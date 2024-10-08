"""Unit test script for the functions in line_plot_2d.py."""

from __future__ import annotations

import os
import shutil  # noqa: F401
import tempfile
import unittest

import numpy as np
from matplotlib.testing.compare import compare_images

from puma import Line2D, Line2DPlot
from puma.utils import logger, set_log_level

set_log_level(logger, "DEBUG")


class Line2DTestCase(unittest.TestCase):
    """Test class for the puma.line_plot_2d functions."""

    def test_wrong_inputs_xvalues(self):
        """Test if providing wrong input type to Line2D raises ValueError."""
        with self.assertRaises(TypeError):
            Line2D(
                x_values="Test",
                y_values=5,
            )

    def test_differnt_input_types(self):
        """Test if providing different input types raises ValueError."""
        with self.assertRaises(ValueError):
            Line2D(
                x_values=[1, 2, 3],
                y_values=np.array([1, 2, 3]),
            )

    def test_empty_input(self):
        """Test if ValueError is raised when one of the input values is zero."""
        with self.assertRaises(ValueError):
            Line2D(
                x_values=[],
                y_values=[1, 2],
            )

        with self.assertRaises(ValueError):
            Line2D(
                x_values=[1, 2],
                y_values=[],
            )

    def test_different_input_shapes(self):
        """Test if ValueError is raised when different lengths given."""
        with self.assertRaises(ValueError):
            Line2D(
                x_values=[1, 2, 3],
                y_values=[1, 2],
            )


class Line2DPlotTestCase(unittest.TestCase):
    """Test class for puma.Line2DPlot."""

    def setUp(self):
        """Set up values needed."""
        # Line values
        self.x_values = np.arange(0.001, 1, 0.001)
        self.y_values = np.arange(0.001, 1, 0.001)

        # Marker values
        self.marker_x = 0.5
        self.marker_y = 0.5

        # Set up directories for comparison plots
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.actual_plots_dir = f"{self.tmp_dir.name}"
        self.expected_plots_dir = os.path.join(os.path.dirname(__file__), "expected_plots")

    def test_basic(self):
        """Test the basic functions of Line2DPlot."""
        frac_plot = Line2DPlot(n_ratio_panels=0)

        # Add line
        frac_plot.add(
            Line2D(
                x_values=self.x_values,
                y_values=self.y_values,
                label="Tagger 1",
                colour="r",
                linestyle="-",
            )
        )

        # Add marker
        frac_plot.add(
            Line2D(
                x_values=self.marker_x,
                y_values=self.marker_y,
                colour="r",
                marker="x",
                label="Marker label",
                markersize=15,
                markeredgewidth=2,
            ),
            is_marker=True,
        )

        frac_plot.xlabel = "Test_x_label"
        frac_plot.ylabel = "Test_y_label"

        # Define a plot name
        name = "test_line2d_all_params_given.png"

        # Draw and save the plot
        frac_plot.draw()
        frac_plot.savefig(f"{self.actual_plots_dir}/{name}")

        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")

        # Check
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

    def test_grid_off(self):
        """Test the basic functions of Line2DPlot."""
        frac_plot = Line2DPlot(n_ratio_panels=0, grid=False)

        # Add line
        frac_plot.add(
            Line2D(
                x_values=self.x_values,
                y_values=self.y_values,
                label="Tagger 1",
                colour="r",
                linestyle="-",
            )
        )

        # Add marker
        frac_plot.add(
            Line2D(
                x_values=self.marker_x,
                y_values=self.marker_y,
                colour="r",
                marker="x",
                label="Marker label",
                markersize=15,
                markeredgewidth=2,
            ),
            is_marker=True,
        )

        frac_plot.xlabel = "Test_x_label"
        frac_plot.ylabel = "Test_y_label"

        # Define a plot name
        name = "test_line2d_grid_off.png"

        # Draw and save the plot
        frac_plot.draw()
        frac_plot.savefig(f"{self.actual_plots_dir}/{name}")

        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")

        # Check
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

    def test_no_param_set(self):
        """Test the basic functions of Line2DPlot."""
        frac_plot = Line2DPlot(n_ratio_panels=0)

        # Add line
        frac_plot.add(
            Line2D(
                x_values=self.x_values,
                y_values=self.y_values,
                label="Tagger 1",
                colour=None,
                linestyle=None,
            )
        )

        # Add marker
        frac_plot.add(
            Line2D(
                x_values=self.marker_x,
                y_values=self.marker_y,
                colour=None,
                marker=None,
                label=None,
                markersize=None,
                markeredgewidth=None,
            ),
            is_marker=True,
        )

        frac_plot.xlabel = "Test_x_label"
        frac_plot.ylabel = "Test_y_label"
        frac_plot.logx = True
        frac_plot.logy = True

        # Define a plot name
        name = "test_line2d_no_params_given.png"

        # Draw and save the plot
        frac_plot.draw()
        frac_plot.savefig(f"{self.actual_plots_dir}/{name}")

        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")

        # Check
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

    def test_double_key(self):
        """Test the basic functions of Line2DPlot."""
        frac_plot = Line2DPlot(n_ratio_panels=0)

        # Add line
        frac_plot.add(
            Line2D(
                x_values=self.x_values,
                y_values=self.y_values,
                label="Tagger 1",
                colour="r",
                linestyle="-",
            ),
            key=1,
        )

        with self.assertRaises(KeyError):
            # Add marker
            frac_plot.add(
                Line2D(
                    x_values=self.marker_x,
                    y_values=self.marker_y,
                    colour="r",
                    marker="x",
                    label="Marker label",
                    markersize=15,
                    markeredgewidth=2,
                ),
                is_marker=True,
                key=1,
            )
