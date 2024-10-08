"""Unit test script for the functions in utils/__init__.py."""

from __future__ import annotations

import os
import shutil  # noqa: F401
import tempfile
import unittest

import numpy as np
from matplotlib.testing.compare import compare_images

from puma import Line2D, Line2DPlot
from puma.utils import get_good_linestyles, logger, set_log_level

set_log_level(logger, "DEBUG")


class LinestylesTestCase(unittest.TestCase):
    """Test class for the linestyle management of puma."""

    def setUp(self):
        # Set up directories for comparison plots
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        self.expected_plots_dir = os.path.join(os.path.dirname(__file__), "expected_plots")

    def test_raise_value_error(self):
        """Test if ValueError is raised for wrong argument type."""
        with self.assertRaises(ValueError):
            get_good_linestyles(4)

    def test_get_good_linestyles(self):
        """Test if the default linestyles obtained are the correct ones."""
        with self.subTest("Testing default linestyles"):
            expected_linestyles = [
                "solid",
                (0, (3, 1, 1, 1)),
                (0, (1, 1)),
                (0, (5, 2)),
                (0, (3, 1, 1, 1, 1, 1)),
                (0, (5, 5)),
                (0, (2, 2)),
                "dashdot",
                (0, (5, 10)),
                (0, (1, 10)),
                (0, (3, 10, 1, 10)),
                (0, (3, 10, 1, 10, 1, 10)),
                (0, (3, 5, 1, 5)),
                (0, (3, 5, 1, 5, 1, 5)),
            ]
            actual_linestyles = get_good_linestyles()
            self.assertListEqual(expected_linestyles, actual_linestyles)

        with self.subTest("Test specifying list of names"):
            linestyle_names = ["densely dashed", "dashdotted"]
            expected_linestyles = [(0, (5, 2)), (0, (3, 5, 1, 5))]
            actual_linestyles = get_good_linestyles(linestyle_names)
            self.assertListEqual(expected_linestyles, actual_linestyles)

        with self.subTest("Test case of only one name given (not list)"):
            expected_linestyle = (0, (5, 2))
            actual_linestyle = get_good_linestyles("densely dashed")
            self.assertTupleEqual(expected_linestyle, actual_linestyle)

    def test_linestyles_accepted_by_mpl(self):
        """Test if all the linestyles from get_good_linestyles() are accepted by
        matplotlib.
        """
        test_plot = Line2DPlot()
        for i, linestyle in enumerate(get_good_linestyles()):
            test_plot.add(
                Line2D(
                    np.linspace(0, 10, 10),
                    i * np.linspace(0, 10, 10),
                    linestyle=linestyle,
                )
            )
        test_plot.draw()
        name = "test_linestyles.png"
        test_plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )
