from __future__ import annotations

import os
import shutil  # noqa: F401
import tempfile
import unittest

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images

from puma.matshow import MatshowPlot
from puma.utils import logger, set_log_level

set_log_level(logger, "DEBUG")


class TestMatshowPlot(unittest.TestCase):
    def setUp(self):
        # Set up directories for comparison plots
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        self.expected_plots_dir = os.path.join(os.path.dirname(__file__), "expected_plots")

    def test_matrix(self):
        # test matrix
        mat = np.array([
            [0.55136792, 0.58809975, 0.43140183],
            [0.83374029, 0.24685411, 0.87419068],
            [0.08393056, 0.3477517, 0.81477693],
            [0.39839215, 0.54854937, 0.48571167],
        ])
        plot_mat = MatshowPlot(colormap=plt.cm.PiYG, x_ticks_rotation=0)
        name = "test_matrix.png"
        plot_mat.draw(mat)
        plot_mat.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")

        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

    def test_matrix_nocbar(self):
        # test matrix
        mat = np.array([
            [0.55136792, 0.58809975, 0.43140183],
            [0.83374029, 0.24685411, 0.87419068],
            [0.08393056, 0.3477517, 0.81477693],
            [0.39839215, 0.54854937, 0.48571167],
        ])
        plot_mat = MatshowPlot(colormap=plt.cm.PiYG, x_ticks_rotation=0, show_cbar=False)
        name = "test_matrix_nocbar.png"
        plot_mat.draw(mat)
        plot_mat.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")

        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

    def test_matrix_fully_customized(self):
        # test matrix
        mat = np.array([
            [0.55136792, 0.58809975, 0.43140183],
            [0.83374029, 0.24685411, 0.87419068],
            [0.08393056, 0.3477517, 0.81477693],
            [0.39839215, 0.54854937, 0.48571167],
        ])
        x_ticks = ["a", "b", "c"]
        y_ticks = ["d", "e", "f", "g"]

        plot_mat = MatshowPlot(
            x_ticklabels=x_ticks,
            x_ticks_rotation=45,
            y_ticklabels=y_ticks,
            show_entries=True,
            show_percentage=True,
            text_color_threshold=0.6,
            cbar_label="Scalar values",
        )
        plot_mat.draw(mat)
        name = "test_matrix_fully_customized.png"
        plot_mat.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")

        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

    def test_matrix_fully_customized_no_entries(self):
        # test matrix
        mat = np.array([
            [0.55136792, 0.58809975, 0.43140183],
            [0.83374029, 0.24685411, 0.87419068],
            [0.08393056, 0.3477517, 0.81477693],
            [0.39839215, 0.54854937, 0.48571167],
        ])
        x_ticks = ["a", "b", "c"]
        y_ticks = ["d", "e", "f", "g"]

        plot_mat = MatshowPlot(
            x_ticklabels=x_ticks,
            x_ticks_rotation=45,
            y_ticklabels=y_ticks,
            show_entries=False,
            show_percentage=True,
            text_color_threshold=0.6,
            cbar_label="Scalar values",
        )
        name = "test_matrix_fully_customized_no_entries.png"
        plot_mat.draw(mat)
        plot_mat.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")

        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )
