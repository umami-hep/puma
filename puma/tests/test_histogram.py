#!/usr/bin/env python

"""
Unit test script for the functions in histogram.py
"""

import os
import tempfile
import unittest

import numpy as np
from matplotlib.testing.compare import compare_images

from puma import Histogram, HistogramPlot
from puma.utils import logger, set_log_level

set_log_level(logger, "DEBUG")


class histogram_TestCase(unittest.TestCase):
    """Test class for the puma.histogram functions."""

    def test_empty_histogram(self):
        """test if providing wrong input type to histogram raises ValueError"""
        with self.assertRaises(ValueError):
            Histogram(values=5)

    def test_divide_before_plotting(self):
        """test if ValueError is raised when dividing before plotting the histograms"""
        hist_1 = Histogram([1, 1, 1, 2, 2])
        hist_2 = Histogram([1, 2, 2, 2])
        with self.assertRaises(ValueError):
            hist_1.divide(hist_2)

    def test_divide_after_plotting_no_norm(self):
        """test if ratio is calculated correctly after plotting (without norm)"""
        hist_1 = Histogram([1, 1, 1, 2, 2])
        hist_2 = Histogram([1, 2, 2, 2])
        bins = np.array([1, 2, 3])
        hist_plot = HistogramPlot(bins=bins, norm=False)
        hist_plot.add(hist_1)
        hist_plot.add(hist_2)
        hist_plot.plot()
        # Since plotting is done with the matplotlib step function, the first bin is
        # duplicated in the ratio calculation (the first one is so to say not plotted)
        # Therefore, we also use duplicated bins here
        expected_ratio = np.array([3, 3, 2 / 3])
        expected_ratio_unc = np.array([3.46410162, 3.46410162, 0.60858062])

        np.testing.assert_almost_equal(expected_ratio, hist_1.divide(hist_2)[0])
        np.testing.assert_almost_equal(expected_ratio_unc, hist_1.divide(hist_2)[1])

    def test_divide_after_plotting_norm(self):
        """test if ratio is calculated correctly after plotting (with norm)"""
        hist_1 = Histogram([1, 1, 1, 2, 2])
        hist_2 = Histogram([1, 2, 2, 2])
        bins = np.array([1, 2, 3])
        hist_plot = HistogramPlot(bins=bins, norm=True)
        hist_plot.add(hist_1)
        hist_plot.add(hist_2)
        hist_plot.plot()
        # Since plotting is done with the matplotlib step function, the first bin is
        # duplicated in the ratio calculation (the first one is so to say not plotted)
        # Therefore, we also use duplicated bins here
        expected_ratio = np.array([2.4, 2.4, 0.53333333])
        expected_ratio_unc = np.array([2.77128129, 2.77128129, 0.4868645])

        np.testing.assert_almost_equal(expected_ratio, hist_1.divide(hist_2)[0])
        np.testing.assert_almost_equal(expected_ratio_unc, hist_1.divide(hist_2)[1])

    def test_ratio_same_histogram(self):
        """test if ratio is 1 for equal histograms (with norm)"""
        hist_1 = Histogram([1, 1, 1, 2, 2])
        hist_2 = Histogram([1, 1, 1, 2, 2])
        bins = np.array([1, 2, 3])
        hist_plot = HistogramPlot(bins=bins, norm=True)
        hist_plot.add(hist_1)
        hist_plot.add(hist_2)
        hist_plot.plot()
        # Since plotting is done with the matplotlib step function, the first bin is
        # duplicated in the ratio calculation (the first one is so to say not plotted)
        # Therefore, we also use duplicated bins here
        expected_ratio = np.ones(3)
        expected_ratio_unc = np.array([0.81649658, 0.81649658, 1])

        np.testing.assert_almost_equal(expected_ratio, hist_1.divide(hist_2)[0])
        np.testing.assert_almost_equal(expected_ratio_unc, hist_1.divide(hist_2)[1])


class histogram_plot_TestCase(unittest.TestCase):
    """Test class for puma.histogram_plot"""

    def setUp(self):

        np.random.seed(42)
        n_random = 10_000
        self.hist_1 = Histogram(
            np.random.normal(size=n_random), label=f"N={n_random:_}"
        )
        self.hist_2 = Histogram(
            np.random.normal(size=2 * n_random), label=f"N={2*n_random:_}"
        )

        # Set up directories for comparison plots
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        self.expected_plots_dir = os.path.join(
            os.path.dirname(__file__), "expected_plots"
        )

    def test_invalid_bins_type(self):
        """check if ValueError is raised when using invalid type in `bins` argument"""
        hist_plot = HistogramPlot(bins=1.1)
        hist_plot.add(self.hist_1, reference=True)
        with self.assertRaises(ValueError):
            hist_plot.plot()

    def test_add_bin_width_to_ylabel(self):
        """check if ValueError is raised when using invalid type in `bins` argument"""
        hist_plot = HistogramPlot(bins=60)
        hist_plot.add(self.hist_1, reference=True)
        with self.assertRaises(ValueError):
            hist_plot.add_bin_width_to_ylabel()

    def test_multiple_references_no_flavour(self):
        """Tests if error is raised in case of non-unique reference histogram"""
        hist_plot = HistogramPlot(n_ratio_panels=1)
        hist_plot.add(self.hist_1, reference=True)
        hist_plot.add(self.hist_2, reference=True)
        hist_plot.add(self.hist_1)
        hist_plot.plot()
        with self.assertRaises(ValueError):
            hist_plot.plot_ratios()

    def test_multiple_references_flavour(self):
        """Tests if error is raised in case of non-unique reference histogram
        when flavoured histograms are used"""
        hist_plot = HistogramPlot(n_ratio_panels=1)
        dummy_array = np.array([1, 1, 2, 3, 2, 3])
        hist_plot.add(Histogram(dummy_array, flavour="ujets"), reference=True)
        hist_plot.add(Histogram(dummy_array, flavour="ujets"), reference=True)
        hist_plot.add(Histogram(dummy_array, flavour="ujets"))
        hist_plot.plot()
        with self.assertRaises(ValueError):
            hist_plot.plot_ratios()

    def test_custom_range(self):
        """check if
        1. bins_range argument is used correctly
        2. deactivate ATLAS branding works
        3. adding bin width to ylabel works
        """
        hist_plot = HistogramPlot(
            bins=20,
            bins_range=(0, 4),
            atlas_brand="",
            atlas_first_tag="Simulation, $\\sqrt{s}=13$ TeV",
            atlas_second_tag="Second tag with additional\ndistance from first tag",
            figsize=(5, 4),
            ylabel="Number of jets",
            n_ratio_panels=1,
            atlas_second_tag_distance=0.3,
        )
        hist_plot.add(self.hist_1, reference=True)
        hist_plot.draw()
        hist_plot.add_bin_width_to_ylabel()

        plotname = "test_histogram_custom_range.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # hist_plot.savefig(f"{self.expected_plots_dir}/{plotname}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )

    def test_output_ratio(self):
        """check with a plot if the ratio is the expected value"""
        hist_plot = HistogramPlot(
            norm=False,
            ymax_ratio_1=4,
            figsize=(6.5, 5),
            n_ratio_panels=1,
        )
        hist_plot.add(self.hist_1, reference=True)
        hist_plot.add(self.hist_2)
        hist_plot.draw()
        hist_plot.axis_ratio_1.axhline(2, color="r", label="Expected ratio")
        hist_plot.axis_ratio_1.legend(frameon=False)

        plotname = "test_histogram_ratio_value.png"
        plotname_transparent = "test_histogram_ratio_value_transparent.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # hist_plot.savefig(f"{self.expected_plots_dir}/{plotname}")

        # Also save this plot with transparent background to test this feature
        hist_plot.transparent = True
        hist_plot.savefig(f"{self.actual_plots_dir}/{plotname_transparent}")
        # Uncomment line below to update expected image
        # hist_plot.savefig(f"{self.expected_plots_dir}/{plotname_transparent}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname_transparent}",
                f"{self.expected_plots_dir}/{plotname_transparent}",
                tol=1,
            )
        )

    def test_output_empty_histogram_norm(self):
        hist_plot = HistogramPlot(
            norm=True,
            figsize=(6.5, 5),
            atlas_second_tag=(
                "Test if ratio is 1 for whole range if reference histogram is empty\n"
                "(+ normalised)"
            ),
            n_ratio_panels=1,
        )
        hist_plot.add(Histogram(np.array([]), label="empty histogram"), reference=True)
        hist_plot.add(self.hist_1)
        hist_plot.draw()

        plotname = "test_histogram_empty_reference_norm.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # hist_plot.savefig(f"{self.expected_plots_dir}/{plotname}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )

    def test_output_empty_histogram_no_norm(self):
        """Test if ratio is 1 for whole range if reference histogram is empty"""
        hist_plot = HistogramPlot(
            norm=False,
            figsize=(6.5, 5),
            atlas_second_tag=(
                "Test if ratio is 1 for whole range if reference histogram is empty"
            ),
            n_ratio_panels=1,
        )
        hist_plot.add(Histogram(np.array([]), label="empty histogram"), reference=True)
        hist_plot.add(self.hist_1)
        hist_plot.draw()

        plotname = "test_histogram_empty_reference_no_norm.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # hist_plot.savefig(f"{self.expected_plots_dir}/{plotname}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )

    def test_output_different_range_histogram(self):
        """Test if ratio yields the expected values for case of different histogram
        ranges"""

        hist_plot = HistogramPlot(
            atlas_second_tag=(
                "Test ratio for the case of different histogram ranges. \n"
            ),
            xlabel="x",
            figsize=(7, 6),
            leg_loc="upper right",
            y_scale=1.5,
            n_ratio_panels=1,
        )
        np.random.seed(42)
        n_random = 10_000
        x1 = np.concatenate(
            (
                np.random.uniform(-2, 0, n_random),
                np.random.uniform(0.5, 0.99, int(0.5 * n_random)),
            )
        )
        x2 = np.random.uniform(0, 2, n_random)
        x3 = np.random.uniform(-1, 1, n_random)
        hist_plot.add(
            Histogram(x1, label="uniform [-2, 0] and uniform [0.5, 1] \n(reference)"),
            reference=True,
        )
        hist_plot.add(Histogram(x2, label="uniform [0, 2]"))
        hist_plot.add(Histogram(x3, label="uniform [-1, 1]"))
        hist_plot.draw()

        plotname = "test_histogram_different_ranges.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # hist_plot.savefig(f"{self.expected_plots_dir}/{plotname}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )

    def test_draw_vlines_histogram(self):
        """Test the drawing of vertical lines in the histogram."""
        hist_plot = HistogramPlot(
            bins=20,
            bins_range=(0, 4),
            atlas_brand="",
            atlas_first_tag="Simulation, $\\sqrt{s}=13$ TeV",
            atlas_tag_outside=True,
            figsize=(5, 4),
            ylabel="Number of jets",
            n_ratio_panels=1,
        )
        hist_plot.add(self.hist_1, reference=True)
        hist_plot.draw_vlines(
            vlines_xvalues=[1, 2, 3],
        )
        hist_plot.draw()

        plotname = "test_draw_vlines_histogram.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # hist_plot.savefig(f"{self.expected_plots_dir}/{plotname}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )

    def test_draw_vlines_histogram_custom_labels(self):
        """Test the drawing of vertical lines in the histogram."""
        hist_plot = HistogramPlot(
            bins=20,
            bins_range=(0, 4),
            atlas_brand="",
            atlas_first_tag="Simulation, $\\sqrt{s}=13$ TeV",
            atlas_second_tag="",
            figsize=(5, 4),
            ylabel="Number of jets",
            n_ratio_panels=1,
        )
        hist_plot.add(self.hist_1, reference=True)
        hist_plot.draw_vlines(
            vlines_xvalues=[1, 2, 3],
            vlines_label_list=["One", "Two", "Three"],
        )
        hist_plot.draw()

        plotname = "test_draw_vlines_histogram_custom_labels.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # hist_plot.savefig(f"{self.expected_plots_dir}/{plotname}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )

    def test_draw_vlines_histogram_same_height(self):
        """Test the drawing of vertical lines in the histogram."""
        hist_plot = HistogramPlot(
            bins=20,
            bins_range=(0, 4),
            atlas_brand="",
            atlas_first_tag="Simulation, $\\sqrt{s}=13$ TeV",
            atlas_second_tag="",
            figsize=(5, 4),
            ylabel="Number of jets",
            n_ratio_panels=1,
        )
        hist_plot.add(self.hist_1, reference=True)
        hist_plot.draw_vlines(
            vlines_xvalues=[1, 2, 3],
            same_height=True,
        )
        hist_plot.draw()

        plotname = "test_draw_vlines_histogram_same_height.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # hist_plot.savefig(f"{self.expected_plots_dir}/{plotname}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )

    def test_draw_vlines_histogram_custom_yheight(self):
        """Test the drawing of vertical lines in the histogram."""
        hist_plot = HistogramPlot(
            bins=20,
            bins_range=(0, 4),
            atlas_brand="",
            atlas_first_tag="Simulation, $\\sqrt{s}=13$ TeV",
            atlas_second_tag="",
            figsize=(5, 4),
            ylabel="Number of jets",
            n_ratio_panels=1,
        )
        hist_plot.add(self.hist_1, reference=True)
        hist_plot.draw_vlines(
            vlines_xvalues=[1, 2, 3],
            vlines_line_height_list=[0.7, 0.6, 0.5],
        )
        hist_plot.draw()

        plotname = "test_draw_vlines_histogram_custom_yheight.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # hist_plot.savefig(f"{self.expected_plots_dir}/{plotname}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )

    def test_ratio_group_options(self):
        """Test different combinations of using ratio groups."""
        hist_plot = HistogramPlot(
            n_ratio_panels=1,
            atlas_brand=None,
            atlas_first_tag="",
            atlas_second_tag=(
                "Unit test plot to test the ratio_group argument \n"
                "of the puma.Histogram class"
            ),
            figsize=(6, 5),
            y_scale=1.5,
        )

        rng = np.random.default_rng(seed=42)
        # add two histograms with flavour=None but ratio_goup set
        hist_plot.add(
            Histogram(
                rng.normal(0, 1, size=10_000),
                ratio_group="ratio group 1",
                label="Ratio group 1 (reference)",
            ),
            reference=True,
        )
        hist_plot.add(
            Histogram(
                rng.normal(0.5, 1, size=10_000),
                ratio_group="ratio group 1",
                label="Ratio group 1",
            ),
        )
        # add two histograms with defining ratio group via flavour argument
        hist_plot.add(
            Histogram(
                rng.normal(3, 1, size=10_000),
                flavour="bjets",
                ratio_group="Ratio group 2",
                label="(reference)",
            ),
            reference=True,
        )
        hist_plot.add(
            Histogram(
                rng.normal(3.5, 1, size=10_000),
                flavour="bjets",
                ratio_group="Ratio group 2",
                linestyle="--",
                linewidth=3,
                alpha=0.3,
            ),
        )
        hist_plot.draw()

        plotname = "test_ratio_groups.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # hist_plot.savefig(f"{self.expected_plots_dir}/{plotname}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )

    def test_flavoured_labels(self):
        """Test different combinations of specifying the label when also specifying a
        flavour for the histogram."""

        rng = np.random.default_rng(seed=42)

        hist_plot = HistogramPlot(
            bins=100,
            atlas_brand=None,
            atlas_first_tag="",
            atlas_second_tag=(
                "Unit test plot to test the behaviour of \n"
                "the legend labels for different combinations \n"
                "of using the flavour label from the global \n"
                "config and not doing so"
            ),
            figsize=(8, 6),
        )
        # No flavour
        hist_plot.add(
            Histogram(rng.normal(0, 1, size=10_000), label="Unflavoured histogram")
        )
        # Flavour, but also label (using the default flavour label + the specified one)
        hist_plot.add(
            Histogram(
                rng.normal(4, 1, size=10_000),
                label="(flavoured, adding default flavour label '$b$-jets' to legend)",
                flavour="bjets",
            )
        )
        # Flavour + label (this time with suppressing the default flavour label)
        hist_plot.add(
            Histogram(
                rng.normal(8, 1, size=10_000),
                label="Flavoured histogram (default flavour label suppressed)",
                flavour="bjets",
                add_flavour_label=False,
                linestyle="--",
            )
        )
        # Flavoured, but using custom colour
        hist_plot.add(
            Histogram(
                rng.normal(12, 1, size=10_000),
                label="(flavoured, with custom colour)",
                flavour="bjets",
                linestyle="dotted",
                colour="b",
            )
        )
        hist_plot.draw()

        plotname = "test_flavoured_labels.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{plotname}")
        # Uncomment line below to update expected image
        # hist_plot.savefig(f"{self.expected_plots_dir}/{plotname}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{plotname}",
                f"{self.expected_plots_dir}/{plotname}",
                tol=1,
            )
        )
