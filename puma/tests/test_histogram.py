"""Unit test script for the functions in histogram.py."""

from __future__ import annotations

import os
import shutil  # noqa: F401
import tempfile
import unittest

import numpy as np
from ftag import Flavours
from matplotlib.testing.compare import compare_images

from puma import Histogram, HistogramPlot
from puma.utils import logger, set_log_level

set_log_level(logger, "DEBUG")


class HistogramTestCase(unittest.TestCase):
    """Test class for the puma.histogram functions."""

    def test_empty_histogram(self):
        """Test if providing wrong input type to histogram raises ValueError."""
        with self.assertRaises(TypeError):
            Histogram(values=5)

    def test_divide_before_plotting(self):
        """Test if ValueError is raised when dividing before plotting the histograms."""
        hist_1 = Histogram([1, 1, 1, 2, 2])
        hist_2 = Histogram([1, 2, 2, 2])
        with self.assertRaises(ValueError):
            hist_1.divide(hist_2)

    def test_divide_after_plotting_no_norm(self):
        """Test if ratio is calculated correctly after plotting (without norm)."""
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
        expected_ratio_unc = np.array([1.73205081, 1.73205081, 0.47140452])

        np.testing.assert_almost_equal(expected_ratio, hist_1.divide(hist_2)[0])
        np.testing.assert_almost_equal(expected_ratio_unc, hist_1.divide(hist_2)[1])

    def test_divide_after_plotting_norm(self):
        """Test if ratio is calculated correctly after plotting (with norm)."""
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
        expected_ratio_unc = np.array([1.38564065, 1.38564065, 0.37712362])

        np.testing.assert_almost_equal(expected_ratio, hist_1.divide(hist_2)[0])
        np.testing.assert_almost_equal(expected_ratio_unc, hist_1.divide(hist_2)[1])

    def test_ratio_same_histogram(self):
        """Test if ratio is 1 for equal histograms (with norm)."""
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
        expected_ratio_unc = np.array([0.57735027, 0.57735027, 0.70710678])

        np.testing.assert_almost_equal(expected_ratio, hist_1.divide(hist_2)[0])
        np.testing.assert_almost_equal(expected_ratio_unc, hist_1.divide(hist_2)[1])

    def test_divide_wrong_bin_edges(self):
        """Test if error is raised if bin edges don't match."""
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
        hist_2.bin_edges = np.array([1, 2])
        with self.assertRaises(ValueError):
            hist_1.divide(hist_2)

    def test_multiple_references_wrong_flavour(self):
        """Tests if warning is raised with wrong flavour."""
        dummy_array = np.array([1, 1, 2, 3, 2, 3])
        with self.assertRaises(KeyError):
            Histogram(dummy_array, flavour="dummy")


class HistogramPlotTestCase(unittest.TestCase):
    """Test class for puma.histogram_plot."""

    def setUp(self):
        np.random.seed(42)
        n_random = 10_000
        self.hist_1 = Histogram(np.random.normal(size=n_random), label=f"N={n_random:_}")
        self.hist_2 = Histogram(np.random.normal(size=2 * n_random), label=f"N={2 * n_random:_}")
        self.data_hist = Histogram(
            np.random.normal(size=3 * n_random),
            label=f"Toy Data, N={3 * n_random:_}",
            is_data=True,
            colour="k",
        )

        # Set up directories for comparison plots
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        self.expected_plots_dir = os.path.join(os.path.dirname(__file__), "expected_plots")

    def test_invalid_bins_type(self):
        """Check if ValueError is raised when using invalid type in `bins` argument."""
        hist_plot = HistogramPlot(bins=1.1)
        hist_plot.add(self.hist_1, reference=True)
        with self.assertRaises(TypeError):
            hist_plot.plot()

    def test_add_bin_width_to_ylabel(self):
        """Check if ValueError is raised when using invalid type in `bins` argument."""
        hist_plot = HistogramPlot(bins=60)
        hist_plot.add(self.hist_1, reference=True)
        with self.assertRaises(ValueError):
            hist_plot.add_bin_width_to_ylabel()

    def test_multiple_references_no_flavour(self):
        """Tests if error is raised in case of non-unique reference histogram."""
        hist_plot = HistogramPlot(n_ratio_panels=1)
        hist_plot.add(self.hist_1, reference=True)
        hist_plot.add(self.hist_2, reference=True)
        hist_plot.add(self.hist_1)
        hist_plot.plot()
        with self.assertRaises(ValueError):
            hist_plot.plot_ratios()

    def test_multiple_references_flavour(self):
        """Tests if error is raised in case of non-unique reference histogram
        when flavoured histograms are used
        .
        """
        hist_plot = HistogramPlot(n_ratio_panels=1)
        dummy_array = np.array([1, 1, 2, 3, 2, 3])
        hist_plot.add(Histogram(dummy_array, flavour="ujets"), reference=True)
        hist_plot.add(Histogram(dummy_array, flavour="ujets"), reference=True)
        hist_plot.add(Histogram(dummy_array, flavour="ujets"))
        hist_plot.plot()
        with self.assertRaises(ValueError):
            hist_plot.plot_ratios()

    def test_two_ratio_panels(self):
        """Tests if error is raised if more than 1 ratio panel is requested."""
        with self.assertRaises(ValueError):
            HistogramPlot(n_ratio_panels=2)

    def test_custom_range(self):
        """Check if
        1. bins_range argument is used correctly
        2. deactivate ATLAS branding works
        3. adding bin width to ylabel works.
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

        name = "test_histogram_custom_range.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

    def test_data_mc(self):
        """Check if data mc looks good."""
        hist_plot = HistogramPlot(
            bins=20,
            bins_range=[-2, 2],
            atlas_brand="",
            atlas_first_tag="Simulation, $\\sqrt{s}=13$ TeV",
            atlas_second_tag="Second tag with additional\ndistance from first tag",
            figsize=(5, 4),
            ylabel="Number of jets",
            n_ratio_panels=1,
            atlas_second_tag_distance=0.3,
            stacked=True,
            norm=False,
        )
        hist_plot.add(self.hist_1)
        hist_plot.add(self.hist_2)
        hist_plot.add(self.data_hist)
        hist_plot.draw()
        hist_plot.add_bin_width_to_ylabel()

        name = "test_histogram_data_mc.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

    def test_discrete_values(self):
        """Check if discrete values are working properly."""
        hist_plot = HistogramPlot(
            bins=np.linspace(0, 10, 100),
            discrete_vals=[0, 5, 7, 9],
            atlas_first_tag="Simulation, $\\sqrt{s}=13$ TeV",
            figsize=(5, 4),
            xlabel="Discrete values",
            ylabel="Number of counts in specified bins",
            n_ratio_panels=1,
        )
        # the entry "1" in `values_1` will be hidden in the histogram since it is not
        # included in the `discrete_vals` list
        values1 = np.array([0, 1, 5, 7])
        values2 = np.array([0, 5, 5, 7])
        hist_plot.add(Histogram(values1), reference=True)
        hist_plot.add(Histogram(values2, linestyle="--"))
        hist_plot.draw()

        name = "test_histogram_discrete_values.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

    def test_output_ratio(self):
        """Check with a plot if the ratio is the expected value."""
        hist_plot = HistogramPlot(
            norm=False,
            ymax_ratio=[4],
            figsize=(6.5, 5),
            n_ratio_panels=1,
        )
        hist_plot.add(self.hist_1, reference=True)
        hist_plot.add(self.hist_2)
        hist_plot.draw()
        hist_plot.ratio_axes[0].axhline(2, color="r", label="Expected ratio")
        hist_plot.ratio_axes[0].legend(frameon=False)

        name = "test_histogram_ratio_value.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

        # Also save this plot with transparent background to test this feature
        name_transparent = "test_histogram_ratio_value_transparent.png"
        hist_plot.transparent = True
        hist_plot.savefig(f"{self.actual_plots_dir}/{name_transparent}")
        # Uncomment line below to update expected image
        # shutil.copy(
        #     f"{self.actual_plots_dir}/{name_transparent}",
        #     f"{self.expected_plots_dir}/{name_transparent}",
        # )
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name_transparent}",
                f"{self.expected_plots_dir}/{name_transparent}",
                tol=1,
            )
        )

    def test_output_empty_histogram_norm(self):
        """Test empty normed histogram."""
        hist_plot = HistogramPlot(
            norm=True,
            figsize=(6.5, 5),
            atlas_second_tag=(
                "Test if ratio is 1 for whole range if reference histogram is empty\n(+"
                " normalised)"
            ),
            n_ratio_panels=1,
        )
        hist_plot.add(Histogram(np.array([]), label="empty histogram"), reference=True)
        hist_plot.add(self.hist_1)
        hist_plot.draw()

        name = "test_histogram_empty_reference_norm.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

    def test_output_empty_histogram_no_norm(self):
        """Test if ratio is 1 for whole range if reference histogram is empty."""
        hist_plot = HistogramPlot(
            norm=False,
            figsize=(6.5, 5),
            atlas_second_tag=("Test if ratio is 1 for whole range if reference histogram is empty"),
            n_ratio_panels=1,
        )
        hist_plot.add(Histogram(np.array([]), label="empty histogram"), reference=True)
        hist_plot.add(self.hist_1)
        hist_plot.draw()

        name = "test_histogram_empty_reference_no_norm.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

    def test_output_different_range_histogram(self):
        """Test if ratio yields the expected values for case of different histogram
        ranges
        .
        """
        hist_plot = HistogramPlot(
            atlas_second_tag=("Test ratio for the case of different histogram ranges. \n"),
            xlabel="x",
            figsize=(7, 6),
            leg_loc="upper right",
            y_scale=1.5,
            n_ratio_panels=1,
        )
        np.random.seed(42)
        n_random = 10_000
        arr_1 = np.concatenate((
            np.random.uniform(-2, 0, n_random),
            np.random.uniform(0.5, 0.99, int(0.5 * n_random)),
        ))
        arr_2 = np.random.uniform(0, 2, n_random)
        arr_3 = np.random.uniform(-1, 1, n_random)
        hist_plot.add(
            Histogram(arr_1, label="uniform [-2, 0] and uniform [0.5, 1] \n(reference)"),
            reference=True,
        )
        hist_plot.add(Histogram(arr_2, label="uniform [0, 2]"))
        hist_plot.add(Histogram(arr_3, label="uniform [-1, 1]"))
        hist_plot.draw()

        name = "test_histogram_different_ranges.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
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
            xs=[1, 2, 3],
        )
        hist_plot.draw()

        name = "test_draw_vlines_histogram.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
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
            xs=[1, 2, 3],
            labels=["One", "Two", "Three"],
        )
        hist_plot.draw()

        name = "test_draw_vlines_histogram_custom_labels.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
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
            xs=[1, 2, 3],
            same_height=True,
        )
        hist_plot.draw()

        name = "test_draw_vlines_histogram_same_height.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
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
            xs=[1, 2, 3],
            ys=[0.7, 0.6, 0.5],
        )
        hist_plot.draw()

        name = "test_draw_vlines_histogram_custom_yheight.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

    def test_ratio_group_argument(self):
        """Test different combinations of using ratio groups."""
        hist_plot = HistogramPlot(
            n_ratio_panels=1,
            atlas_brand=None,
            atlas_first_tag="",
            atlas_second_tag=(
                "Unit test plot to test the ratio_group argument \n"
                "of the puma.Histogram class.\n"
                "Also testing the linestyle legend which is\n"
                "placed here at (0.25, 0.75)."
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
                colour=Flavours["ujets"].colour,
                label=Flavours["ujets"].label,
            ),
            reference=True,
        )
        hist_plot.add(
            Histogram(
                rng.normal(0.5, 1, size=10_000),
                colour=Flavours["ujets"].colour,
                ratio_group="ratio group 1",
                linestyle="--",
            ),
        )
        hist_plot.add(
            Histogram(
                rng.normal(3, 1, size=10_000),
                label=Flavours["bjets"].label,
                colour=Flavours["bjets"].colour,
                ratio_group="Ratio group 2",
            ),
            reference=True,
        )
        hist_plot.add(
            Histogram(
                rng.normal(3.5, 1, size=10_000),
                colour=Flavours["bjets"].colour,
                ratio_group="Ratio group 2",
                linestyle="--",
                linewidth=3,
                alpha=0.3,
            ),
        )
        hist_plot.draw()

        hist_plot.make_linestyle_legend(
            labels=["Reference", "Shifted"],
            linestyles=["solid", "dashed"],
            bbox_to_anchor=(0.25, 0.75),
            loc="upper right",
        )

        name = "test_ratio_groups.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")

        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1.4,
            )
        )

    def test_flavoured_labels(self):
        """Test different combinations of specifying the label when also specifying a
        flavour for the histogram.
        """
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
        hist_plot.add(Histogram(rng.normal(0, 1, size=10_000), label="Unflavoured histogram"))
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

        name = "test_flavoured_labels.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

    def test_weights(self):
        """Output plot with weights."""
        values = np.array([])
        values = np.array([0, 1, 2, 2, 3])
        weights = np.array([1, -1, 3, -2, 1])
        hist_exp = np.array([1, -1, 2])
        unc_exp = np.sqrt(np.array([1, (-1) ** 2, 3**2 + (-2) ** 2 + 1]))

        hist_plot = HistogramPlot(
            bins=3,
            figsize=(6, 5),
            atlas_first_tag=None,
            atlas_second_tag=(
                "Test plot for the behaviour of weights \n(both positive and negative)"
                f"\nExpected bin counts: {hist_exp}"
                f"\nExpected bin uncertainties: {unc_exp}"
            ),
            norm=False,
        )
        hist_plot.add(
            Histogram(
                values,
                weights=weights,
            )
        )
        hist_plot.draw()

        name = "test_histogram_weights.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

    def test_weights_wrong_shape(self):
        """Check if ValueError is raised if wieghts has."""
        values = np.array([0, 1, 2, 2, 3])
        weights = np.array([1, -1])

        with self.assertRaises(ValueError):
            Histogram(
                values,
                weights=weights,
            )

    def test_underoverflow_bin(self):
        """Test if under/overflow bins work as expected."""
        vals = [-1, 1, 2, 3, 6]
        vals_with_inf = [-1, 1, 2, 6, np.inf]

        hist_plot = HistogramPlot(bins=5, bins_range=(0, 5), underoverflow=False)
        hist_plot.atlas_second_tag = "This plot does not have under/overflow bins"
        hist_plot.add(Histogram(vals, colour="b"))
        hist_plot.add(Histogram(vals_with_inf, colour="r", linestyle="dotted"))
        hist_plot.draw()

        name = "test_histogram_without_underoverflow_bins.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

        hist_plot = HistogramPlot(bins=5, bins_range=(0, 5))
        hist_plot.atlas_second_tag = "This plot has under/overflow bins"
        hist_plot.add(Histogram(vals, colour="b"))
        hist_plot.add(Histogram(vals_with_inf, colour="r", linestyle="dotted"))
        hist_plot.draw()

        name = "test_histogram_with_underoverflow_bins.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

    def test_plot_filled_hist(self):
        bin_edges = [0, 1, 2, 3, 4, 5]
        bin_counts = [5, 4, 7, 12, 2]

        vals = [0, 1, 1, 5, 4, 2, 1, 3, 3, 5, 5, 5, 5, 5]

        hist_filled = Histogram(bin_counts, bin_edges=bin_edges)
        hist_notfilled = Histogram(vals)

        hist_plot = HistogramPlot(bins=bin_edges, underoverflow=False)
        hist_plot.add(hist_filled)
        hist_plot.add(hist_notfilled)

        hist_plot.draw()
        name = "test_filled_histogram.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{name}")

        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")

        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )

    def test_plot_filled_hist_sumW2(self):
        bin_edges = [0, 1, 2, 3, 4, 5]
        bin_counts = [5, 4, 7, 12, 2]
        sum_squared_weights = [10, 7, 12, 21, 5]

        hist_plot = HistogramPlot(bins=bin_edges, underoverflow=False)
        hist_plot.add(
            Histogram(bin_counts, bin_edges=bin_edges, sum_squared_weights=sum_squared_weights)
        )

        hist_plot.draw()
        name = "test_filled_histogram_sumW2.png"
        hist_plot.savefig(f"{self.actual_plots_dir}/{name}")

        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")

        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            )
        )
