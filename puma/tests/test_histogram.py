"""Unit test script for the functions in histogram.py."""

from __future__ import annotations

import json
import os
import shutil  # noqa: F401
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
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

    def test_divide(self):
        """Test the division of two histograms."""
        hist_1 = Histogram(values=[1, 1, 2, 2, 3, 3], bins=3)
        hist_2 = Histogram(values=[1, 1, 2, 2, 3, 3], bins=3)
        expected_ratio = np.array([1, 1, 1, 1])
        expected_ratio_unc = np.array([
            np.sqrt(2) / 2,
            np.sqrt(2) / 2,
            np.sqrt(2) / 2,
            np.sqrt(2) / 2,
        ])

        np.testing.assert_almost_equal(expected_ratio, hist_1.divide(hist_2)[0])
        np.testing.assert_almost_equal(expected_ratio_unc, hist_1.divide(hist_2)[1])

    def test_divide_different_binning(self):
        """Test if ValueError is raised when dividing histograms with different binning."""
        hist_1 = Histogram(values=[1, 1, 2, 2, 3, 3], bins=3)
        hist_2 = Histogram(values=[1, 1, 2, 2, 3, 3], bins=2)
        with self.assertRaises(ValueError):
            hist_1.divide(hist_2)

    def test_divide_norm(self):
        """Test if ratio is calculated correctly (with norm)."""
        bins = np.array([1, 2, 3])
        hist_1 = Histogram(values=[1, 1, 1, 2, 2], bins=bins, norm=True)
        hist_2 = Histogram(values=[1, 2, 2, 2], bins=bins, norm=True)
        expected_ratio = np.array([2.4, 2.4, 0.53333333])
        expected_ratio_unc = np.array([1.38564065, 1.38564065, 0.37712362])

        np.testing.assert_almost_equal(expected_ratio, hist_1.divide(hist_2)[0])
        np.testing.assert_almost_equal(expected_ratio_unc, hist_1.divide(hist_2)[1])

    def test_divide_data_mc(self):
        """Test the division of a data and a MC histogram."""
        bins = 3
        hist_1 = Histogram(values=[1, 1, 2, 2, 3, 3], bins=bins)
        hist_2 = np.ones(shape=bins) * (1 / 3)
        expected_ratio = np.array([1, 1, 1, 1])
        expected_ratio_unc = np.array([
            np.sqrt(2) / 2,
            np.sqrt(2) / 2,
            np.sqrt(2) / 2,
            np.sqrt(2) / 2,
        ])

        np.testing.assert_almost_equal(expected_ratio, hist_1.divide_data_mc(hist_2)[0])
        np.testing.assert_almost_equal(expected_ratio_unc, hist_1.divide_data_mc(hist_2)[1])

    def test_divide_data_mc_wrong_binnings(self):
        """Test the division of a data and a MC histogram with wrong binning shapes."""
        bins = 3
        hist_1 = Histogram(values=[1, 1, 2, 2, 3, 3], bins=bins)
        hist_2 = np.ones(shape=4) * (1 / 3)

        with self.assertRaises(ValueError):
            hist_1.divide_data_mc(hist_2)

    def test_multiple_references_wrong_flavour(self):
        """Tests if warning is raised with wrong flavour."""
        dummy_array = np.array([1, 1, 2, 3, 2, 3])
        with self.assertRaises(KeyError):
            Histogram(values=dummy_array, bins=3, flavour="dummy")

    def test_get_discrete_values(self):
        """Test the get_discrete_values standard behaviour."""
        bins = np.array([1, 2, 3])
        hist_1 = Histogram(
            values=[1, 1, 1, 2, 2],
            bins=bins,
            norm=True,
            discrete_vals=[1, 2],
        )

        np.testing.assert_almost_equal(hist_1.bin_edges, np.array([0, 1, 2]))

    def test_get_discrete_values_one_bin_error(self):
        """Test the get_discrete_values error if only one bin is defined."""
        with self.assertRaises(ValueError):
            Histogram(
                values=[1, 1, 1, 2, 2],
                bins=np.array([1]),
                norm=True,
                discrete_vals=[1, 2, 3],
            )

    def test_get_discrete_values_binwidth_error(self):
        """Test the get_discrete_values error if the bin width is above 1."""
        with self.assertRaises(ValueError):
            Histogram(
                values=[1, 1, 1, 2, 2],
                bins=np.array([1, 3]),
                norm=True,
                discrete_vals=[1, 2, 3],
            )

    def test_custom_range(self):
        """Test that the range is correctly adapted if defined."""
        hist_1 = Histogram(
            values=[1, 1, 1, 2, 2],
            bins=4,
            bins_range=(0, 4),
            norm=True,
        )
        np.testing.assert_almost_equal(hist_1.bin_edges, np.array([0, 1, 2, 3, 4]))

    def test_no_bins_no_bin_edges(self):
        """Test the raise of ValueError when neither bins nor bin_edges are provided."""
        with self.assertRaises(ValueError):
            Histogram(values=np.array([1, 2, 3]))

    def test_no_bin_edges_with_sum_squared_weights(self):
        """Test the logger warning that sum squared weights are ignored."""
        with self.assertLogs("puma", level="WARNING") as captured:
            Histogram(values=np.array([1, 2, 3]), bins=3, sum_squared_weights=np.array([1, 1, 1]))

        self.assertTrue(
            any("Parameter `sum_squared_weights` is ignored" in msg for msg in captured.output),
            "Expected warning not found",
        )

    def test_bins_and_bin_edges(self):
        """Test the logger warning that bins are ignored."""
        with self.assertLogs("puma", level="WARNING") as captured:
            Histogram(values=np.array([1, 2, 3]), bins=3, bin_edges=np.array([0, 1, 2, 3, 4]))

        self.assertTrue(
            any(
                "When bin_edges are provided, bins are not considered!" in msg
                for msg in captured.output
            ),
            "Expected warning not found",
        )


class HistogramIOTestCase(unittest.TestCase):
    """
    Unit-tests that ensure `Histogram` serialisation is lossless and robust.

    Each test creates a deterministic histogram, writes it to disk, reloads
    it, and verifies that arrays, metadata and behaviour are preserved.
    """

    # ------------------------------------------------------------------
    # helper to build a minimal deterministic Histogram
    # ------------------------------------------------------------------
    @staticmethod
    def _make_hist() -> Histogram:
        """
        Create a minimal but deterministic `Histogram` instance.

        Returns
        -------
        Histogram
            A histogram with three bins, assigned flavour and custom colour.
        """
        return Histogram(
            values=np.arange(6),
            bins=3,
            flavour="bjets",
            colour=(0.1, 0.6, 0.3),
            linestyle=(0, (1, 2)),
            label="unit-test",
        )

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------
    def testencodedecode_roundtrip(self):
        """
        `encode` must turn every value into a JSON/YAML-serialisable
        form, and `decode` must restore the original objects exactly.
        """
        h = self._make_hist()

        encoded = h.encode(h.args_to_store)

        # --- encoded structure is JSON-safe --------------------------------
        json.dumps(encoded)
        # numpy arrays became {"__ndarray__": …, "dtype": …}
        for field in ("bin_edges", "hist", "unc", "band"):
            self.assertIn("__ndarray__", encoded[field])

        # tuples are tagged
        self.assertEqual(
            encoded["colour"]["__tuple__"],
            [0.1, 0.6, 0.3],
        )

        # flavour became a tag (Label to name)
        self.assertEqual(
            encoded["flavour"]["__label__"],
            "bjets",
        )

        # --- decode restores original Python objects -----------------------
        decoded = Histogram.decode(encoded)

        self.assertEqual(decoded["colour"], h.colour)
        self.assertEqual(decoded["flavour"], h.flavour)
        self.assertIsNone(decoded["discrete_vals"])

        # ndarray round-trip
        np.testing.assert_array_equal(decoded["hist"], h.hist)

    # ------------------------------------------------------------------
    # save / load (JSON)
    # ------------------------------------------------------------------
    def test_json_roundtrip(self):
        """Saving to JSON and loading back must reproduce the object."""
        h = self._make_hist()

        with tempfile.TemporaryDirectory() as tmpd:
            path = Path(tmpd) / "hist.json"
            h.save(path)

            # The written file parses as valid JSON
            with path.open() as f:
                raw = json.load(f)
            self.assertIn("hist", raw)

            clone = Histogram.load(path)

            # numerical arrays survive bit-perfect
            np.testing.assert_array_equal(h.hist, clone.hist)
            np.testing.assert_array_equal(h.bin_edges, clone.bin_edges)
            np.testing.assert_array_equal(h.unc, clone.unc)
            np.testing.assert_array_equal(h.band, clone.band)

            # metadata
            self.assertEqual(clone.label, h.label)
            self.assertEqual(clone.flavour, h.flavour)
            self.assertEqual(clone.colour, h.colour)

            # functional behaviour intact
            ratio, _ = h.divide(clone)
            np.testing.assert_allclose(ratio, np.ones_like(ratio))

    # ------------------------------------------------------------------
    # save / load (YAML)
    # ------------------------------------------------------------------
    def test_yaml_roundtrip(self):
        """YAML round-trip should preserve arrays & metadata."""
        h = self._make_hist()

        with tempfile.TemporaryDirectory() as tmpd:
            path = Path(tmpd) / "hist.yaml"
            h.save(path)

            # YAML parses
            with path.open() as f:
                raw = yaml.safe_load(f)
            self.assertIn("hist", raw)

            clone = Histogram.load(path)
            self.assertEqual(clone.label, h.label)

    # ------------------------------------------------------------------
    # load-time overrides
    # ------------------------------------------------------------------
    def test_load_override(self):
        """
        `extra_kwargs` passed to `load` must override attributes stored
        in the file.
        """
        h = self._make_hist()
        with tempfile.TemporaryDirectory() as tmpd:
            path = Path(tmpd) / "hist.json"
            h.save(path)

            clone = Histogram.load(path, colour="magenta", label="override")
            self.assertEqual(clone.colour, "magenta")
            self.assertEqual(clone.label, "override")

    # ------------------------------------------------------------------
    # unknown file extensions
    # ------------------------------------------------------------------
    def test_invalid_extensions(self):
        """Unsupported suffixes must raise :class:`ValueError`."""
        h = self._make_hist()

        with tempfile.TemporaryDirectory() as tmpd:
            # save
            with self.assertRaises(ValueError):
                h.save(Path(tmpd) / "hist.txt")

            # load
            bogus = Path(tmpd) / "hist.conf"
            bogus.write_text("dummy")
            with self.assertRaises(ValueError):
                Histogram.load(bogus)


class HistogramPlotTestCase(unittest.TestCase):
    """Test class for puma.histogram_plot."""

    def setUp(self):
        np.random.seed(42)
        self.n_random = 10_000
        self.hist_1_values = np.random.normal(size=1 * self.n_random)
        self.hist_2_values = np.random.normal(size=2 * self.n_random)
        self.hist_3_values = np.random.normal(size=3 * self.n_random)
        self.hist_1 = Histogram(
            values=self.hist_1_values,
            bins=20,
            bins_range=(-2, 2),
            label=f"N={self.n_random:_}",
            norm=False,
        )
        self.hist_2 = Histogram(
            values=self.hist_2_values,
            bins=20,
            bins_range=(-2, 2),
            label=f"N={2 * self.n_random:_}",
            norm=False,
        )
        self.data_hist = Histogram(
            values=self.hist_3_values,
            bins=20,
            bins_range=(-2, 2),
            label=f"Toy Data, N={3 * self.n_random:_}",
            norm=False,
            is_data=True,
            colour="k",
        )

        # Set up directories for comparison plots
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        self.expected_plots_dir = os.path.join(os.path.dirname(__file__), "expected_plots")

    def test_different_norm_settings(self):
        """Check if ValueError is raised when using different norm settings."""
        dummy_array = np.array([1, 1, 2, 3, 2, 3])
        hist_plot = HistogramPlot()
        hist_plot.add(
            Histogram(values=dummy_array, bins=3, norm=True, flavour="ujets"),
            reference=True,
        )
        hist_plot.add(
            Histogram(
                values=dummy_array,
                bins=3,
                norm=False,
                flavour="ujets",
            )
        )
        with self.assertRaises(ValueError):
            hist_plot.draw()

    def test_different_bin_edges_settings(self):
        """Check if ValueError is raised when using different bin edges."""
        dummy_array = np.array([1, 1, 2, 3, 2, 3])
        hist_plot = HistogramPlot()
        hist_plot.add(
            Histogram(
                values=dummy_array,
                bin_edges=np.array([1, 2, 3]),
                flavour="ujets",
            ),
            reference=True,
        )
        hist_plot.add(
            Histogram(
                values=dummy_array,
                bin_edges=np.array([1, 2, 3, 4]),
                flavour="ujets",
            )
        )
        with self.assertRaises(ValueError):
            hist_plot.draw()

    def test_stacked_and_normed(self):
        """Check if ValueError is raised when stacked and normed are both true."""
        dummy_array = np.array([1, 1, 2, 3, 2, 3])
        hist_plot = HistogramPlot(stacked=True)
        hist_plot.add(
            Histogram(
                values=dummy_array,
                bins=3,
                norm=True,
                flavour="ujets",
            ),
            reference=True,
        )
        hist_plot.add(
            Histogram(
                values=dummy_array,
                bins=3,
                norm=True,
                flavour="ujets",
            )
        )
        with self.assertRaises(ValueError):
            hist_plot.draw()

    def test_add_bin_width_to_ylabel(self):
        """Check if ValueError is raised when using invalid type in `bins` argument."""
        hist_plot = HistogramPlot()
        hist_plot.add(self.hist_1, reference=True)
        with self.assertRaises(ValueError):
            hist_plot.add_bin_width_to_ylabel()

    def test_add_bin_width_to_ylabel_smaller_certain_value(self):
        """Check if the ylabel is correctly set at small bin widths."""
        hist_plot = HistogramPlot()
        hist_plot.add(
            Histogram(
                values=np.array([1, 2]),
                bins=np.linspace(0, 1, 1000),
                norm=True,
                flavour="ujets",
            ),
            reference=True,
        )
        hist_plot.draw()

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
        hist_plot.add(Histogram(values=dummy_array, bins=3, flavour="ujets"), reference=True)
        hist_plot.add(Histogram(values=dummy_array, bins=3, flavour="ujets"), reference=True)
        hist_plot.add(Histogram(values=dummy_array, bins=3, flavour="ujets"))
        hist_plot.plot()
        with self.assertRaises(ValueError):
            hist_plot.plot_ratios()

    def test_two_ratio_panels(self):
        """Tests if error is raised if more than 1 ratio panel is requested."""
        with self.assertRaises(ValueError):
            HistogramPlot(n_ratio_panels=2)

    def test_data_mc(self):
        """Check if data mc looks good."""
        hist_plot = HistogramPlot(
            atlas_brand="",
            atlas_first_tag="Simulation, $\\sqrt{s}=13$ TeV",
            atlas_second_tag="Second tag with additional\ndistance from first tag",
            figsize=(5, 4),
            ylabel="Number of jets",
            n_ratio_panels=1,
            atlas_second_tag_distance=0.3,
            stacked=True,
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
        bins = np.linspace(0, 10, 100)
        discrete_vals = [0, 5, 7, 9]

        hist_plot.add(
            Histogram(
                values=values1,
                bins=bins,
                discrete_vals=discrete_vals,
                norm=True,
            ),
            reference=True,
        )
        hist_plot.add(
            Histogram(
                values=values2,
                bins=bins,
                discrete_vals=discrete_vals,
                norm=True,
                linestyle="--",
            )
        )
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
            figsize=(6.5, 5),
            atlas_second_tag=(
                "Test if ratio is 1 for whole range if reference histogram is empty\n(+"
                " normalised)"
            ),
            n_ratio_panels=1,
        )
        hist_plot.add(
            Histogram(
                values=np.array([]),
                bins=40,
                bins_range=(-4, 4),
                label="empty histogram",
                norm=True,
            ),
            reference=True,
        )
        hist_plot.add(
            Histogram(
                values=self.hist_1_values,
                bins=40,
                bins_range=(-4, 4),
                label=f"N={self.n_random:_}",
                norm=True,
            )
        )
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
            figsize=(6.5, 5),
            atlas_second_tag=("Test if ratio is 1 for whole range if reference histogram is empty"),
            n_ratio_panels=1,
        )
        hist_plot.add(
            Histogram(
                values=np.array([]),
                bins=40,
                bins_range=(-4, 4),
                label="empty histogram",
                norm=False,
            ),
            reference=True,
        )
        hist_plot.add(
            Histogram(
                values=self.hist_1_values,
                bins=40,
                bins_range=(-4, 4),
                label=f"N={self.n_random:_}",
                norm=False,
            )
        )
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
            Histogram(
                values=arr_1,
                bins=40,
                bins_range=(-2, 2),
                label="uniform [-2, 0] and uniform [0.5, 1] \n(reference)",
            ),
            reference=True,
        )
        hist_plot.add(
            Histogram(
                values=arr_2,
                bins=40,
                bins_range=(-2, 2),
                label="uniform [0, 2]",
            )
        )
        hist_plot.add(
            Histogram(
                values=arr_3,
                bins=40,
                bins_range=(-2, 2),
                label="uniform [-1, 1]",
            )
        )
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
            atlas_brand="",
            atlas_first_tag="Simulation, $\\sqrt{s}=13$ TeV",
            atlas_tag_outside=True,
            figsize=(5, 4),
            ylabel="Number of jets",
            n_ratio_panels=1,
        )
        hist_plot.add(
            Histogram(
                values=self.hist_1_values,
                bins=20,
                bins_range=(0, 4),
                label=f"N={self.n_random:_}",
                norm=True,
            ),
            reference=True,
        )
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
            atlas_brand="",
            atlas_first_tag="Simulation, $\\sqrt{s}=13$ TeV",
            atlas_second_tag="",
            figsize=(5, 4),
            ylabel="Number of jets",
            n_ratio_panels=1,
        )
        hist_plot.add(
            Histogram(
                values=self.hist_1_values,
                bins=20,
                bins_range=(0, 4),
                label=f"N={self.n_random:_}",
                norm=True,
            ),
            reference=True,
        )
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
            atlas_brand="",
            atlas_first_tag="Simulation, $\\sqrt{s}=13$ TeV",
            atlas_second_tag="",
            figsize=(5, 4),
            ylabel="Number of jets",
            n_ratio_panels=1,
        )
        hist_plot.add(
            Histogram(
                values=self.hist_1_values,
                bins=20,
                bins_range=(0, 4),
                label=f"N={self.n_random:_}",
                norm=True,
            ),
            reference=True,
        )
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
            atlas_brand="",
            atlas_first_tag="Simulation, $\\sqrt{s}=13$ TeV",
            atlas_second_tag="",
            figsize=(5, 4),
            ylabel="Number of jets",
            n_ratio_panels=1,
        )
        hist_plot.add(
            Histogram(
                values=self.hist_1_values,
                bins=20,
                bins_range=(0, 4),
                label=f"N={self.n_random:_}",
                norm=True,
            ),
            reference=True,
        )
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
                values=rng.normal(0, 1, size=10_000),
                bins=40,
                bins_range=(-4, 8),
                ratio_group="ratio group 1",
                colour=Flavours["ujets"].colour,
                label=Flavours["ujets"].label,
            ),
            reference=True,
        )
        hist_plot.add(
            Histogram(
                values=rng.normal(0.5, 1, size=10_000),
                bins=40,
                bins_range=(-4, 8),
                colour=Flavours["ujets"].colour,
                ratio_group="ratio group 1",
                linestyle="--",
            ),
        )
        hist_plot.add(
            Histogram(
                values=rng.normal(3, 1, size=10_000),
                bins=40,
                bins_range=(-4, 8),
                label=Flavours["bjets"].label,
                colour=Flavours["bjets"].colour,
                ratio_group="Ratio group 2",
            ),
            reference=True,
        )
        hist_plot.add(
            Histogram(
                rng.normal(3.5, 1, size=10_000),
                bins=40,
                bins_range=(-4, 8),
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
            Histogram(
                values=rng.normal(0, 1, size=10_000),
                bins=100,
                bins_range=(-4, 16),
                label="Unflavoured histogram",
            )
        )
        # Flavour, but also label (using the default flavour label + the specified one)
        hist_plot.add(
            Histogram(
                values=rng.normal(4, 1, size=10_000),
                bins=100,
                bins_range=(-4, 16),
                label="(flavoured, adding default flavour label '$b$-jets' to legend)",
                flavour="bjets",
            )
        )
        # Flavour + label (this time with suppressing the default flavour label)
        hist_plot.add(
            Histogram(
                values=rng.normal(8, 1, size=10_000),
                bins=100,
                bins_range=(-4, 16),
                label="Flavoured histogram (default flavour label suppressed)",
                flavour="bjets",
                add_flavour_label=False,
                linestyle="--",
            )
        )
        # Flavoured, but using custom colour
        hist_plot.add(
            Histogram(
                values=rng.normal(12, 1, size=10_000),
                bins=100,
                bins_range=(-4, 16),
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
        values = np.array([0, 1, 2, 2, 3])
        weights = np.array([1, -1, 3, -2, 1])
        hist_exp = np.array([1, -1, 2])
        unc_exp = np.sqrt(np.array([1, (-1) ** 2, 3**2 + (-2) ** 2 + 1]))

        hist_plot = HistogramPlot(
            figsize=(6, 5),
            atlas_first_tag=None,
            atlas_second_tag=(
                "Test plot for the behaviour of weights \n(both positive and negative)"
                f"\nExpected bin counts: {hist_exp}"
                f"\nExpected bin uncertainties: {unc_exp}"
            ),
        )
        hist_plot.add(Histogram(values=values, bins=3, weights=weights, norm=False))
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

        hist_plot = HistogramPlot()
        hist_plot.atlas_second_tag = "This plot does not have under/overflow bins"
        hist_plot.add(
            Histogram(
                values=vals,
                bins=5,
                bins_range=(0, 5),
                underoverflow=False,
                colour="b",
            )
        )
        hist_plot.add(
            Histogram(
                values=vals_with_inf,
                bins=5,
                bins_range=(0, 5),
                underoverflow=False,
                colour="r",
                linestyle="dotted",
            )
        )
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

        hist_plot = HistogramPlot()
        hist_plot.atlas_second_tag = "This plot has under/overflow bins"
        hist_plot.add(
            Histogram(
                vals,
                bins=5,
                bins_range=(0, 5),
                colour="b",
            )
        )
        hist_plot.add(
            Histogram(
                vals_with_inf,
                bins=5,
                bins_range=(0, 5),
                colour="r",
                linestyle="dotted",
            )
        )
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

        hist_filled = Histogram(bin_counts, bin_edges=bin_edges, underoverflow=False)
        hist_notfilled = Histogram(values=vals, bins=bin_edges)

        hist_plot = HistogramPlot()
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

        hist_plot = HistogramPlot()
        hist_plot.add(
            Histogram(
                values=bin_counts,
                bin_edges=bin_edges,
                underoverflow=False,
                sum_squared_weights=sum_squared_weights,
            )
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
