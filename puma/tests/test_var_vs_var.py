"""Unit test script for the functions in var_vs_var.py."""

from __future__ import annotations

import json
import os
import shutil  # noqa: F401
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from matplotlib.testing.compare import compare_images

from puma import VarVsVar, VarVsVarPlot
from puma.utils.logging import logger, set_log_level

set_log_level(logger, "DEBUG")


class VarVsVarTestCase(unittest.TestCase):
    """Test class for the puma.var_vs_var functions."""

    def setUp(self):
        self.x_var = np.linspace(100, 250, 20)
        self.y_var_mean = np.exp(-np.linspace(6, 10, 20)) * 10e3
        self.y_var_std = np.sin(self.y_var_mean)

    def test_var_vs_var_init_wrong_mean_shape(self):
        """Test var_vs_var init."""
        with self.assertRaises(ValueError):
            VarVsVar(np.ones(4), np.ones(5), np.ones(5))

    def test_var_vs_var_init_wrong_y_mean_std_shape(self):
        """Test var_vs_var init."""
        with self.assertRaises(ValueError):
            VarVsVar(np.ones(4), np.ones(4), np.ones(5))

    def test_var_vs_var_init_wrong_x_mean_widths_shape(self):
        """Test var_vs_var init."""
        with self.assertRaises(ValueError):
            VarVsVar(np.ones(4), np.ones(4), np.ones(4), x_var_widths=np.ones(5))

    def test_var_vs_var_init(self):
        """Test var_vs_var init."""
        VarVsVar(
            np.ones(6),
            np.ones(6),
            np.ones(6),
            x_var_widths=np.ones(6),
            key="test",
            fill=True,
            plot_y_std=False,
        )

    def test_var_vs_var_eq(self):
        """Test var_vs_var eq."""
        var_plot = VarVsVar(
            np.ones(6),
            np.ones(6),
            np.ones(6),
            x_var_widths=np.ones(6),
            key="test",
            fill=True,
            plot_y_std=False,
        )
        self.assertEqual(var_plot, var_plot)

    def test_var_vs_var_eq_different_classes(self):
        """Test var_vs_var eq."""
        var_plot = VarVsVar(
            np.ones(6),
            np.ones(6),
            np.ones(6),
            x_var_widths=np.ones(6),
            key="test",
            fill=True,
            plot_y_std=False,
        )
        self.assertNotEqual(var_plot, np.ones(6))

    def test_var_vs_var_divide_same(self):
        """Test var_vs_var divide."""
        var_plot = VarVsVar(
            x_var=self.x_var,
            y_var_mean=self.y_var_mean,
            y_var_std=self.y_var_std,
        )
        np.testing.assert_array_almost_equal(var_plot.divide(var_plot)[0], np.ones(20))

    def test_var_vs_var_divide_different_shapes(self):
        """Test var_vs_eff divide."""
        var_plot = VarVsVar(
            x_var=self.x_var,
            y_var_mean=self.y_var_mean,
            y_var_std=self.y_var_std,
        )
        var_plot_comp = VarVsVar(
            x_var=np.repeat(self.x_var, 2),
            y_var_mean=np.repeat(self.y_var_mean, 2),
            y_var_std=np.repeat(self.y_var_std, 2),
        )
        with self.assertRaises(ValueError):
            var_plot.divide(var_plot_comp)


class VarVsVarIOTestCase(unittest.TestCase):
    """
    Ensure that VarVsVar serialisation (`save` / `load`) is loss-less and that
    the private encode/decode helpers correctly tag / restore exotic types.
    """

    # ------------------------------------------------------------------
    # helper to build a deterministic VarVsVar
    # ------------------------------------------------------------------
    @staticmethod
    def _make_curve() -> VarVsVar:
        return VarVsVar(
            x_var=np.array([0.0, 1.0, 2.0, 3.0]),
            y_var_mean=np.array([10.0, 20.0, 30.0, 40.0]),
            y_var_std=np.array([1.0, 2.0, 3.0, 4.0]),
            x_var_widths=np.array([0.5, 0.5, 0.5, 0.5]),
            key="unit-test",
            colour=(0.2, 0.4, 0.6),  # forwarded to PlotLineObject
            linestyle=(0, (3, 1)),
            label="curve",
        )

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------
    def testencodedecode_roundtrip(self):
        """VarVsVar.encode/decode must round-trip every field exactly."""
        v = self._make_curve()

        encoded = v.encode(v.args_to_store)

        # ── encoded structure is JSON-safe ───────────────────────────────
        json.dumps(encoded)  # must not raise
        for field in ("x_var", "y_var_mean", "y_var_std", "x_var_widths"):
            # numpy arrays became {"__ndarray__": …}
            self.assertIn("__ndarray__", encoded[field])

        # tuples are tagged
        self.assertEqual(encoded["colour"]["__tuple__"], [0.2, 0.4, 0.6])

        # ── decoding restores original Python objects ────────────────────
        decoded = VarVsVar.decode(encoded)
        np.testing.assert_array_equal(decoded["y_var_std"], v.y_var_std)
        self.assertEqual(decoded["colour"], v.colour)  # tuple restored
        self.assertTrue(decoded["fill"])

    # ------------------------------------------------------------------
    # save / load (JSON)
    # ------------------------------------------------------------------
    def test_json_roundtrip(self):
        """Saving to JSON and reloading must reproduce the object 1-to-1."""
        v = self._make_curve()

        with tempfile.TemporaryDirectory() as tmpd:
            path = Path(tmpd) / "curve.json"
            v.save(path)

            # file parses as valid JSON
            with path.open() as f:
                raw = json.load(f)
            self.assertIn("y_var_mean", raw)

            clone = VarVsVar.load(path)

            # numerical arrays survive bit-perfect
            np.testing.assert_array_equal(v.x_var, clone.x_var)
            np.testing.assert_array_equal(v.y_var_mean, clone.y_var_mean)
            np.testing.assert_array_equal(v.y_var_std, clone.y_var_std)
            np.testing.assert_array_equal(v.x_var_widths, clone.x_var_widths)

            # metadata
            self.assertEqual(clone.key, v.key)
            self.assertEqual(clone.colour, v.colour)
            self.assertEqual(clone.linestyle, v.linestyle)
            self.assertEqual(clone.label, v.label)

            # functional behaviour intact
            ratio, _ = v.divide(clone)
            np.testing.assert_allclose(ratio, np.ones_like(ratio))

    # ------------------------------------------------------------------
    # save / load (YAML)
    # ------------------------------------------------------------------
    def test_yaml_roundtrip(self):
        """YAML round-trip preserves arrays & metadata."""
        v = self._make_curve()

        with tempfile.TemporaryDirectory() as tmpd:
            path = Path(tmpd) / "curve.yaml"
            v.save(path)

            with path.open() as f:
                raw = yaml.safe_load(f)
            self.assertIn("x_var", raw)

            clone = VarVsVar.load(path)
            self.assertEqual(clone.label, v.label)

    # ------------------------------------------------------------------
    # load-time overrides
    # ------------------------------------------------------------------
    def test_load_override(self):
        """`extra_kwargs` given to `load` must override stored attributes."""
        v = self._make_curve()

        with tempfile.TemporaryDirectory() as tmpd:
            path = Path(tmpd) / "curve.json"
            v.save(path)
            clone = VarVsVar.load(path, colour="magenta", label="override")

            self.assertEqual(clone.colour, "magenta")
            self.assertEqual(clone.label, "override")

    # ------------------------------------------------------------------
    # unknown file extensions
    # ------------------------------------------------------------------
    def test_invalid_extensions(self):
        """Unsupported suffixes must raise ValueError."""
        v = self._make_curve()

        with tempfile.TemporaryDirectory() as tmpd:
            # save
            with self.assertRaises(ValueError):
                v.save(Path(tmpd) / "curve.txt")

            # load
            bogus = Path(tmpd) / "curve.conf"
            bogus.write_text("dummy")
            with self.assertRaises(ValueError):
                VarVsVar.load(bogus)


class VarVsVarPlotTestCase(unittest.TestCase):
    """Test class for the puma.var_vs_var_plot."""

    def setUp(self):
        # Set up temp directory for comparison plots
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint:disable=R1732
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        self.expected_plots_dir = os.path.join(os.path.dirname(__file__), "expected_plots")
        np.random.seed(42)
        n_random = 21

        # background (same for both taggers)
        self.x_var = np.linspace(0, 250, num=n_random)
        self.y_var_mean = np.exp(-self.x_var / 200) * 10
        self.y_var_std = np.abs(np.sin(self.y_var_mean))
        self.x_var_widths = np.ones_like(self.x_var) * 5

        self.y_var_mean_2 = np.exp(-self.x_var / 100) * 10
        self.y_var_std_2 = np.abs(np.sin(self.y_var_mean_2))

        self.test = VarVsVar(
            x_var=self.x_var,
            y_var_mean=self.y_var_mean,
            y_var_std=self.y_var_std,
            label=r"$10e^{-x/200}$",
            fill=False,
            is_marker=True,
        )
        self.test_2 = VarVsVar(
            x_var=self.x_var,
            y_var_mean=self.y_var_mean_2,
            y_var_std=self.y_var_std_2,
            x_var_widths=self.x_var_widths,
            label=r"$10e^{-x/100}$",
            fill=True,
            plot_y_std=False,
            is_marker=False,
        )

    def test_n_ratio_panels(self):
        """Check if ValueError is raised when we require more than 1 ratio panel."""
        with self.assertRaises(ValueError):
            VarVsVarPlot(
                n_ratio_panels=np.random.randint(2, 10),
            )

    def test_no_reference(self):
        """Check if ValueError is raised when plot ratios without reference."""
        test_plot = VarVsVarPlot(
            n_ratio_panels=1,
        )
        test_plot.add(self.test)
        test_plot.add(self.test_2)
        with self.assertRaises(ValueError):
            test_plot.plot_ratios()

    def test_overwrite_reference(self):
        """Check correct reference overwrite."""
        test_plot = VarVsVarPlot(
            n_ratio_panels=1,
        )
        test_plot.add(self.test, reference=True)
        test_plot.add(self.test_2, reference=True)

    def test_same_keys(self):
        """Check if KeyError is rased when we add VarVsVar object with existing key."""
        test_plot = VarVsVarPlot(
            n_ratio_panels=1,
        )
        test_plot.add(self.test, key="1")
        with self.assertRaises(KeyError):
            test_plot.add(self.test_2, key="1")

    def test_output_plot(self):
        """Test output plot."""
        # define the curves

        test_plot = VarVsVarPlot(
            ylabel=r"$\overline{N}_{trk}$",
            xlabel=r"$p_{T}$ [GeV]",
            grid=True,
            logy=False,
            atlas_second_tag="Unit test plot based on exponential decay.",
            n_ratio_panels=1,
            figsize=(9, 6),
        )
        test_plot.add(self.test, reference=True)
        test_plot.add(self.test_2, reference=False)

        test_plot.draw_hline(4)
        test_plot.draw()

        name = "test_var_vs_var.png"
        test_plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertEqual(
            None,
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=5,
            ),
        )

    def test_get_reference_name_multiple_references(self):
        """Testing get_reference_name ValueError for multiple references."""
        test_plot = VarVsVarPlot(
            ylabel=r"$\overline{N}_{trk}$",
            xlabel=r"$p_{T}$ [GeV]",
            grid=True,
            logy=False,
            atlas_second_tag="Unit test plot based on exponential decay.",
            n_ratio_panels=1,
            figsize=(9, 6),
        )
        test_plot.add(
            VarVsVar(
                x_var=self.x_var,
                y_var_mean=self.y_var_mean,
                y_var_std=self.y_var_std,
                label=r"$10e^{-x/200}$",
                fill=False,
                is_marker=True,
                ratio_group="test",
            ),
            reference=True,
        )
        test_plot.add(
            VarVsVar(
                x_var=self.x_var,
                y_var_mean=self.y_var_mean,
                y_var_std=self.y_var_std,
                label=r"$10e^{-x/200}$",
                fill=False,
                is_marker=True,
                ratio_group="test",
            ),
            reference=True,
        )
        test_plot.add(
            VarVsVar(
                x_var=self.x_var,
                y_var_mean=self.y_var_mean,
                y_var_std=self.y_var_std,
                label=r"$10e^{-x/200}$",
                fill=False,
                is_marker=True,
                ratio_group="test",
            ),
            reference=False,
        )

        with self.assertRaises(ValueError):
            test_plot.draw()
