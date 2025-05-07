"""Unit test script for the functions in roc.py."""

from __future__ import annotations

import os
import shutil  # noqa: F401
import tempfile
import unittest

import numpy as np
from matplotlib.testing.compare import compare_images

from puma import Roc, RocPlot
from puma.utils.logging import logger, set_log_level

set_log_level(logger, "DEBUG")


class RocTestCase(unittest.TestCase):
    """Test class for the puma.roc functions."""

    def setUp(self):
        self.sig_eff = np.linspace(0.4, 1, 100)
        self.bkg_rej = np.exp(-self.sig_eff) * 10e3

    def test_roc_init(self):
        """Test roc init."""
        with self.assertRaises(ValueError):
            Roc(np.ones(4), np.ones(5))

    def test_ratio_same_object(self):
        """Test roc divide function."""
        roc_curve = Roc(self.sig_eff, self.bkg_rej)
        roc_curve_ref = Roc(self.sig_eff, self.bkg_rej)
        _, ratio, _ = roc_curve.divide(roc_curve_ref)

        np.testing.assert_array_almost_equal(ratio, np.ones(len(self.bkg_rej)))

    def test_ratio_factor_two(self):
        """Test roc divide function."""
        roc_curve = Roc(self.sig_eff, self.bkg_rej)
        roc_curve_ref = Roc(self.sig_eff, self.bkg_rej * 2)
        _, ratio, _ = roc_curve.divide(roc_curve_ref)

        np.testing.assert_array_almost_equal(ratio, 1 / 2 * np.ones(len(self.bkg_rej)))

    def test_ratio_different_sig_interval(self):
        """Test roc divide function."""
        sig_eff = np.linspace(0.4, 0.9, 6)
        sig_eff_ref = np.linspace(0.6, 1, 5)
        bkg_rej = np.exp(-sig_eff) * 10e3
        bkg_rej_ref = np.exp(-sig_eff_ref) * 10e3
        roc_curve = Roc(sig_eff, bkg_rej)
        roc_curve_ref = Roc(sig_eff_ref, bkg_rej_ref * 2)
        with self.assertRaises(ValueError) as ctx:
            _, _, _ = roc_curve.divide(roc_curve_ref)
        self.assertEqual(
            "Signal efficiencies of the two ROCs do not match.",
            str(ctx.exception),
        )

    def test_ratio_factor_two_inverse(self):
        """Test roc divide function."""
        roc_curve = Roc(self.sig_eff, self.bkg_rej)
        roc_curve_ref = Roc(self.sig_eff, self.bkg_rej * 2)
        _, ratio, _ = roc_curve.divide(roc_curve_ref, inverse=True)

        np.testing.assert_array_almost_equal(ratio, 2 * np.ones(len(self.bkg_rej)))

    def test_binomial_error_no_ntest(self):
        """Test roc binomial_error function."""
        roc_curve = Roc(self.sig_eff, self.bkg_rej)
        with self.assertRaises(ValueError):
            roc_curve.binomial_error()

    def test_binomial_error_only_zeros(self):
        """Test roc binomial_error function."""
        roc_curve = Roc(self.sig_eff, np.zeros(len(self.sig_eff)), n_test=10e5)
        np.testing.assert_array_almost_equal(roc_curve.binomial_error(), [])

    def test_binomial_error_example(self):
        """Test roc binomial_error function."""
        error_rej = np.array([8.717798, 35.0, 99.498744])
        roc_curve = Roc(np.array([0.1, 0.2, 0.3]), np.array([20, 50, 100]), n_test=100)
        np.testing.assert_array_almost_equal(roc_curve.binomial_error(), error_rej)

    def test_binomial_error_example_norm(self):
        """Test roc binomial_error function."""
        error_rej = np.array([8.717798, 35.0, 99.498744]) / np.array([20, 50, 100])
        roc_curve = Roc(np.array([0.1, 0.2, 0.3]), np.array([20, 50, 100]), n_test=100)
        np.testing.assert_array_almost_equal(roc_curve.binomial_error(norm=True), error_rej)

    def test_binomial_error_example_pass_ntest(self):
        """Test roc binomial_error function."""
        error_rej = np.array([8.717798, 35.0, 99.498744])
        roc_curve = Roc(np.array([0.1, 0.2, 0.3]), np.array([20, 50, 100]))
        np.testing.assert_array_almost_equal(roc_curve.binomial_error(n_test=100), error_rej)


class RocMaskTestCase(unittest.TestCase):
    """Test class for the puma.roc non_zero_mask function."""

    def setUp(self):
        self.sig_eff = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        self.bkg_rej = np.array([0, 0.2, 0, 0.4, 0.5, 0, 0.7])

    def test_non_zero_mask(self):
        """Test roc non_zero_mask function."""
        roc_curve = Roc(self.sig_eff, self.bkg_rej)
        np.testing.assert_array_almost_equal(
            roc_curve.non_zero_mask, [False, True, False, True, True, False, True]
        )

    def test_non_zero_mask_xmin(self):
        """Test roc non_zero_mask function."""
        roc_curve = Roc(self.sig_eff, self.bkg_rej, xmin=0.4)
        np.testing.assert_array_almost_equal(
            roc_curve.non_zero_mask, [False, False, False, True, True, False, True]
        )

    def test_non_zero_mask_xmax(self):
        """Test roc non_zero_mask function."""
        roc_curve = Roc(self.sig_eff, self.bkg_rej, xmax=0.6)
        np.testing.assert_array_almost_equal(
            roc_curve.non_zero_mask, [False, True, False, True, True, False, False]
        )

    def test_non_zero_mask_xmin_xmax(self):
        """Test roc non_zero_mask function."""
        roc_curve = Roc(self.sig_eff, self.bkg_rej, xmax=0.6, xmin=0.4)
        np.testing.assert_array_almost_equal(
            roc_curve.non_zero_mask, [False, False, False, True, True, False, False]
        )

    def test_non_zero(self):
        """Test roc non_zero function."""
        roc_curve = Roc(self.sig_eff, self.bkg_rej)
        result_bkg_rej = self.bkg_rej[[False, True, False, True, True, False, True]]
        result_sig_eff = self.sig_eff[[False, True, False, True, True, False, True]]
        np.testing.assert_array_almost_equal(roc_curve.non_zero, (result_bkg_rej, result_sig_eff))

    def test_non_zero_xmin(self):
        """Test roc non_zero function."""
        roc_curve = Roc(self.sig_eff, self.bkg_rej, xmin=0.4)
        result_bkg_rej = self.bkg_rej[[False, False, False, True, True, False, True]]
        result_sig_eff = self.sig_eff[[False, False, False, True, True, False, True]]
        np.testing.assert_array_almost_equal(roc_curve.non_zero, (result_bkg_rej, result_sig_eff))

    def test_non_zero_xmax(self):
        """Test roc non_zero function."""
        roc_curve = Roc(self.sig_eff, self.bkg_rej, xmax=0.6)
        result_bkg_rej = self.bkg_rej[[False, True, False, True, True, False, False]]
        result_sig_eff = self.sig_eff[[False, True, False, True, True, False, False]]
        np.testing.assert_array_almost_equal(roc_curve.non_zero, (result_bkg_rej, result_sig_eff))

    def test_non_zero_xmin_xmax(self):
        """Test roc non_zero function."""
        roc_curve = Roc(self.sig_eff, self.bkg_rej, xmax=0.6, xmin=0.4)
        result_bkg_rej = self.bkg_rej[[False, False, False, True, True, False, False]]
        result_sig_eff = self.sig_eff[[False, False, False, True, True, False, False]]
        np.testing.assert_array_almost_equal(roc_curve.non_zero, (result_bkg_rej, result_sig_eff))


class RocOutputTestCase(unittest.TestCase):
    """Test class for the puma.roc_plot function."""

    def setUp(self):
        # Set up temp directory for comparison plots
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        self.expected_plots_dir = os.path.join(os.path.dirname(__file__), "expected_plots")

        self.sig_eff = np.array([0.5, 0.56, 0.63, 0.76, 0.83, 0.84, 0.85, 0.88, 0.9, 0.93, 1])
        self.u_rej_1 = np.array([1542, 918, 426, 78, 31, 26, 22, 12, 8, 4, 1])
        self.u_rej_2 = np.array([3061, 1642, 709, 112, 41, 34, 28, 15, 9, 4, 1])
        self.c_rej_1 = np.array([26, 17, 10, 4.1, 2.9, 2.7, 2.6, 2.2, 1.9, 1.6, 1])
        self.c_rej_2 = np.array([45, 27, 14, 4, 3, 2.9, 2.7, 2.2, 2.0, 1.63, 1])

        # sig efficiency with different start
        self.sig_eff_short = np.linspace(0.5, 1, 100)

    def test_output_two_curves_no_ratio(self):
        """Test with two curves of same flavour, without ratio panel."""
        plot = RocPlot(
            n_ratio_panels=0,
            ylabel="Light-jet rejection",
            xlabel="$b$-jet efficiency",
            atlas_second_tag=(
                "$\\sqrt{s}=13$ TeV, PFlow Jets\n$t\\bar{t}$ dummy sample," " $f_{c}=0.018$"
            ),
            y_scale=1.5,
        )

        # Add two roc curves
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.u_rej_1,
                rej_class="ujets",
                label="reference",
            ),
            reference=True,
        )
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.u_rej_2,
                rej_class="ujets",
                label="test",
            )
        )

        # Draw the figure
        plot.draw()

        name = "test_roc_two_curves_no_ratio.png"
        plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copyfile(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=2.5,
            )
        )

    def test_output_two_curves_one_ratio(self):
        """Test with two curves of same flavour, one ratio panel."""
        plot = RocPlot(
            n_ratio_panels=1,
            ylabel="Light-jet rejection",
            xlabel="$b$-jet efficiency",
            atlas_second_tag=(
                "$\\sqrt{s}=13$ TeV, PFlow Jets\n$t\\bar{t}$ dummy sample," " $f_{c}=0.018$"
            ),
            y_scale=1.5,
            # logy=False,
        )

        # Add two roc curves
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.u_rej_1,
                rej_class="ujets",
                label="reference",
            ),
            reference=True,
        )
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.u_rej_2,
                rej_class="ujets",
                label="test",
            )
        )

        plot.set_ratio_class(1, "ujets")

        # Draw the figure
        plot.draw()

        name = "test_roc_two_curves_1_ratio.png"
        plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copyfile(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=2.5,
            )
        )

    def test_output_two_curves_one_ratio_uncertainties(self):
        """Test with two curves of same flavour, one ratio panel."""
        plot = RocPlot(
            n_ratio_panels=1,
            ylabel="Light-jet rejection",
            xlabel="$b$-jet efficiency",
            atlas_second_tag=(
                "$\\sqrt{s}=13$ TeV, PFlow Jets\n$t\\bar{t}$ dummy sample," " $f_{c}=0.018$"
            ),
            y_scale=1.5,
            # logy=False,
        )

        # Add two roc curves
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.u_rej_1 * 2,
                rej_class="ujets",
                label="reference",
                n_test=1_000,
            ),
            reference=True,
        )
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.u_rej_2 * 2,
                rej_class="ujets",
                label="test",
                n_test=1_000,
            )
        )

        plot.set_ratio_class(1, "ujets")

        # Draw the figure
        plot.draw()

        name = "test_roc_two_curves_1_ratio_unc.png"
        plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copyfile(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=2.5,
            )
        )

    def test_output_four_curves_two_ratio(self):
        """Test with two curves for each flavour, two ratio panels."""
        plot = RocPlot(
            n_ratio_panels=2,
            ylabel="Background rejection",
            xlabel="$b$-jet efficiency",
            atlas_second_tag=(
                "$\\sqrt{s}=13$ TeV, PFlow Jets\n$t\\bar{t}$ dummy sample," " $f_{c}=0.018$"
            ),
        )

        # Add four roc curves
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.u_rej_1,
                rej_class="ujets",
                label="reference",
            ),
            reference=True,
        )
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.u_rej_2,
                rej_class="ujets",
                label="test",
            )
        )
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.c_rej_1,
                rej_class="cjets",
                label="reference",
            ),
            reference=True,
        )
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.c_rej_2,
                rej_class="cjets",
                label="test",
            )
        )

        plot.set_ratio_class(1, "ujets")
        plot.set_ratio_class(2, "cjets")

        # Draw the figure
        plot.draw()

        name = "test_roc_four_curves_2_ratio.png"
        plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copyfile(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=2.5,
            )
        )

    def test_output_ratio_legend_four_curves_two_ratio(self):
        """Test with two curves for each flavour, two ratio panels, and ratio legend."""
        plot = RocPlot(
            n_ratio_panels=2,
            ylabel="Background rejection",
            xlabel="$b$-jet efficiency",
            atlas_second_tag=(
                "$\\sqrt{s}=13$ TeV, PFlow Jets\n$t\\bar{t}$ dummy sample," " $f_{c}=0.018$"
            ),
        )

        # Add four roc curves
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.u_rej_1,
                rej_class="ujets",
                label="reference",
            ),
            reference=True,
        )
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.u_rej_2,
                rej_class="ujets",
                label="test",
            )
        )
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.c_rej_1,
                rej_class="cjets",
                label="reference",
            ),
            reference=True,
        )
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.c_rej_2,
                rej_class="cjets",
                label="test",
            )
        )

        plot.set_ratio_class(1, "ujets")
        plot.set_ratio_class(2, "cjets")

        # Draw the figure
        plot.draw()

        name = "test_output_ratio_legend_four_curves_two_ratio.png"
        plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copyfile(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=2.5,
            )
        )

    def test_output_four_curves_two_ratio_uncertainties(self):
        """Test with two curves for each flavour, two ratio panels and binom. unc."""
        plot = RocPlot(
            n_ratio_panels=2,
            ylabel="Background rejection",
            xlabel="$b$-jet efficiency",
            atlas_second_tag=(
                "$\\sqrt{s}=13$ TeV, PFlow Jets\n$t\\bar{t}$ dummy sample," " $f_{c}=0.018$"
            ),
        )

        # Add four roc curves
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.u_rej_1,
                rej_class="ujets",
                label="reference",
                n_test=1_000,
            ),
            reference=True,
        )
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.u_rej_2,
                rej_class="ujets",
                label="test",
                n_test=1_000,
            )
        )
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.c_rej_1,
                rej_class="cjets",
                label="reference",
                n_test=1_000,
            ),
            reference=True,
        )
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.c_rej_2,
                rej_class="cjets",
                label="test",
                n_test=1_000,
            )
        )

        plot.set_ratio_class(1, "ujets")
        plot.set_ratio_class(2, "cjets")

        # Draw the figure
        plot.draw()

        name = "test_roc_four_curves_2_ratio_unc.png"
        plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copyfile(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=2.5,
            )
        )

    def test_output_ratio_labelpad(self):
        """Test for labelpad support."""
        plot = RocPlot(
            n_ratio_panels=1,
            ylabel="Light-jet rejection",
            xlabel="$b$-jet efficiency",
            atlas_second_tag=(
                "$\\sqrt{s}=13$ TeV, PFlow Jets\n$t\\bar{t}$ dummy sample," " $f_{c}=0.018$"
            ),
            y_scale=1.5,
            # logy=False,
        )

        # Add two roc curves
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.u_rej_1,
                rej_class="ujets",
                label="reference",
            ),
            reference=True,
        )
        plot.add_roc(
            Roc(
                self.sig_eff,
                self.u_rej_2,
                rej_class="ujets",
                label="test",
            )
        )

        plot.set_ratio_class(1, "ujets")

        # Draw the figure
        plot.draw(labelpad=20)

        name = "test_roc_ratio_labelpad.png"
        plot.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copyfile(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertIsNone(
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=2.5,
            )
        )
