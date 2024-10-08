"""Unit test script for the functions in var_vs_eff.py."""

from __future__ import annotations

import os
import shutil  # noqa: F401
import tempfile
import unittest

import numpy as np
from ftag import Flavours
from matplotlib.testing.compare import compare_images

from puma import VarVsEff, VarVsEffPlot
from puma.utils.logging import logger, set_log_level

set_log_level(logger, "DEBUG")


class VarVsEffTestCase(unittest.TestCase):
    """Test class for the puma.var_vs_eff functions."""

    def setUp(self):
        self.working_point = 0.77
        self.disc_sig = np.linspace(-6, +6, 100)
        self.x_var_sig = np.exp(-self.disc_sig) * 10e3
        self.disc_bkg = np.linspace(-5.5, +6.6, 120)
        self.x_var_bkg = np.exp(-self.disc_bkg * 0.8) * 10e3 + 30

    def test_var_vs_eff_init_wrong_sig_shape(self):
        """Test var_vs_eff init."""
        with self.assertRaises(ValueError):
            VarVsEff(np.ones(4), np.ones(5))

    def test_var_vs_eff_init_wrong_bkg_shape(self):
        """Test var_vs_eff init."""
        with self.assertRaises(ValueError):
            VarVsEff(np.ones(6), np.ones(6), np.ones(4), np.ones(5))

    def test_var_vs_eff_init_flat_eff_disc_cut(self):
        """Test var_vs_eff init."""
        with self.assertRaises(ValueError):
            VarVsEff(
                np.ones(6),
                np.ones(6),
                flat_per_bin=True,
                disc_cut=1.0,
                working_point=0.77,
            )

    def test_var_vs_eff_init_flat_eff_no_wp(self):
        """Test var_vs_eff init."""
        with self.assertRaises(ValueError):
            VarVsEff(np.ones(6), np.ones(6), flat_per_bin=True, disc_cut=1.0)

    def test_var_vs_eff_init_disc_cut_wp(self):
        """Test var_vs_eff init."""
        with self.assertRaises(ValueError):
            VarVsEff(np.ones(6), np.ones(6), disc_cut=1.0, working_point=0.77)

    def test_var_vs_eff_init_no_disc_cut_no_wp(self):
        """Test var_vs_eff init."""
        with self.assertRaises(ValueError):
            VarVsEff(np.ones(6), np.ones(6))

    def test_var_vs_eff_init_disc_cut_wrong_shape(self):
        """Test var_vs_eff init."""
        with self.assertRaises(ValueError):
            VarVsEff(np.ones(6), np.ones(6), disc_cut=[1.0, 2.0])

    def test_var_vs_eff_set_bin_edges_list(self):
        """Test var_vs_eff _set_bin_edges."""
        var_plot = VarVsEff(
            x_var_sig=[0, 1, 2], disc_sig=[3, 4, 5], bins=[0, 1, 2], working_point=0.7
        )
        np.testing.assert_array_almost_equal(var_plot.bin_edges, [0, 1, 2])

    def test_var_vs_eff_set_bin_edges_only_signal(self):
        """Test var_vs_eff _set_bin_edges."""
        var_plot = VarVsEff(x_var_sig=[0, 1, 2], disc_sig=[3, 4, 5], bins=2, working_point=0.7)
        np.testing.assert_array_almost_equal(var_plot.bin_edges, [0, 1, 2], decimal=4)

    def test_var_vs_eff_set_bin_edges(self):
        """Test var_vs_eff _set_bin_edges."""
        var_plot = VarVsEff(
            x_var_sig=[0, 1, 2],
            disc_sig=[3, 4, 5],
            x_var_bkg=[-1, 1, 3],
            disc_bkg=[3, 4, 5],
            bins=2,
            working_point=0.7,
        )
        np.testing.assert_array_almost_equal(var_plot.bin_edges, [-1, 1, 3], decimal=4)

    def test_var_vs_eff_flat_eff_sig_eff(self):
        """Test var_vs_eff sig_eff."""
        n_bins = 4
        var_plot = VarVsEff(
            x_var_sig=self.disc_sig,
            disc_sig=self.x_var_sig,
            x_var_bkg=self.disc_bkg,
            disc_bkg=self.x_var_bkg,
            working_point=self.working_point,
            flat_per_bin=True,
            bins=n_bins,
        )
        np.testing.assert_array_almost_equal(
            var_plot.sig_eff[0], [self.working_point] * n_bins, decimal=2
        )

    def test_var_vs_eff_flat_eff_sig_rej(self):
        """Test var_vs_eff sig_rej."""
        n_bins = 4
        var_plot = VarVsEff(
            x_var_sig=self.disc_sig,
            disc_sig=self.x_var_sig,
            x_var_bkg=self.disc_bkg,
            disc_bkg=self.x_var_bkg,
            working_point=self.working_point,
            flat_per_bin=True,
            bins=n_bins,
        )
        np.testing.assert_array_almost_equal(
            var_plot.sig_rej[0], [1 / self.working_point] * n_bins, decimal=2
        )

    def test_var_vs_eff_one_bin(self):
        """Test var_vs_eff."""
        n_bins = 1
        var_plot = VarVsEff(
            x_var_sig=self.disc_sig,
            disc_sig=self.x_var_sig,
            x_var_bkg=self.disc_bkg,
            disc_bkg=self.x_var_bkg,
            working_point=self.working_point,
            flat_per_bin=True,
            bins=n_bins,
        )
        var_plot_comp = VarVsEff(
            x_var_sig=self.disc_sig,
            disc_sig=self.x_var_sig,
            x_var_bkg=self.disc_bkg,
            disc_bkg=self.x_var_bkg,
            working_point=self.working_point,
            bins=n_bins,
        )
        var_plot_list = VarVsEff(
            x_var_sig=self.disc_sig,
            disc_sig=self.x_var_sig,
            x_var_bkg=self.disc_bkg,
            disc_bkg=self.x_var_bkg,
            disc_cut=var_plot.disc_cut,
            bins=n_bins,
        )
        with self.subTest("Comparison var_plot_comp"):
            np.testing.assert_array_almost_equal(var_plot.sig_eff, var_plot_comp.sig_eff)
        with self.subTest("Comparison var_plot_list"):
            np.testing.assert_array_almost_equal(var_plot.sig_eff, var_plot_list.sig_eff)

    def test_var_vs_eff_divide_same(self):
        """Test var_vs_eff divide."""
        n_bins = 1
        var_plot = VarVsEff(
            x_var_sig=self.disc_sig,
            disc_sig=self.x_var_sig,
            x_var_bkg=self.disc_bkg,
            disc_bkg=self.x_var_bkg,
            working_point=self.working_point,
            flat_per_bin=True,
            bins=n_bins,
        )
        var_plot.y_var_mean, var_plot.y_var_std = var_plot.get("sig_eff")
        np.testing.assert_array_almost_equal(var_plot.divide(var_plot)[0], np.ones(1))

    def test_var_vs_eff_wrong_mode(self):
        """Test var_vs_eff wrong mode."""
        n_bins = 1
        var_plot = VarVsEff(
            x_var_sig=self.disc_sig,
            disc_sig=self.x_var_sig,
            x_var_bkg=self.disc_bkg,
            disc_bkg=self.x_var_bkg,
            working_point=self.working_point,
            flat_per_bin=True,
            bins=n_bins,
        )
        with self.assertRaises(ValueError):
            var_plot.y_var_mean, var_plot.y_var_std = var_plot.get("test")

    def test_var_vs_eff_divide_different_binning(self):
        """Test var_vs_eff divide."""
        var_plot = VarVsEff(
            x_var_sig=self.disc_sig,
            disc_sig=self.x_var_sig,
            x_var_bkg=self.disc_bkg,
            disc_bkg=self.x_var_bkg,
            working_point=self.working_point,
            flat_per_bin=True,
            bins=1,
        )
        var_plot_comp = VarVsEff(
            x_var_sig=self.disc_sig,
            disc_sig=self.x_var_sig,
            x_var_bkg=self.disc_bkg,
            disc_bkg=self.x_var_bkg,
            working_point=self.working_point,
            flat_per_bin=True,
            bins=2,
        )
        with self.assertRaises(ValueError):
            var_plot.y_var_mean, var_plot.y_var_std = var_plot.get("sig_eff")
            var_plot_comp.y_var_mean, var_plot_comp.y_var_std = var_plot.get("sig_eff")
            var_plot.divide(var_plot_comp)

    def test_var_vs_eff_eq_different_classes(self):
        """Test var_vs_eff eq."""
        var_plot = VarVsEff(x_var_sig=[0, 1, 2], disc_sig=[3, 4, 5], bins=2, working_point=0.7)
        self.assertNotEqual(var_plot, np.ones(6))

    def test_var_vs_eff_get(self):
        var_plot = VarVsEff(
            x_var_sig=self.disc_sig,
            disc_sig=self.x_var_sig,
            x_var_bkg=self.disc_bkg,
            disc_bkg=self.x_var_bkg,
            working_point=self.working_point,
            flat_per_bin=True,
            bins=1,
        )
        mode_options = ["sig_eff", "bkg_eff", "sig_rej", "bkg_rej"]
        for mode in mode_options:
            var_plot.get(mode)


class VarVsEffOutputTestCase(unittest.TestCase):
    """Test class for the puma.var_vs_eff_plot output."""

    def setUp(self):
        # Set up temp directory for comparison plots
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint:disable=R1732
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        self.expected_plots_dir = os.path.join(os.path.dirname(__file__), "expected_plots")
        # Generate discriminant and pT distribution for sig and bkg for two taggers
        # We want that both taggers yield the same discriminant values for the bkg
        # jets, a gaussian located at 0.
        # For low pT, we want both taggers to perform similar (discriminant values
        # are a gaussian located at 1).
        # For high pT (here pT>110), the "new" tagger signal is better separated from
        # bkg (discriminant values are then a gaussian located at 2)
        np.random.seed(42)
        n_random = 10_000

        # background (same for both taggers)
        self.disc_bkg = np.random.normal(loc=0, size=2 * n_random)
        self.x_var_bkg = np.random.uniform(0, 250, size=2 * n_random)

        # reference tagger (constant separation power for whole pT range)
        self.disc_sig_1 = np.random.normal(loc=1, size=2 * n_random)
        self.x_var_sig_1 = np.random.uniform(0, 250, size=2 * n_random)

        # new tagger (better separation for pT > 110)
        self.disc_sig_2 = np.concatenate((
            np.random.normal(loc=1, size=n_random),
            np.random.normal(loc=3, size=n_random),
        ))
        self.x_var_sig_2 = np.concatenate((
            np.random.uniform(0, 110, size=n_random),
            np.random.uniform(110, 250, size=n_random),
        ))

        # Define pT bins
        self.bins = [20, 30, 40, 60, 85, 110, 140, 175, 250]

    def test_var_vs_eff_plot_mode_option(self):
        with self.assertRaises(ValueError):
            VarVsEffPlot(
                mode="test",
                ylabel="Background rejection",
                xlabel=r"$p_{T}$ [GeV]",
                logy=True,
                atlas_second_tag="test",
                y_scale=1.5,
                n_ratio_panels=1,
                figsize=(9, 6),
            )

    def test_output_plot_flat_per_bin_bkg_rejection(self):
        """Test output plot with flat eff per bin - bkg rejection."""
        # define the curves
        ref_light = VarVsEff(
            x_var_sig=self.x_var_sig_1,
            disc_sig=self.disc_sig_1,
            x_var_bkg=self.x_var_bkg,
            disc_bkg=self.disc_bkg,
            bins=self.bins,
            working_point=0.5,
            disc_cut=None,
            flat_per_bin=True,
            label="reference model",
        )
        better_light = VarVsEff(
            x_var_sig=self.x_var_sig_2,
            disc_sig=self.disc_sig_2,
            x_var_bkg=self.x_var_bkg,
            disc_bkg=self.disc_bkg,
            bins=self.bins,
            working_point=0.5,
            disc_cut=None,
            flat_per_bin=True,
            linestyle="dashed",
            label="better model (by construction better for $p_T$ > 110)",
        )
        plot_bkg_rej = VarVsEffPlot(
            mode="bkg_rej",
            ylabel="Background rejection",
            xlabel=r"$p_{T}$ [GeV]",
            logy=True,
            atlas_second_tag=(
                "Unit test plot based on gaussian distributions. \n"
                "'reference model' should have bkg rejection of ~ 6.1 for the whole"
                " whole $p_T$ range\n"
                "'better model' should have bkg efficiency of ~ 6.1 for $p_T < 110$\n"
                "'better model' should have quite large bkg efficiency $p_T > 110$\n"
            ),
            y_scale=1.5,
            n_ratio_panels=1,
            figsize=(9, 6),
        )
        plot_bkg_rej.add(ref_light, reference=True)
        plot_bkg_rej.add(better_light)

        plot_bkg_rej.draw()

        name = "test_pt_dependence_rejection.png"
        plot_bkg_rej.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertEqual(
            None,
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            ),
        )

    def test_output_plot_flat_eff_bin_bkg_efficiency(self):
        """Test output plot with flat eff per bin."""
        # define the curves
        ref_light = VarVsEff(
            x_var_sig=self.x_var_sig_1,
            disc_sig=self.disc_sig_1,
            x_var_bkg=self.x_var_bkg,
            disc_bkg=self.disc_bkg,
            bins=self.bins,
            working_point=0.5,
            disc_cut=None,
            flat_per_bin=True,
            label="reference model",
        )
        better_light = VarVsEff(
            x_var_sig=self.x_var_sig_2,
            disc_sig=self.disc_sig_2,
            x_var_bkg=self.x_var_bkg,
            disc_bkg=self.disc_bkg,
            bins=self.bins,
            working_point=0.5,
            disc_cut=None,
            flat_per_bin=True,
            label="better model (by construction better for $p_T$ > 110)",
        )
        plot_bkg_rej = VarVsEffPlot(
            mode="bkg_eff",
            ylabel="background efficiency",
            xlabel=r"$p_{T}$ [GeV]",
            logy=False,
            atlas_second_tag=(
                "Unit test plot based on gaussian distributions. \n"
                "'reference model' should have bkg efficiency of ~ 0.16 "
                " whole $p_T$ range\n"
                "'better model' should have bkg efficiency of ~ 0.16 for $p_T < 110$\n"
                "'better model' should have bkg efficiency of ~ 0 for $p_T > 110$\n"
            ),
            y_scale=1.5,
            n_ratio_panels=1,
            figsize=(9, 6),
        )
        plot_bkg_rej.add(ref_light, reference=True)
        plot_bkg_rej.add(better_light)
        for cut in ref_light.disc_cut:
            print(cut)

        plot_bkg_rej.draw()

        name = "test_pt_dependence_bkg_efficiency.png"
        plot_bkg_rej.savefig(f"{self.actual_plots_dir}/{name}")
        # Uncomment line below to update expected image
        # shutil.copy(f"{self.actual_plots_dir}/{name}", f"{self.expected_plots_dir}/{name}")
        self.assertEqual(
            None,
            compare_images(
                f"{self.actual_plots_dir}/{name}",
                f"{self.expected_plots_dir}/{name}",
                tol=1,
            ),
        )

    def test_var_vs_eff_info_str_fixed_eff(self):
        flat_wp_plot = VarVsEffPlot(
            mode="bkg_eff",
        )
        signal = Flavours["bjets"]

        flat_wp_plot.apply_modified_atlas_second_tag(signal=signal, working_point=0.7)

        expected_tag = "70.0% $b$-jet efficiency"
        self.assertEqual(flat_wp_plot.atlas_second_tag, expected_tag)

    def test_var_vs_eff_info_str_flat_wp(self):
        flat_wp_plot = VarVsEffPlot(
            mode="bkg_eff",
            atlas_second_tag="test",
        )
        signal = Flavours["bjets"]

        flat_wp_plot.apply_modified_atlas_second_tag(
            signal=signal, working_point=0.7, flat_per_bin=True
        )

        expected_tag = "test\nFlat 70.0% $b$-jet efficiency per bin"
        self.assertEqual(flat_wp_plot.atlas_second_tag, expected_tag)

    def test_var_vs_eff_info_str_fixed_disc(self):
        flat_wp_plot = VarVsEffPlot(
            mode="bkg_eff",
        )
        signal = Flavours["bjets"]

        flat_wp_plot.apply_modified_atlas_second_tag(signal=signal, disc_cut=3.0)

        expected_tag = "$D_{b}$ > 3.0"
        self.assertEqual(flat_wp_plot.atlas_second_tag, expected_tag)
