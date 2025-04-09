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
        """Prepare some common data to use in multiple tests."""
        # Perfectly overlapping random data for signal and background for demonstration
        np.random.seed(42)
        self.x_sig = np.random.normal(0, 1, 1000)
        self.disc_sig = np.random.rand(1000)

        self.x_bkg = np.random.normal(0, 1, 1000)
        self.disc_bkg = np.random.rand(1000)

    def test_init_success_with_working_point(self):
        """Test that VarVsEff initializes correctly when `working_point` is provided."""
        obj = VarVsEff(
            x_var_sig=self.x_sig,
            disc_sig=self.disc_sig,
            x_var_bkg=self.x_bkg,
            disc_bkg=self.disc_bkg,
            bins=10,
            working_point=0.8,
        )
        self.assertEqual(obj.n_bins, 10)
        self.assertIsNotNone(obj.disc_binned_sig)
        self.assertIsNotNone(obj.disc_binned_bkg)

    def test_init_success_with_disc_cut(self):
        """Test that VarVsEff initializes correctly when `disc_cut` is provided."""
        obj = VarVsEff(
            x_var_sig=self.x_sig,
            disc_sig=self.disc_sig,
            x_var_bkg=self.x_bkg,
            disc_bkg=self.disc_bkg,
            bins=5,
            disc_cut=0.5,
        )
        self.assertEqual(obj.n_bins, 5)
        # disc_cut should become an array of the same length as n_bins
        self.assertEqual(len(obj.disc_cut), 5)

    def test_init_raises_for_mismatched_lengths(self):
        """
        Test that VarVsEff raises ValueError if x_var_sig and disc_sig
        have different lengths.
        """
        with self.assertRaises(ValueError):
            VarVsEff(x_var_sig=self.x_sig[:500], disc_sig=self.disc_sig, disc_cut=0.5)

    def test_init_raises_for_bkg_mismatched_lengths(self):
        """
        Test that VarVsEff raises ValueError if x_var_bkg and disc_bkg
        have different lengths.
        """
        with self.assertRaises(ValueError):
            VarVsEff(
                x_var_sig=self.x_sig,
                disc_sig=self.disc_sig,
                x_var_bkg=self.x_bkg[:500],
                disc_bkg=self.disc_bkg,
                disc_cut=0.5,
            )

    def test_init_raises_for_no_cut_and_no_wp(self):
        """
        Test that VarVsEff raises ValueError if neither `working_point`
        nor `disc_cut` is specified.
        """
        with self.assertRaises(ValueError):
            VarVsEff(
                x_var_sig=self.x_sig,
                disc_sig=self.disc_sig,
                x_var_bkg=self.x_bkg,
                disc_bkg=self.disc_bkg,
            )

    def test_init_raises_for_flat_per_bin_with_disc_cut(self):
        """
        Test that VarVsEff raises ValueError if flat_per_bin=True and
        disc_cut is also provided.
        """
        with self.assertRaises(ValueError):
            VarVsEff(
                x_var_sig=self.x_sig,
                disc_sig=self.disc_sig,
                x_var_bkg=self.x_bkg,
                disc_bkg=self.disc_bkg,
                flat_per_bin=True,
                disc_cut=0.5,
            )

    def test_init_raises_for_flat_per_bin_without_wp(self):
        """
        Test that VarVsEff raises ValueError if flat_per_bin=True and
        no `working_point` is given.
        """
        with self.assertRaises(ValueError):
            VarVsEff(
                x_var_sig=self.x_sig,
                disc_sig=self.disc_sig,
                x_var_bkg=self.x_bkg,
                disc_bkg=self.disc_bkg,
                flat_per_bin=True,
            )

    def test_init_raises_for_flat_per_bin_with_non_float_wp(self):
        """
        Test that VarVsEff raises ValueError if flat_per_bin=True
        and working_point is not a float.
        """
        with self.assertRaises(ValueError):
            VarVsEff(
                x_var_sig=self.x_sig,
                disc_sig=self.disc_sig,
                x_var_bkg=self.x_bkg,
                disc_bkg=self.disc_bkg,
                flat_per_bin=True,
                working_point=[0.8, 0.9],
            )

    def test_init_raises_for_both_disc_cut_and_wp(self):
        """
        Test that VarVsEff raises ValueError if both disc_cut and
        working_point are provided.
        """
        with self.assertRaises(ValueError):
            VarVsEff(x_var_sig=self.x_sig, disc_sig=self.disc_sig, disc_cut=0.5, working_point=0.8)

    def test_init_raises_for_disc_cut_length_mismatch(self):
        """
        Test that VarVsEff raises ValueError if disc_cut is array-like
        but its length doesn't match the number of bins.
        """
        with self.assertRaises(ValueError):
            VarVsEff(
                x_var_sig=self.x_sig,
                disc_sig=self.disc_sig,
                bins=5,
                disc_cut=[0.1, 0.2, 0.3],  # length 3 not matching bins=5
            )

    def test_efficiency_and_rejection(self):
        """
        Test that we can retrieve the signal/background efficiency
        and rejection without errors.
        """
        obj = VarVsEff(
            x_var_sig=self.x_sig,
            disc_sig=self.disc_sig,
            x_var_bkg=self.x_bkg,
            disc_bkg=self.disc_bkg,
            bins=10,
            disc_cut=0.5,
        )
        # normal cut
        sig_eff, sig_eff_err = obj.sig_eff
        bkg_eff, bkg_eff_err = obj.bkg_eff
        sig_rej, sig_rej_err = obj.sig_rej
        bkg_rej, bkg_rej_err = obj.bkg_rej

        self.assertEqual(len(sig_eff), obj.n_bins)
        self.assertEqual(len(sig_eff_err), obj.n_bins)
        self.assertEqual(len(bkg_eff), obj.n_bins)
        self.assertEqual(len(bkg_eff_err), obj.n_bins)
        self.assertEqual(len(sig_rej), obj.n_bins)
        self.assertEqual(len(sig_rej_err), obj.n_bins)
        self.assertEqual(len(bkg_rej), obj.n_bins)
        self.assertEqual(len(bkg_rej_err), obj.n_bins)

    def test_inverse_cut(self):
        """Test the inverse_cut functionality in get()."""
        obj = VarVsEff(
            x_var_sig=self.x_sig,
            disc_sig=self.disc_sig,
            x_var_bkg=self.x_bkg,
            disc_bkg=self.disc_bkg,
            bins=5,
            disc_cut=0.5,
        )
        normal_sig_eff, _ = obj.get("sig_eff", inverse_cut=False)
        inverse_sig_eff, _ = obj.get("sig_eff", inverse_cut=True)
        normal_bkg_rej, _ = obj.get("bkg_rej", inverse_cut=False)
        inverse_bkg_rej, _ = obj.get("bkg_rej", inverse_cut=True)

        # For a random uniform disc in [0,1], the sum of normal and inverse
        # efficiency across the entire dataset should be close to 1.0 per bin.
        # We won't test exact equality, but at least a sanity check:
        for e1, e2 in zip(normal_sig_eff, inverse_sig_eff):
            self.assertTrue(np.isclose(e1 + e2, 1.0, atol=0.05))

        self.assertTrue(np.isclose(normal_bkg_rej[0] + inverse_bkg_rej[0], 4.426316, atol=0.05))
        self.assertTrue(np.isclose(normal_bkg_rej[1] + inverse_bkg_rej[1], 4.002809, atol=0.05))

    def test_bkg_eff_sig_err(self):
        """
        Test the bkg_eff_sig_err property (returns background efficiencies
        and signal efficiency errors).
        """
        obj = VarVsEff(
            x_var_sig=self.x_sig,
            disc_sig=self.disc_sig,
            x_var_bkg=self.x_bkg,
            disc_bkg=self.disc_bkg,
            bins=5,
            disc_cut=0.5,
        )
        eff, err = obj.bkg_eff_sig_err
        self.assertEqual(len(eff), obj.n_bins)
        self.assertEqual(len(err), obj.n_bins)

    def test_equality_operator(self):
        """
        Test that two VarVsEff objects are considered equal if
        they have the same data and parameters.
        """
        obj1 = VarVsEff(
            x_var_sig=self.x_sig, disc_sig=self.disc_sig, bins=5, disc_cut=0.5, key="tagger1"
        )
        obj2 = VarVsEff(
            x_var_sig=self.x_sig, disc_sig=self.disc_sig, bins=5, disc_cut=0.5, key="tagger1"
        )
        self.assertTrue(obj1 == obj2)

        # Modify one parameter in obj2
        obj2.working_point = 0.9
        self.assertFalse(obj1 == obj2)

    def test_all_signal_in_underflow_bin_triggers_logger(self):
        """
        Test forces all signal x-values to be below the min bin edge,
        so bin_indices_sig == 0 for all entries. This triggers the logger.error call
        in _apply_binning.
        """
        x_var_sig = np.array([-10, -9, -8])  # all below the chosen bin range
        disc_sig = np.array([0.1, 0.2, 0.3])
        # bins that start at 0, so everything is in the underflow bin
        bins = [0, 1, 2]

        with self.assertLogs(logger="puma", level="ERROR") as cm:
            obj = VarVsEff(
                x_var_sig=x_var_sig,
                disc_sig=disc_sig,
                bins=bins,
                working_point=0.5,
            )
        self.assertTrue(
            any("All your signal is in the underflow bin." in msg for msg in cm.output),
            "Expected logger.error about underflow bin was not triggered.",
        )
        # also ensure we still get properly binned arrays
        self.assertEqual(len(obj.disc_binned_sig), len(bins) - 1)
        self.assertTrue(
            all(len(bin_) == 0 for bin_ in obj.disc_binned_sig),
            "All bins should be empty for signal.",
        )

    def test_disc_cut_as_list_or_array(self):
        """
        Covers the scenario in _get_disc_cuts where
        disc_cut is already a list/ndarray. Ensures no error is raised and
        the property is set as-is.
        """
        x_var_sig = np.array([0.1, 0.4, 0.6, 0.9])
        disc_sig = np.array([0.2, 0.3, 0.8, 0.7])
        disc_cut_array = [0.5, 0.5, 0.5]  # must match the number of bins below
        bins = [0, 0.3, 0.7, 1.0]

        obj = VarVsEff(
            x_var_sig=x_var_sig,
            disc_sig=disc_sig,
            bins=bins,
            disc_cut=disc_cut_array,
        )
        self.assertListEqual(obj.disc_cut, disc_cut_array, "disc_cut should match the array given.")

    def test_working_point_as_array_of_two_values(self):
        """
        Covers the scenario in _get_disc_cuts where
        working_point is an array. This triggers the PCFT scenario (i.e., 2D cuts).
        """
        x_var_sig = np.array([0, 1, 2, 3])
        disc_sig = np.array([0.1, 0.5, 0.8, 0.9])
        bins = 2
        wp_array = [0.7, 0.9]  # triggers np.column_stack

        obj = VarVsEff(
            x_var_sig=x_var_sig,
            disc_sig=disc_sig,
            bins=bins,
            working_point=wp_array,
        )
        # disc_cut should be a shape (n_bins, 2) array
        self.assertEqual(len(obj.disc_cut.shape), 2)
        self.assertEqual(obj.disc_cut.shape[1], 2, "disc_cut should have shape (n_bins, 2).")

    def test_efficiency_raises_typeerror_for_invalid_cut_type(self):
        """If the cut is neither float nor np.ndarray, raise TypeError."""
        obj = VarVsEff(
            x_var_sig=np.array([0, 1]),
            disc_sig=np.array([0.2, 0.8]),
            bins=1,
            disc_cut=0.5,
        )
        with self.assertRaises(TypeError):
            # Pass an invalid cut type (string)
            obj.efficiency(np.array([0.2, 0.8]), cut="invalid_cut_type")

    def test_bkg_eff_sig_err_property(self):
        """Calls bkg_eff_sig_err especially the debug statements and return lines."""
        x_var_sig = np.random.uniform(0, 1, 100)
        disc_sig = np.random.uniform(0, 1, 100)
        x_var_bkg = np.random.uniform(0, 1, 100)
        disc_bkg = np.random.uniform(0, 1, 100)

        obj = VarVsEff(
            x_var_sig=x_var_sig,
            disc_sig=disc_sig,
            x_var_bkg=x_var_bkg,
            disc_bkg=disc_bkg,
            bins=2,
            disc_cut=0.5,
        )
        eff, err = obj.bkg_eff_sig_err
        # Just verify shapes and that we don't crash
        self.assertEqual(len(eff), 2)
        self.assertEqual(len(err), 2)

    def test_pcft_working_point(self):
        """Calls PCFT working point."""
        x_var_sig = np.random.uniform(0, 1, 100)
        disc_sig = np.random.uniform(0, 1, 100)
        x_var_bkg = np.random.uniform(0, 1, 100)
        disc_bkg = np.random.uniform(0, 1, 100)

        obj = VarVsEff(
            x_var_sig=x_var_sig,
            disc_sig=disc_sig,
            x_var_bkg=x_var_bkg,
            disc_bkg=disc_bkg,
            bins=2,
            working_point=[0.01, 0.06],
        )
        rej, rej_err = obj.bkg_rej
        self.assertEqual(len(rej), 2)
        self.assertEqual(len(rej_err), 2)
        self.assertAlmostEqual(rej[0], 11.2)
        self.assertAlmostEqual(rej[1], 14.66666666)
        self.assertAlmostEqual(rej_err[0], 4.77995816)
        self.assertAlmostEqual(rej_err[1], 8.17403296)

        eff, eff_err = obj.bkg_eff_sig_err
        self.assertEqual(len(eff), 2)
        self.assertEqual(len(eff_err), 2)
        self.assertAlmostEqual(eff[0], 0.08928571)
        self.assertAlmostEqual(eff[1], 0.06818182)
        self.assertAlmostEqual(eff_err[0], 0.03640475)
        self.assertAlmostEqual(eff_err[1], 0.02569958)

    def test_pcft_working_point_cut_type_error(self):
        """Check if the type error is raised correctly."""
        x_var_sig = np.random.uniform(0, 1, 100)
        disc_sig = np.random.uniform(0, 1, 100)
        x_var_bkg = np.random.uniform(0, 1, 100)
        disc_bkg = np.random.uniform(0, 1, 100)

        with self.assertRaises(TypeError) as ctx:
            _ = VarVsEff(
                x_var_sig=x_var_sig,
                disc_sig=disc_sig,
                x_var_bkg=x_var_bkg,
                disc_bkg=disc_bkg,
                bins=2,
                working_point="Error",
            )
        self.assertEqual(
            "`working_point` must either be a list or a float! You gave <class 'str'>",
            str(ctx.exception),
        )

    def test_equality_with_different_type(self):
        """If other is not an instance of VarVsEff, return False immediately."""
        x_var_sig = np.array([0, 1])
        disc_sig = np.array([0.2, 0.8])
        obj = VarVsEff(
            x_var_sig=x_var_sig,
            disc_sig=disc_sig,
            bins=1,
            disc_cut=0.5,
        )
        self.assertFalse(obj == "a string", "Comparing with non-VarVsEff should return False.")

    def test_get_bkg_eff_sig_err(self):
        """Covers get('bkg_eff_sig_err')."""
        x_var_sig = np.array([0, 1])
        disc_sig = np.array([0.2, 0.8])
        x_var_bkg = np.array([0, 1])
        disc_bkg = np.array([0.3, 0.7])
        obj = VarVsEff(
            x_var_sig=x_var_sig,
            disc_sig=disc_sig,
            x_var_bkg=x_var_bkg,
            disc_bkg=disc_bkg,
            bins=1,
            disc_cut=0.5,
        )
        eff, err = obj.get("bkg_eff_sig_err")
        self.assertIsNotNone(eff)
        self.assertIsNotNone(err)

    def test_get_sig_rej(self):
        """Covers get('sig_rej')."""
        x_var_sig = np.array([0, 1])
        disc_sig = np.array([0.2, 0.8])
        x_var_bkg = np.array([0, 1])
        disc_bkg = np.array([0.3, 0.7])
        obj = VarVsEff(
            x_var_sig=x_var_sig,
            disc_sig=disc_sig,
            x_var_bkg=x_var_bkg,
            disc_bkg=disc_bkg,
            bins=1,
            disc_cut=0.5,
        )
        eff, err = obj.get("sig_rej")
        self.assertIsNotNone(eff)
        self.assertIsNotNone(err)

    def test_get_invalid_mode_resets_inverse_cut(self):
        """Passing an unsupported mode raises ValueError AND resets self.inverse_cut to False."""
        x_var_sig = np.array([0, 1])
        disc_sig = np.array([0.2, 0.8])
        obj = VarVsEff(
            x_var_sig=x_var_sig,
            disc_sig=disc_sig,
            bins=1,
            disc_cut=0.5,
        )

        # We set inverse_cut to True so we can check it gets reset on ValueError
        obj.inverse_cut = True
        with self.assertRaises(ValueError):
            obj.get("not_a_valid_mode", inverse_cut=True)

        # After the exception, self.inverse_cut should be False again
        self.assertFalse(
            obj.inverse_cut, "inverse_cut should be reset to False after raising ValueError."
        )


class VarVsEffOutputTestCase(unittest.TestCase):
    """Test class for the VarVsEffPlot output."""

    def setUp(self):
        """
        Prepare a temporary directory for saving output plots and
        generate data for signal/background distributions.

        The data is crafted so that:
        - Background distributions (self.disc_bkg, self.x_var_bkg) are the same
          for any "tagger".
        - 'reference tagger' has a constant separation power for the full pT range.
        - 'new tagger' has improved separation for pT > 110.
        """
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint:disable=R1732
        self.actual_plots_dir = f"{self.tmp_dir.name}/"
        # Replace this with the actual path to your "expected_plots" directory
        self.expected_plots_dir = os.path.join(os.path.dirname(__file__), "expected_plots")

        np.random.seed(42)
        n_random = 10_000

        # background
        self.disc_bkg = np.random.normal(loc=0, size=2 * n_random)
        self.x_var_bkg = np.random.uniform(0, 250, size=2 * n_random)

        # reference tagger
        self.disc_sig_1 = np.random.normal(loc=1, size=2 * n_random)
        self.x_var_sig_1 = np.random.uniform(0, 250, size=2 * n_random)

        # new tagger
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

    def tearDown(self):
        """Clean up the temporary directory."""
        self.tmp_dir.cleanup()

    def test_var_vs_eff_plot_mode_option(self):
        """Ensure passing an invalid mode to VarVsEffPlot raises ValueError."""
        with self.assertRaises(ValueError):
            VarVsEffPlot(
                mode="test",  # invalid mode
                ylabel="Background rejection",
                xlabel=r"$p_{T}$ [GeV]",
                logy=True,
                atlas_second_tag="test",
                y_scale=1.5,
                n_ratio_panels=1,
                figsize=(9, 6),
            )

    def test_output_plot_flat_per_bin_bkg_rejection(self):
        """
        Test output plot with 'bkg_rej' mode and flat-per-bin setup.

        We compare the generated image against a reference file
        to ensure no regressions in the plot appearance.
        """
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
        """
        Test output plot with 'bkg_eff' mode and flat-per-bin setup.

        We again compare the generated image with a stored reference.
        """
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
        """
        Test that apply_modified_atlas_second_tag() produces the correct text
        when a working point is provided (no disc_cut).
        """
        flat_wp_plot = VarVsEffPlot(
            mode="bkg_eff",
        )
        signal = Flavours["bjets"]

        flat_wp_plot.apply_modified_atlas_second_tag(signal=signal, working_point=0.7)
        expected_tag = "70% $b$-jet efficiency"
        self.assertEqual(flat_wp_plot.atlas_second_tag, expected_tag)

    def test_var_vs_eff_info_str_pcft_working_point(self):
        """
        Test that apply_modified_atlas_second_tag() produces the correct text
        when a PCFT working point is provided (no disc_cut).
        """
        flat_wp_plot = VarVsEffPlot(
            mode="bkg_eff",
        )
        signal = Flavours["bjets"]

        flat_wp_plot.apply_modified_atlas_second_tag(signal=signal, working_point=[0.7, 0.77])
        expected_tag = "70% - 77% $b$-jet efficiency"
        self.assertEqual(flat_wp_plot.atlas_second_tag, expected_tag)

    def test_var_vs_eff_info_str_flat_wp(self):
        """
        Test that apply_modified_atlas_second_tag() adds 'Flat ... per bin'
        if both working_point and flat_per_bin=True are specified.
        """
        flat_wp_plot = VarVsEffPlot(
            mode="bkg_eff",
            atlas_second_tag="test",
        )
        signal = Flavours["bjets"]

        flat_wp_plot.apply_modified_atlas_second_tag(
            signal=signal, working_point=0.7, flat_per_bin=True
        )

        expected_tag = "test\nFlat 70% $b$-jet efficiency per bin"
        self.assertEqual(flat_wp_plot.atlas_second_tag, expected_tag)

    def test_var_vs_eff_info_str_fixed_disc(self):
        """
        Test that apply_modified_atlas_second_tag() produces the correct text
        when a disc_cut is specified (and no working_point).
        """
        flat_wp_plot = VarVsEffPlot(
            mode="bkg_eff",
        )
        signal = Flavours["bjets"]

        flat_wp_plot.apply_modified_atlas_second_tag(signal=signal, disc_cut=3.0)
        expected_tag = "$D_{b}$ > 3.0"
        self.assertEqual(flat_wp_plot.atlas_second_tag, expected_tag)
