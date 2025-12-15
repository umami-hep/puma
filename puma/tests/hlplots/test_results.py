"""Unit test script for the functions in hlplots/tagger.py."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from ftag import Flavours, get_mock_file
from ftag.hdf5 import structured_from_dict

from puma.histogram import Histogram, HistogramPlot
from puma.hlplots import Results, separate_kwargs
from puma.hlplots.tagger import Tagger
from puma.utils import logger, set_log_level

set_log_level(logger, "DEBUG")


class ResultsTestCase(unittest.TestCase):
    """Test class for the Results class."""

    def test_set_signal_hcc(self):
        """Test set_signal for hcc."""
        results = Results(signal="hcc", sample="test", category="xbb")
        self.assertEqual(
            results.backgrounds,
            [*Flavours.by_category("xbb").backgrounds(Flavours.hcc)],
        )
        self.assertEqual(results.signal, Flavours.hcc)

    def test_add_duplicated(self):
        """Test empty string as model name."""
        dummy_tagger_1 = Tagger("dummy")
        dummy_tagger_2 = Tagger("dummy")
        results = Results(signal="bjets", sample="test")
        results.add(dummy_tagger_1)
        with self.assertRaises(KeyError):
            results.add(dummy_tagger_2)

    def test_add_2_taggers(self):
        """Test empty string as model name."""
        dummy_tagger_1 = Tagger("dummy")
        dummy_tagger_2 = Tagger("dummy_2")
        results = Results(signal="bjets", sample="test")
        results.add(dummy_tagger_1)
        results.add(dummy_tagger_2)
        self.assertEqual(
            list(results.taggers.keys()),
            ["dummy (dummy)", "dummy_2 (dummy_2)"],  # pylint: disable=W0212
        )

    def test_get_taggers(self):
        """Test empty string as model name."""
        dummy_tagger_1 = Tagger("dummy")
        dummy_tagger_2 = Tagger("dummy_2")
        results = Results(signal="bjets", sample="test")
        results.add(dummy_tagger_1)
        results.add(dummy_tagger_2)
        retrieved_dummy_tagger_2 = results["dummy_2 (dummy_2)"]
        self.assertEqual(retrieved_dummy_tagger_2.name, dummy_tagger_2.name)

    def test_load_taggers_from_file(self):
        """Test for Results.load_taggers_from_file function."""
        fname = get_mock_file()[0]
        results = Results(signal="bjets", sample="test")
        taggers = [Tagger("MockTagger")]
        results.load_taggers_from_file(taggers, fname)
        self.assertEqual(list(results.taggers.values()), taggers)

    def test_load_taggers_from_file_with_perf_vars(self):
        """Test for Results.load_taggers_from_file function."""
        fname = get_mock_file()[0]
        results = Results(signal="bjets", sample="test", perf_vars=["pt", "eta"])
        taggers = [Tagger("MockTagger")]
        results.load_taggers_from_file(taggers, fname)
        self.assertEqual(list(results.taggers.values()), taggers)

    def test_add_taggers_with_cuts_override_perf_vars(self):
        """Test for Results.load_taggers_from_file function."""
        rng = np.random.default_rng(seed=16)
        cuts = [("eta", ">", 0)]
        tagger_cuts = [("pt", ">", 20)]
        fname = get_mock_file(num_jets=1000)[0]
        results = Results(signal="bjets", sample="test", perf_vars=["pt", "eta"])
        taggers = [Tagger("MockTagger", cuts=tagger_cuts)]

        results.load_taggers_from_file(
            taggers,
            fname,
            cuts=cuts,
            perf_vars={
                "pt": rng.exponential(100, size=1000),
                "eta": rng.normal(0, 1, size=1000),
            },
        )
        self.assertEqual(list(results.taggers.values()), taggers)

    def test_add_taggers_with_cuts(self):
        fname = get_mock_file()[0]
        cuts = [("eta", ">", 0)]
        tagger_cuts = [("pt", ">", 20)]
        results = Results(signal="bjets", sample="test")
        taggers = [Tagger("MockTagger", cuts=tagger_cuts)]
        results.load_taggers_from_file(taggers, fname, cuts=cuts)
        self.assertEqual(list(results.taggers.values()), taggers)

    def test_add_taggers_taujets(self):
        # get mock file and rename variables match taujets
        fname = get_mock_file()[0]
        results = Results(
            signal="bjets",
            sample="test",
        )
        taggers = [Tagger("MockTagger", fxs={"fu": 0.1, "fc": 0.1, "ftau": 0.1})]
        results.load_taggers_from_file(taggers, fname)
        assert "MockTagger_ptau" in taggers[0].scores.dtype.names
        taggers[0].discriminant("bjets")

    def test_add_taggers_hbb(self):
        # get mock file and rename variables match hbb
        f = get_mock_file()[1]
        d = {}
        d["R10TruthLabel"] = f["jets"]["HadronConeExclTruthLabelID"]
        d["MockTagger_phbb"] = f["jets"]["MockTagger_pb"]
        d["MockTagger_phcc"] = f["jets"]["MockTagger_pc"]
        d["MockTagger_ptop"] = f["jets"]["MockTagger_pu"]
        d["MockTagger_pqcd"] = f["jets"]["MockTagger_pu"]
        d["pt"] = f["jets"]["pt"]
        array = structured_from_dict(d)
        with tempfile.TemporaryDirectory() as tmp_file:
            fname = Path(tmp_file) / "test.h5"
            with h5py.File(fname, "w") as f:
                f.create_dataset("jets", data=array)

            results = Results(signal="hbb", sample="test", category="xbb")
            results.load_taggers_from_file(
                [
                    Tagger(
                        "MockTagger",
                        category="xbb",
                        output_flavours=["hbb", "hcc", "top", "qcd"],
                    )
                ],
                fname,
                label_var="R10TruthLabel",
            )

    def test_add_taggers_keep_nan(self):
        # get mock file and add nans
        f = get_mock_file()[1]
        d = {}
        d["HadronConeExclTruthLabelID"] = f["jets"]["HadronConeExclTruthLabelID"]
        d["MockTagger_pb"] = f["jets"]["MockTagger_pb"]
        d["MockTagger_pc"] = f["jets"]["MockTagger_pc"]
        d["MockTagger_pu"] = f["jets"]["MockTagger_pu"]
        d["pt"] = f["jets"]["pt"]
        n_nans = np.random.choice(range(100), 10)
        d["MockTagger_pb"][n_nans] = np.nan
        array = structured_from_dict(d)
        with tempfile.TemporaryDirectory() as tmp_file:
            fname = Path(tmp_file) / "test.h5"
            with h5py.File(fname, "w") as f:
                f.create_dataset("jets", data=array)

            results = Results(signal="bjets", sample="test", remove_nan=False)
            with self.assertRaises(ValueError):
                results.load_taggers_from_file([Tagger("MockTagger")], fname)

    def test_add_taggers_remove_nan(self):
        # get mock file and add nans
        f = get_mock_file()[1]
        d = {}
        d["HadronConeExclTruthLabelID"] = f["jets"]["HadronConeExclTruthLabelID"]
        d["MockTagger_pb"] = f["jets"]["MockTagger_pb"]
        d["MockTagger_pc"] = f["jets"]["MockTagger_pc"]
        d["MockTagger_pu"] = f["jets"]["MockTagger_pu"]
        d["pt"] = f["jets"]["pt"]
        n_nans = np.random.choice(range(100), 10, replace=False)
        d["MockTagger_pb"][n_nans] = np.nan
        array = structured_from_dict(d)
        with tempfile.TemporaryDirectory() as tmp_file:
            fname = Path(tmp_file) / "test.h5"
            with h5py.File(fname, "w") as f:
                f.create_dataset("jets", data=array)

            results = Results(signal="bjets", sample="test", remove_nan=True)
            with self.assertLogs("puma", "WARNING") as cm:
                results.load_taggers_from_file(
                    taggers=[Tagger("MockTagger", output_flavours=["ujets", "cjets", "bjets"])],
                    file_path=fname,
                )
            self.assertEqual(
                cm.output,
                [f"WARNING:puma:{len(n_nans)} NaN values found in loaded data. Removing them."],
            )

    def test_add_taggers_ValueError(self):
        """Testing raise of ValueError if NaNs are still present."""
        # get mock file and add nans
        f = get_mock_file()[1]
        d = {}
        d["HadronConeExclTruthLabelID"] = f["jets"]["HadronConeExclTruthLabelID"]
        d["MockTagger_pb"] = f["jets"]["MockTagger_pb"]
        d["MockTagger_pc"] = f["jets"]["MockTagger_pc"]
        d["MockTagger_pu"] = f["jets"]["MockTagger_pu"]
        d["pt"] = f["jets"]["pt"]
        n_nans = np.random.choice(range(100), 10, replace=False)
        d["MockTagger_pb"][n_nans] = np.nan
        array = structured_from_dict(d)
        with tempfile.TemporaryDirectory() as tmp_file:
            fname = Path(tmp_file) / "test.h5"
            with h5py.File(fname, "w") as f:
                f.create_dataset("jets", data=array)

            results = Results(signal="bjets", sample="test", remove_nan=False)
            with self.assertRaises(ValueError):
                results.load_taggers_from_file(
                    taggers=[Tagger("MockTagger", output_flavours=["ujets", "cjets", "bjets"])],
                    file_path=fname,
                )


class ResultsPlotsTestCase(unittest.TestCase):
    """Test class for the Results class running plots."""

    def setUp(self) -> None:
        """Set up for unit tests."""
        f = get_mock_file()[1]
        dummy_tagger_1 = Tagger("MockTagger", output_flavours=["ujets", "cjets", "bjets"])
        dummy_tagger_1.labels = np.array(
            f["jets"]["HadronConeExclTruthLabelID"],
            dtype=[("HadronConeExclTruthLabelID", "i4")],
        )
        dummy_tagger_1.scores = f["jets"]
        dummy_tagger_1.label = "dummy tagger"
        self.dummy_tagger_1 = dummy_tagger_1

    def test_plot_probs_bjets(self):
        """Test that png file is being created."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fc": 0.05, "fu": 0.95}
        with tempfile.TemporaryDirectory() as tmp_file:
            results = Results(signal="bjets", sample="test", output_dir=tmp_file)
            results.add(self.dummy_tagger_1)
            results.plot_probs(bins=40, bins_range=(0, 1))
            for fpath in results.saved_plots:
                assert fpath.is_file()
            results.saved_plots = []

    def test_plot_discs_bjets(self):
        """Test that png file is being created."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fc": 0.05, "fu": 0.95}
        with tempfile.TemporaryDirectory() as tmp_file:
            results = Results(signal="bjets", sample="test", output_dir=tmp_file)
            results.add(self.dummy_tagger_1)
            results.plot_discs(bins=40, bins_range=(-2, 15))
            for fpath in results.saved_plots:
                assert fpath.is_file()
            results.saved_plots = []

    def test_plot_discs_cjets(self):
        """Test that png file is being created."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fb": 0.05, "fu": 0.95}
        with tempfile.TemporaryDirectory() as tmp_file:
            results = Results(signal="cjets", sample="test", output_dir=tmp_file)
            results.add(self.dummy_tagger_1)
            results.plot_discs(bins=40, bins_range=(-2, 15), wp_vlines=[60])
            for fpath in results.saved_plots:
                assert fpath.is_file()
            results.saved_plots = []

    def test_plot_roc_bjets(self):
        """Test that png file is being created."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fc": 0.05, "fu": 0.95}
        with tempfile.TemporaryDirectory() as tmp_file:
            results = Results(signal="bjets", sample="test", output_dir=tmp_file)
            results.add(self.dummy_tagger_1)
            results.plot_rocs()
            for fpath in results.saved_plots:
                assert fpath.is_file()
            results.saved_plots = []

    def test_plot_roc_cjets(self):
        """Test that png file is being created."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fb": 0.05, "fu": 0.95}
        with tempfile.TemporaryDirectory() as tmp_file:
            results = Results(signal="cjets", sample="test", output_dir=tmp_file)
            results.add(self.dummy_tagger_1)
            results.plot_rocs(fontsize=5)
            for fpath in results.saved_plots:
                assert fpath.is_file()
            results.saved_plots = []

    def test_plot_var_perf_err(self):
        """Tests the performance plots throws errors with invalid inputs."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fc": 0.05, "fu": 0.95}
        self.dummy_tagger_1.disc_cut = 2
        rng = np.random.default_rng(seed=16)
        self.dummy_tagger_1.perf_vars = {
            "pt": rng.exponential(100, size=len(self.dummy_tagger_1.scores))
        }
        with tempfile.TemporaryDirectory() as tmp_file:
            results = Results(signal="bjets", sample="test", output_dir=tmp_file)
            results.add(self.dummy_tagger_1)
            with self.assertRaises(ValueError):
                results.plot_var_perf(
                    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
                )
            with self.assertRaises(ValueError):
                results.plot_var_perf(
                    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
                    disc_cut=1,
                    working_point=0.5,
                )

    def test_plot_var_eff_per_flat_rej_err(self):
        """Tests the performance vs flat rejection plots throws errors
        with invalid inputs.
        """
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fc": 0.05, "fu": 0.95}
        self.dummy_tagger_1.disc_cut = 2
        rng = np.random.default_rng(seed=16)
        self.dummy_tagger_1.perf_vars = {
            "pt": rng.exponential(100, size=len(self.dummy_tagger_1.scores))
        }
        with tempfile.TemporaryDirectory() as tmp_file, self.assertRaises(ValueError):
            results = Results(signal="bjets", sample="test", output_dir=tmp_file)
            results.plot_flat_rej_var_perf(
                bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
                fixed_rejections={"cjets": 10, "ujets": 100},
                working_point=0.5,
            )
        with tempfile.TemporaryDirectory() as tmp_file, self.assertRaises(ValueError):
            results.plot_flat_rej_var_perf(
                bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
                fixed_rejections={"cjets": 10, "ujets": 100},
                disc_cut=0.5,
            )

    def test_plot_var_perf_bjets(self):
        """Test that png file is being created."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fc": 0.05, "fu": 0.95}
        self.dummy_tagger_1.disc_cut = 2
        rng = np.random.default_rng(seed=16)
        self.dummy_tagger_1.perf_vars = {
            "pt": rng.exponential(100, size=len(self.dummy_tagger_1.scores))
        }
        with tempfile.TemporaryDirectory() as tmp_file:
            results = Results(signal="bjets", sample="test", output_dir=tmp_file)
            results.add(self.dummy_tagger_1)
            results.plot_var_perf(
                bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
                working_point=0.7,
            )
            for fpath in results.saved_plots:
                assert fpath.is_file()
            results.saved_plots = []

    def test_plot_var_perf_bjets_disc_cut(self):
        """Test that png file is being created."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fc": 0.05, "fu": 0.95}
        self.dummy_tagger_1.disc_cut = 2
        rng = np.random.default_rng(seed=16)
        self.dummy_tagger_1.perf_vars = {
            "pt": rng.exponential(100, size=len(self.dummy_tagger_1.scores))
        }
        with tempfile.TemporaryDirectory() as tmp_file:
            results = Results(signal="bjets", sample="test", output_dir=tmp_file)
            results.add(self.dummy_tagger_1)
            results.plot_var_perf(
                bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
                disc_cut=0.5,
            )
            for fpath in results.saved_plots:
                assert fpath.is_file()
            results.saved_plots = []

    def test_plot_var_perf_bjets_pcft(self):
        """Test that png file is being created."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fc": 0.05, "fu": 0.95}
        self.dummy_tagger_1.disc_cut = 2
        rng = np.random.default_rng(seed=16)
        self.dummy_tagger_1.perf_vars = {
            "pt": rng.exponential(100, size=len(self.dummy_tagger_1.scores))
        }
        with tempfile.TemporaryDirectory() as tmp_file:
            results = Results(signal="bjets", sample="test", output_dir=tmp_file)
            results.add(self.dummy_tagger_1)
            results.plot_var_perf(
                bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
                working_point=[0.5, 0.8],
            )
            for fpath in results.saved_plots:
                assert fpath.is_file()
            results.saved_plots = []

    def test_plot_var_perf_extra_kwargs(self):
        """Test that png file is being created."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fc": 0.05, "fu": 0.95}
        self.dummy_tagger_1.disc_cut = 2
        rng = np.random.default_rng(seed=16)
        self.dummy_tagger_1.perf_vars = {
            "pt": rng.exponential(100, size=len(self.dummy_tagger_1.scores))
        }
        with tempfile.TemporaryDirectory() as tmp_file:
            results = Results(signal="bjets", sample="test", output_dir=tmp_file)
            results.add(self.dummy_tagger_1)
            results.plot_var_perf(
                bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
                working_point=0.7,
                y_scale=1.3,
            )
            for fpath in results.saved_plots:
                assert fpath.is_file()
            results.saved_plots = []

    def test_plot_var_perf_multi_bjets(self):
        """Test that png file is being created."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fc": 0.05, "fu": 0.95}
        self.dummy_tagger_1.disc_cut = 2
        rng = np.random.default_rng(seed=16)
        self.dummy_tagger_1.perf_vars = {
            "pt": rng.exponential(100, size=len(self.dummy_tagger_1.scores)),
            "eta": rng.normal(0, 1, size=len(self.dummy_tagger_1.scores)),
        }
        with tempfile.TemporaryDirectory() as tmp_file:
            results = Results(signal="bjets", sample="test", output_dir=tmp_file)
            results.add(self.dummy_tagger_1)
            results.plot_var_perf(
                bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
                working_point=0.7,
                perf_var="pt",
            )
            results.plot_var_perf(
                bins=np.linspace(-0.5, 0.5, 10), working_point=0.7, perf_var="eta"
            )
            for fpath in results.saved_plots:
                assert fpath.is_file()
            results.saved_plots = []

    def test_plot_var_perf_cjets(self):
        """Test that png file is being created."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fb": 0.05, "fu": 0.95}
        self.dummy_tagger_1.working_point = 0.5
        rng = np.random.default_rng(seed=16)
        self.dummy_tagger_1.perf_vars = {
            "pt": rng.exponential(100, size=len(self.dummy_tagger_1.scores))
        }
        with tempfile.TemporaryDirectory() as tmp_file:
            results = Results(signal="cjets", sample="test", output_dir=tmp_file)
            results.add(self.dummy_tagger_1)
            results.plot_var_perf(
                h_line=self.dummy_tagger_1.working_point,
                bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
                working_point=0.7,
            )
            for fpath in results.saved_plots:
                assert fpath.is_file()
            results.saved_plots = []

    def test_plot_beff_vs_flat_rej(self):
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fc": 0.05, "fu": 0.95}
        self.dummy_tagger_1.working_point = 0.5
        rng = np.random.default_rng(seed=16)
        self.dummy_tagger_1.perf_vars = {
            "pt": rng.exponential(100, size=len(self.dummy_tagger_1.scores))
        }
        with tempfile.TemporaryDirectory() as tmp_file:
            results = Results(signal="bjets", sample="test", output_dir=tmp_file)
            results.add(self.dummy_tagger_1)
            results.plot_flat_rej_var_perf(
                fixed_rejections={"cjets": 10, "ujets": 100},
                bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
                h_line=0.5,
            )
            for fpath in results.saved_plots:
                assert fpath.is_file()
            results.saved_plots = []

    def test_plot_beff_vs_flat_rej_extra_kwargs(self):
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fc": 0.05, "fu": 0.95}
        self.dummy_tagger_1.working_point = 0.5
        rng = np.random.default_rng(seed=16)
        self.dummy_tagger_1.perf_vars = {
            "pt": rng.exponential(100, size=len(self.dummy_tagger_1.scores))
        }
        with tempfile.TemporaryDirectory() as tmp_file:
            results = Results(signal="bjets", sample="test", output_dir=tmp_file)
            results.add(self.dummy_tagger_1)
            results.plot_flat_rej_var_perf(
                fixed_rejections={"cjets": 10, "ujets": 100},
                bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
                h_line=0.5,
                y_scale=1.3,
            )
            for fpath in results.saved_plots:
                assert fpath.is_file()
            results.saved_plots = []

    def test_plot_ceff_vs_flat_rej(self):
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fb": 0.05, "fu": 0.95}
        self.dummy_tagger_1.working_point = 0.5
        rng = np.random.default_rng(seed=16)
        self.dummy_tagger_1.perf_vars = {
            "pt": rng.exponential(100, size=len(self.dummy_tagger_1.scores))
        }
        with tempfile.TemporaryDirectory() as tmp_file:
            results = Results(signal="cjets", sample="test", output_dir=tmp_file)
            results.add(self.dummy_tagger_1)
            results.plot_flat_rej_var_perf(
                fixed_rejections={"bjets": 10, "ujets": 100},
                bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
            )
            for fpath in results.saved_plots:
                assert fpath.is_file()
            results.saved_plots = []

    def test_plot_fraction_scans_hbb_error(self):
        """Test that correct error is raised."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fc": 0.05, "fu": 0.95}
        with tempfile.TemporaryDirectory() as tmp_file:
            results = Results(signal="hbb", sample="test", category="xbb", output_dir=tmp_file)
            results.add(self.dummy_tagger_1)
            with self.assertRaises(ValueError):
                results.plot_fraction_scans(
                    backgrounds_to_plot=["cjets", "ujets"],
                    rej=False,
                    plot_optimal_fraction_values=True,
                )

    def test_plot_fraction_scans_bjets_eff(self):
        """Test that png file is being created."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fc": 0.05, "fu": 0.95}
        with tempfile.TemporaryDirectory() as tmp_file:
            results = Results(signal="bjets", sample="test", output_dir=tmp_file)
            results.add(self.dummy_tagger_1)
            results.plot_fraction_scans(
                backgrounds_to_plot=["cjets", "ujets"],
                rej=False,
                plot_optimal_fraction_values=True,
            )
            for fpath in results.saved_plots:
                assert fpath.is_file()
            results.saved_plots = []

    def test_plot_fraction_scans_cjets_rej(self):
        """Test that png file is being created."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fb": 0.05, "fu": 0.95}
        with tempfile.TemporaryDirectory() as tmp_file:
            results = Results(signal="cjets", sample="test", output_dir=tmp_file)
            results.add(self.dummy_tagger_1)
            results.plot_fraction_scans(
                backgrounds_to_plot=["bjets", "ujets"],
                rej=False,
                plot_optimal_fraction_values=True,
            )
            for fpath in results.saved_plots:
                assert fpath.is_file()
            results.saved_plots = []

    def test_plot_fraction_scans_multiple_bkg_error(self):
        """Test error of more than two backgrounds."""
        self.dummy_tagger_1.reference = True
        self.dummy_tagger_1.fxs = {"fc": 0.05, "fu": 0.95}
        with tempfile.TemporaryDirectory() as tmp_file:
            results = Results(signal="bjets", sample="test", output_dir=tmp_file)
            results.add(self.dummy_tagger_1)
            with self.assertRaises(ValueError):
                results.plot_fraction_scans(
                    backgrounds_to_plot=["bjets", "ujets", "taujets"],
                    rej=False,
                    plot_optimal_fraction_values=True,
                )

    def test_make_plot_error(self):
        """Test error of non-existing plot type."""
        results = Results(signal="bjets", sample="test")
        with self.assertRaises(ValueError):
            results.make_plot(plot_type="crash", kwargs={})


def _class_known_keys(cls: type[Any]) -> set[str]:
    """Return all valid option names for a class from dataclass fields and __init__ params.

    Parameters
    ----------
    cls : type[Any]
        Class for which to return the known keys

    Returns
    -------
    set[str]
        Set of strings of the known keys
    """
    """"""
    keys: set[str] = set()

    if is_dataclass(cls):
        keys.update(f.name for f in fields(cls))

    sig = inspect.signature(cls.__init__)
    for name, p in sig.parameters.items():
        if name == "self" or p.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        keys.add(name)

    return keys


def _apply_expected_for_class(
    provided_defaults: dict[str, Any],
    kwargs: dict[str, Any] | None,
    class_keys: set[str],
) -> dict[str, Any]:
    """Build expected output for one class.

    - Take only keys that belong to the class.
    - Start with provided defaults for those keys.
    - Overlay kwargs (if any) for those keys (kwargs win).

    Parameters
    ----------
    provided_defaults : dict[str, Any]
        Provided defaults for the class
    kwargs : dict[str, Any] | None
        kwargs provided
    class_keys : set[str]
        Class keys

    Returns
    -------
    dict[str, Any]
        Expected output for one class
    """
    expected = {k: v for k, v in provided_defaults.items() if k in class_keys}
    if kwargs:
        expected |= {k: v for k, v in kwargs.items() if k in class_keys}
    return expected


@dataclass
class _FactoryDataclass:
    # Required: no default -> should go through the "else: ... = None" path
    a: int
    # Has default_factory -> should go through the "df is not MISSING" path
    b: list[int] = field(default_factory=list)


class SeparateKwargsTestCase(unittest.TestCase):
    """Tests for separate_kwargs with data-driven expectations."""

    def setUp(self) -> None:
        self.separate_kwargs = separate_kwargs
        self.classes = [HistogramPlot, Histogram]

        # Provided defaults (per-class) and incoming kwargs.
        # NOTE: keys that aren't part of a class will be ignored automatically.
        self.defaults: list[dict[str, Any]] = [
            {"xlabel": "X label", "bins_range": (1.0, 2.0), "unknown_plot": "ignored"},
            {"xmin": 15, "unknown_curve": "ignored"},
        ]
        self.kwargs = {
            "ylabel": "Y label",
            "ymin": 5,
            "bins_range": (0.5, 3.0),  # will override default if 'bins_range' exists in class 0
            "ghost": "ignored",
            "xmin": 999,  # will override default if 'xmin' exists in class 1
        }

        # Precompute class key sets
        self.class_keys = [_class_known_keys(cls) for cls in self.classes]

    def test_default_behaviour(self) -> None:
        """With kwargs, return only updated keys; kwargs override provided defaults."""
        out = self.separate_kwargs(
            kwargs=self.kwargs,
            classes=self.classes,
            defaults=self.defaults,
        )

        # Build expected per class from the rules,
        # but only keep keys that were actually updated (provided defaults or kwargs).
        expected = []
        for i, keys in enumerate(self.class_keys):
            exp = _apply_expected_for_class(self.defaults[i], self.kwargs, keys)
            expected.append(exp)

        self.assertEqual(len(out), len(expected))
        for got, exp in zip(out, expected, strict=False):
            self.assertDictEqual(got, exp)

    def test_kwargs_none(self) -> None:
        """With None, keep only keys overridden by provided defaults (no untouched defaults)."""
        out = self.separate_kwargs(
            kwargs=None,
            classes=self.classes,
            defaults=self.defaults,
        )

        expected = []
        for i, keys in enumerate(self.class_keys):
            exp = _apply_expected_for_class(self.defaults[i], None, keys)
            expected.append(exp)

        self.assertEqual(len(out), len(expected))
        for got, exp in zip(out, expected, strict=False):
            self.assertDictEqual(got, exp)

    def test_ignores_unknown_keys(self) -> None:
        """Unknown keys in provided defaults or kwargs must be ignored."""
        defaults: list[dict[str, Any]] = [
            {"nope1": 123, "xlabel": "XL"},  # only 'xlabel' might belong to class 0
            {"nope2": 456, "xmin": 1},  # only 'xmin' might belong to class 1
        ]
        kwargs = {"nope3": 789}  # should be ignored everywhere

        out = self.separate_kwargs(
            kwargs=kwargs,
            classes=self.classes,
            defaults=defaults,
        )

        expected = []
        for i, keys in enumerate(self.class_keys):
            exp = _apply_expected_for_class(defaults[i], kwargs, keys)
            expected.append(exp)

        self.assertEqual(len(out), len(expected))
        for got, exp in zip(out, expected, strict=False):
            self.assertDictEqual(got, exp)

    def test_no_updates_returns_empty(self) -> None:
        """If neither provided defaults nor kwargs touch a class key, that class dict is empty."""
        # Choose keys you believe won't exist in either class
        defaults = [{"__nothing__": 1}, {"__still_nothing__": 2}]
        kwargs = {"__ghost__": 3}

        out = self.separate_kwargs(
            kwargs=kwargs,
            classes=self.classes,
            defaults=defaults,
        )

        # Expect empty dicts for any class whose keys weren't updated at all
        for i, got in enumerate(out):
            # Build expected using helper; should be empty if no overlap
            exp = _apply_expected_for_class(defaults[i], kwargs, self.class_keys[i])
            self.assertDictEqual(got, exp)  # likely {}

    def test_kwargs_take_precedence_when_overlap_exists(self) -> None:
        """
        If both provided defaults and kwargs set the same key for a class, kwargs should win.
        Only checks the rule when such a key actually exists in the class.
        """
        # For each class, find a key present in both provided defaults and kwargs and in the class.
        new_defaults = [dict(d) for d in self.defaults]
        new_kwargs = dict(self.kwargs)

        for i, keys in enumerate(self.class_keys):
            # find a candidate overlap key
            overlap = (keys & new_defaults[i].keys()) & new_kwargs.keys()
            if not overlap:
                # If none exists, try to force one by duplicating a defaults key into kwargs
                for k in keys & new_defaults[i].keys():
                    new_kwargs[k] = "__KWARGS_WINS__"
                    overlap = {k}
                    break

            # If we still have no overlap for this class, skip the check for this class
            if not overlap:
                continue

            # Run the function
            out = self.separate_kwargs(new_kwargs, self.classes, new_defaults)

            # Assert precedence for each overlapping key we created/found
            for k in overlap:
                self.assertIn(k, out[i])
                self.assertEqual(out[i][k], new_kwargs[k])

    def test_required_and_factory_both_exercised_and_overridden(self) -> None:
        """
        - 'a' (required) initially becomes None in class_vars, then overridden by provided defaults.
        - 'b' is realized via default_factory(), then overridden by kwargs.
        Only updated keys are returned.
        """
        classes = [_FactoryDataclass]
        defaults: list[dict[str, Any]] = [{"a": 123}]  # overrides required field
        kwargs: dict[str, Any] = {"b": [7, 8, 9]}  # overrides factory-produced list

        (out,) = separate_kwargs(kwargs=kwargs, classes=classes, defaults=defaults)

        # Both keys should appear because both were updated (by defaults/kwargs, respectively).
        self.assertDictEqual(out, {"a": 123, "b": [7, 8, 9]})

    def test_factory_value_is_dropped_when_not_updated(self) -> None:
        """
        If no provided defaults and no kwargs touch the class keys,
        the result is empty—even though default_factory() and the 'a' None path ran.
        """
        classes = [_FactoryDataclass]
        defaults: list[dict[str, Any]] = [{}]
        kwargs: dict[str, Any] | None = None

        (out,) = separate_kwargs(kwargs=kwargs, classes=classes, defaults=defaults)

        # Nothing was updated, so nothing is returned.
        self.assertDictEqual(out, {})
