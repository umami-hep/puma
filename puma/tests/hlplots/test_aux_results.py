"""Unit test script for the functions in hlplots/aux_results.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from ftag.hdf5 import structured_from_dict

from puma.hlplots import AuxResults
from puma.hlplots.tagger import Tagger
from puma.utils import get_dummy_tagger_aux, logger, set_log_level

set_log_level(logger, "DEBUG")


class AuxResultsTestCase(unittest.TestCase):
    """Test class for the AuxResults class."""

    def test_add_duplicated(self):
        """Test empty string as model name."""
        dummy_tagger_1 = Tagger("dummy")
        dummy_tagger_2 = Tagger("dummy")
        results = AuxResults(sample="test")
        results.add(dummy_tagger_1)
        with self.assertRaises(KeyError):
            results.add(dummy_tagger_2)

    def test_add_2_taggers(self):
        """Test empty string as model name."""
        dummy_tagger_1 = Tagger("dummy")
        dummy_tagger_2 = Tagger("dummy_2")
        results = AuxResults(sample="test")
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
        results = AuxResults(sample="test")
        results.add(dummy_tagger_1)
        results.add(dummy_tagger_2)
        retrieved_dummy_tagger_2 = results["dummy_2 (dummy_2)"]
        self.assertEqual(retrieved_dummy_tagger_2.name, dummy_tagger_2.name)

    def test_load_taggers_from_file(self):
        """Test for AuxResults.load_taggers_from_file function."""
        np.random.default_rng(seed=16)
        fname = get_dummy_tagger_aux()[0]
        results = AuxResults(sample="test")
        taggers = [Tagger("GN2")]
        results.load_taggers_from_file(
            taggers,
            fname,
        )
        self.assertEqual(list(results.taggers.values()), taggers)

    def test_load_taggers_with_cuts(self):
        np.random.default_rng(seed=16)
        fname = get_dummy_tagger_aux()[0]
        cuts = [("eta", ">", 0)]
        tagger_cuts = [("pt", ">", 20)]
        results = AuxResults(sample="test")
        taggers = [Tagger("GN2", cuts=tagger_cuts)]
        results.load_taggers_from_file(
            taggers,
            fname,
            cuts=cuts,
        )
        self.assertEqual(list(results.taggers.values()), taggers)

    def test_load_taggers_with_cuts_override_perf_vars(self):
        rng = np.random.default_rng(seed=16)
        fname = get_dummy_tagger_aux()[0]
        cuts = [("eta", ">", 0)]
        tagger_cuts = [("pt", ">", 20)]
        results = AuxResults(sample="test")
        taggers = [Tagger("GN2", cuts=tagger_cuts)]
        results.load_taggers_from_file(
            taggers,
            fname,
            cuts=cuts,
            perf_vars={
                "pt": rng.exponential(100, size=9999),
                "eta": rng.normal(0, 1, size=9999),
            },
        )
        self.assertEqual(list(results.taggers.values()), taggers)

    def test_load_taggers_with_aux_perf_vars_eta(self):
        np.random.default_rng(seed=16)
        fname = get_dummy_tagger_aux()[0]
        results = AuxResults(
            sample="test",
            aux_perf_vars=["pt", "eta", "dphi"],
        )
        taggers = [Tagger("GN2")]
        results.load_taggers_from_file(taggers, fname)
        tagger_key = list(results.taggers.keys())[0]
        self.assertEqual(
            set(results.taggers[tagger_key].aux_perf_vars.keys()), {"pt", "eta", "dphi"}
        )

    def test_load_taggers_with_aux_perf_vars_deta(self):
        np.random.default_rng(seed=16)
        fname = get_dummy_tagger_aux()[0]
        results = AuxResults(
            sample="test",
            perf_vars=["eta"],
            aux_perf_vars=["pt", "deta", "dphi"],
        )
        taggers = [Tagger("GN2")]
        results.load_taggers_from_file(taggers, fname)
        tagger_key = list(results.taggers.keys())[0]
        self.assertEqual(
            set(results.taggers[tagger_key].aux_perf_vars.keys()), {"pt", "eta", "dphi"}
        )

    def test_add_taggers_keep_nan(self):
        # get mock file and add nans
        f = get_dummy_tagger_aux(size=500)[1]
        d = {}
        d["HadronConeExclTruthLabelID"] = f["jets"]["HadronConeExclTruthLabelID"]
        d["pt"] = f["jets"]["pt"]
        n_nans = np.random.choice(range(len(d["pt"])), 10, replace=False)
        d["pt"][n_nans] = np.nan
        jet_array = structured_from_dict(d)
        track_array = f["tracks"]
        with tempfile.TemporaryDirectory() as tmp_file:
            fname = Path(tmp_file) / "test.h5"
            with h5py.File(fname, "w") as f:
                f.create_dataset("jets", data=jet_array)
                f.create_dataset("tracks", data=track_array)
            results = AuxResults(sample="test", remove_nan=False)
            with self.assertRaises(ValueError):
                results.load_taggers_from_file([Tagger("GN2")], fname)

    def test_add_taggers_remove_nan(self):
        # get mock file and add nans
        f = get_dummy_tagger_aux(size=500)[1]
        d = {}
        d["HadronConeExclTruthLabelID"] = f["jets"]["HadronConeExclTruthLabelID"]
        d["pt"] = f["jets"]["pt"]
        n_nans = np.random.choice(range(len(d["pt"])), 10, replace=False)
        d["pt"][n_nans] = np.nan
        jet_array = structured_from_dict(d)
        track_array = f["tracks"]
        with tempfile.TemporaryDirectory() as tmp_file:
            fname = Path(tmp_file) / "test.h5"
            with h5py.File(fname, "w") as f:
                f.create_dataset("jets", data=jet_array)
                f.create_dataset("tracks", data=track_array)
            results = AuxResults(sample="test", remove_nan=True)
            with self.assertLogs("puma", "WARNING") as cm:
                results.load_taggers_from_file([Tagger("GN2")], fname)
            self.assertEqual(
                cm.output,
                [
                    f"WARNING:puma:{len(n_nans)} NaN values found in loaded data."
                    " Removing them.",
                ],
            )


class AuxResultsPlotsTestCase(unittest.TestCase):
    """Test class for the AuxResults class running plots."""

    def setUp(self) -> None:
        """Set up for unit tests."""
        f = get_dummy_tagger_aux()[1]
        dummy_tagger = Tagger("GN2")
        dummy_tagger.labels = np.array(
            f["jets"]["HadronConeExclTruthLabelID"],
            dtype=[("HadronConeExclTruthLabelID", "i4")],
        )
        dummy_tagger.aux_scores = {
            "vertexing": f["tracks"]["GN2_aux_VertexIndex"],
            "track_origin": f["tracks"]["GN2_aux_TrackOrigin"],
        }
        dummy_tagger.aux_labels = {
            "vertexing": f["tracks"]["ftagTruthVertexIndex"],
            "track_origin": f["tracks"]["ftagTruthOriginLabel"],
        }
        dummy_tagger.perf_vars = {"pt": f["jets"]["pt"]}
        dummy_tagger.aux_perf_vars = {
            "pt": f["tracks"]["pt"],
            "eta": f["tracks"]["eta"],
            "dphi": f["tracks"]["dphi"],
        }
        dummy_tagger.label = "dummy tagger"
        dummy_tagger_no_aux = Tagger("GN2_NoAux", aux_tasks=[])
        dummy_tagger_no_aux.perf_vars = {"pt": f["jets"]["pt"]}
        dummy_tagger_no_aux.label = "dummy tagger no aux"
        self.dummy_tagger = dummy_tagger
        self.dummy_tagger_no_aux = dummy_tagger_no_aux

    def assertIsFile(self, path: str):
        """Check for file to exist.
        Taken from https://stackoverflow.com/a/59198749/10896585.

        Parameters
        ----------
        path : str
            Path to file.

        Raises
        ------
        AssertionError
            if file does not exist
        """
        if not Path(path).resolve().is_file():
            raise AssertionError(f"File does not exist: {path}")

    def test_plot_trackorigin_cm_minmal(self):
        self.dummy_tagger.reference = True
        with tempfile.TemporaryDirectory() as tmp_file:
            auxresults = AuxResults(sample="test", output_dir=tmp_file)
            auxresults.add(self.dummy_tagger)
            auxresults.plot_track_origin_confmat()
            self.assertIsFile(auxresults.get_filename(self.dummy_tagger.name + "_trackOrigin_cm"))

    def test_plot_trackorigin_cm_full(self):
        self.dummy_tagger.reference = True
        with tempfile.TemporaryDirectory() as tmp_file:
            auxresults = AuxResults(sample="test", output_dir=tmp_file)
            auxresults.add(self.dummy_tagger)
            auxresults.plot_track_origin_confmat(minimal_plot=False)
            self.assertIsFile(auxresults.get_filename(self.dummy_tagger.name + "_trackOrigin_cm"))

    def test_plot_var_vtx_perf_bjets(self):
        """Test that png files are being created for tagger with aux tasks."""
        self.dummy_tagger.reference = True
        with tempfile.TemporaryDirectory() as tmp_file:
            auxresults = AuxResults(sample="test", output_dir=tmp_file)
            auxresults.add(self.dummy_tagger)
            auxresults.plot_var_vtx_perf(vtx_flavours=["bjets"])
            self.assertIsFile(auxresults.get_filename("bjets_vtx_eff_vs_pt"))
            self.assertIsFile(auxresults.get_filename("bjets_vtx_purity_vs_pt"))
            self.assertIsFile(auxresults.get_filename("bjets_vtx_trk_eff_vs_pt"))
            self.assertIsFile(auxresults.get_filename("bjets_vtx_trk_purity_vs_pt"))

    def test_plot_var_vtx_perf_ujets(self):
        """Test that png files are being created for tagger with aux tasks."""
        self.dummy_tagger.reference = True
        with tempfile.TemporaryDirectory() as tmp_file:
            auxresults = AuxResults(sample="test", output_dir=tmp_file)
            auxresults.add(self.dummy_tagger)
            auxresults.plot_var_vtx_perf(no_vtx_flavours=["ujets"])
            self.assertIsFile(auxresults.get_filename("ujets_vtx_fakes_vs_pt"))

    def test_plot_var_vtx_perf_alljets(self):
        """Test that png files are being created for tagger with aux tasks."""
        self.dummy_tagger.reference = True
        with tempfile.TemporaryDirectory() as tmp_file:
            auxresults = AuxResults(sample="test", output_dir=tmp_file)
            auxresults.add(self.dummy_tagger)
            auxresults.plot_var_vtx_perf(vtx_flavours=["bjets", "cjets"], no_vtx_flavours=["ujets"])
            self.assertIsFile(auxresults.get_filename("bjets_vtx_eff_vs_pt"))
            self.assertIsFile(auxresults.get_filename("bjets_vtx_purity_vs_pt"))
            self.assertIsFile(auxresults.get_filename("bjets_vtx_trk_eff_vs_pt"))
            self.assertIsFile(auxresults.get_filename("bjets_vtx_trk_purity_vs_pt"))
            self.assertIsFile(auxresults.get_filename("cjets_vtx_eff_vs_pt"))
            self.assertIsFile(auxresults.get_filename("cjets_vtx_purity_vs_pt"))
            self.assertIsFile(auxresults.get_filename("cjets_vtx_trk_eff_vs_pt"))
            self.assertIsFile(auxresults.get_filename("cjets_vtx_trk_purity_vs_pt"))
            self.assertIsFile(auxresults.get_filename("ujets_vtx_fakes_vs_pt"))

    def test_plot_var_vtx_noaux(self):
        """Test that error is raised if no tagger with vertexing is added."""
        self.dummy_tagger.reference = True
        with tempfile.TemporaryDirectory() as tmp_file:
            auxresults = AuxResults(sample="test", output_dir=tmp_file)
            auxresults.add(self.dummy_tagger_no_aux)
            with self.assertRaises(ValueError):
                auxresults.plot_var_vtx_perf(vtx_flavours=["bjets"])

    def test_plot_var_vtx_perf_no_flavours(self):
        """Test vertexing performance when no jet flavours are specified."""
        self.dummy_tagger.reference = True
        with tempfile.TemporaryDirectory() as tmp_file:
            auxresults = AuxResults(sample="test", output_dir=tmp_file)
            auxresults.add(self.dummy_tagger)
            with self.assertRaises(ValueError):
                auxresults.plot_var_vtx_perf()

    def test_plot_var_vtx_perf_empty(self):
        """Test vertexing performance function with empty data."""
        self.dummy_tagger_no_aux.reference = True
        with tempfile.TemporaryDirectory() as tmp_file:
            auxresults = AuxResults(sample="test", output_dir=tmp_file)
            auxresults.add(self.dummy_tagger_no_aux)
            with self.assertRaises(ValueError):
                auxresults.plot_var_vtx_perf(vtx_flavours=["bjets"])

    def test_plot_var_vtx_perf_bjets_inclusive_vertexing(self):
        """Test that png files are being created for tagger with inclusive vertexing enabled."""
        self.dummy_tagger.reference = True
        with tempfile.TemporaryDirectory() as tmp_file:
            auxresults = AuxResults(sample="test", output_dir=tmp_file)
            auxresults.add(self.dummy_tagger)
            auxresults.plot_var_vtx_perf(vtx_flavours=["bjets"], incl_vertexing=True)
            self.assertIsFile(auxresults.get_filename("bjets_vtx_eff_vs_pt", suffix="incl"))
            self.assertIsFile(auxresults.get_filename("bjets_vtx_purity_vs_pt", suffix="incl"))
            self.assertIsFile(auxresults.get_filename("bjets_vtx_trk_eff_vs_pt", suffix="incl"))
            self.assertIsFile(auxresults.get_filename("bjets_vtx_trk_purity_vs_pt", suffix="incl"))

    def test_plot_var_vtx_perf_ujets_inclusive_vertexing(self):
        """Test that png files are being created for tagger with inclusive vertexing enabled."""
        self.dummy_tagger.reference = True
        with tempfile.TemporaryDirectory() as tmp_file:
            auxresults = AuxResults(sample="test", output_dir=tmp_file)
            auxresults.add(self.dummy_tagger)
            auxresults.plot_var_vtx_perf(no_vtx_flavours=["ujets"], incl_vertexing=True)
            self.assertIsFile(auxresults.get_filename("ujets_vtx_fakes_vs_pt", suffix="incl"))

    def test_plot_var_mass_inclusive_vertexing(self):
        """Test that png files are being created for mass reconstruction."""
        self.dummy_tagger.reference = True
        with tempfile.TemporaryDirectory() as tmp_file:
            auxresults = AuxResults(
                sample="test",
                aux_perf_vars=["pt", "eta", "dphi"],
                output_dir=tmp_file,
            )
            auxresults.add(self.dummy_tagger)
            auxresults.plot_vertex_mass(vtx_flavours=["bjets"], incl_vertexing=True)
            self.assertIsFile(auxresults.get_filename("bjets_sv_mass", suffix="incl"))
            self.assertIsFile(auxresults.get_filename("bjets_sv_mass_diff"))

    def test_plot_var_mass_exclusive_vertexing(self):
        """Test that png files are being created for mass reconstruction."""
        self.dummy_tagger.reference = True
        with tempfile.TemporaryDirectory() as tmp_file:
            auxresults = AuxResults(
                sample="test",
                aux_perf_vars=["pt", "eta", "dphi"],
                output_dir=tmp_file,
            )
            auxresults.add(self.dummy_tagger)
            auxresults.plot_vertex_mass(vtx_flavours=["bjets"], incl_vertexing=False)
            self.assertIsFile(auxresults.get_filename("bjets_sv_mass", suffix="excl"))
