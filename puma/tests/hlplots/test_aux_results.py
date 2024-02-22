"""Unit test script for the functions in hlplots/aux_results.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from puma.hlplots import AuxResults
from puma.hlplots.tagger import Tagger
from puma.utils import get_dummy_tagger_aux, logger, set_log_level

set_log_level(logger, "DEBUG")


class AuxResultsTestCase(unittest.TestCase):
    """Test class for the AuxResults class."""

    def test_add_taggers_from_file(self):
        """Test for AuxResults.add_taggers_from_file function."""
        np.random.default_rng(seed=16)
        fname = get_dummy_tagger_aux()[0]
        results = AuxResults(sample="test")
        taggers = [Tagger("GN2")]
        results.add_taggers_from_file(
            taggers,
            fname,
        )
        self.assertEqual(list(results.taggers.values()), taggers)

    def test_add_taggers_with_cuts(self):
        np.random.default_rng(seed=16)
        fname = get_dummy_tagger_aux()[0]
        cuts = [("eta", ">", 0)]
        tagger_cuts = [("pt", ">", 20)]
        results = AuxResults(sample="test")
        taggers = [Tagger("GN2", cuts=tagger_cuts)]
        results.add_taggers_from_file(
            taggers,
            fname,
            cuts=cuts,
        )
        self.assertEqual(list(results.taggers.values()), taggers)

    def test_add_taggers_with_cuts_override_perf_vars(self):
        rng = np.random.default_rng(seed=16)
        fname = get_dummy_tagger_aux()[0]
        cuts = [("eta", ">", 0)]
        tagger_cuts = [("pt", ">", 20)]
        results = AuxResults(sample="test")
        taggers = [Tagger("GN2", cuts=tagger_cuts)]
        results.add_taggers_from_file(
            taggers,
            fname,
            cuts=cuts,
            perf_vars={
                "pt": rng.exponential(100, size=9999),
                "eta": rng.normal(0, 1, size=9999),
            },
        )
        self.assertEqual(list(results.taggers.values()), taggers)


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
            "vertexing": f["tracks"]["GN2_VertexIndex"],
            "track_origin": f["tracks"]["GN2_TrackOrigin"],
        }
        dummy_tagger.aux_labels = {
            "vertexing": f["tracks"]["ftagTruthVertexIndex"],
            "track_origin": f["tracks"]["ftagTruthOriginLabel"],
        }
        dummy_tagger.perf_vars = {"pt": f["jets"]["pt"]}
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

    def test_plot_var_vtx_perf_alljets(self):
        """Test that png files are being created for tagger with aux tasks."""
        self.dummy_tagger.reference = True
        with tempfile.TemporaryDirectory() as tmp_file:
            auxresults = AuxResults(sample="test", output_dir=tmp_file)
            auxresults.add(self.dummy_tagger)
            auxresults.plot_var_vtx_perf()
            self.assertIsFile(auxresults.get_filename("alljets_vtx_eff_vs_pt"))
            self.assertIsFile(auxresults.get_filename("alljets_vtx_purity_vs_pt"))
            self.assertIsFile(auxresults.get_filename("alljets_vtx_nreco_vs_pt"))
            self.assertIsFile(auxresults.get_filename("alljets_vtx_trk_eff_vs_pt"))
            self.assertIsFile(auxresults.get_filename("alljets_vtx_trk_purity_vs_pt"))

    def test_plot_var_vtx_perf_bjets(self):
        """Test that png files are being created for tagger with aux tasks."""
        self.dummy_tagger.reference = True
        with tempfile.TemporaryDirectory() as tmp_file:
            auxresults = AuxResults(sample="test", output_dir=tmp_file)
            auxresults.add(self.dummy_tagger)
            auxresults.plot_var_vtx_perf(flavour="bjets")
            self.assertIsFile(auxresults.get_filename("bjets_vtx_eff_vs_pt"))
            self.assertIsFile(auxresults.get_filename("bjets_vtx_purity_vs_pt"))
            self.assertIsFile(auxresults.get_filename("bjets_vtx_nreco_vs_pt"))
            self.assertIsFile(auxresults.get_filename("bjets_vtx_trk_eff_vs_pt"))
            self.assertIsFile(auxresults.get_filename("bjets_vtx_trk_purity_vs_pt"))

    def test_plot_var_vtx_perf_empty(self):
        """Test vertexing performance function with empty data."""
        self.dummy_tagger_no_aux.reference = True
        with tempfile.TemporaryDirectory() as tmp_file:
            auxresults = AuxResults(sample="test", output_dir=tmp_file)
            auxresults.add(self.dummy_tagger_no_aux)
            with self.assertRaises(ValueError):
                auxresults.plot_var_vtx_perf()
