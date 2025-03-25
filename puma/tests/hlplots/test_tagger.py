"""Unit test script for the functions in hlplots/tagger.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from ftag import Cuts
from numpy.lib.recfunctions import structured_to_unstructured as s2u
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from puma.hlplots import Tagger
from puma.utils import logger, set_log_level

set_log_level(logger, "DEBUG")


class TaggerBasisTestCase(unittest.TestCase):
    """Test class for the Tagger class."""

    def test_wrong_flavour_for_category(self):
        """Test value error if a flavour is defined which is not supported in the category."""
        with self.assertRaises(ValueError):
            Tagger(
                name="test",
                category="single-btag",
                output_flavours=["bjets", "cjets", "ujets", "hbb"],
            )

    def test_empty_string_tagger_name(self):
        """Test empty string as model name."""
        tagger = Tagger("")
        self.assertEqual(tagger.name, "")

    def test_more_tagger_args(self):
        tagger = Tagger("dummy", sample_path="dummy_path", cuts=[("pt", ">", 20)])
        assert isinstance(tagger.sample_path, Path)
        assert isinstance(tagger.cuts, Cuts)
        assert repr(tagger) == "dummy (dummy)"

    def test_wrong_template(self):
        """Test wrong template."""
        template_wrong = {"test": 1}
        with self.assertRaises(TypeError):
            Tagger("dummy", **template_wrong)  # pylint: disable=W0212,E1123

    def test_label_template(self):
        """Test template with label."""
        template_label = {"label": 1.5}
        tagger = Tagger("dummy", **template_label)  # pylint: disable=W0212
        self.assertEqual(tagger.label, 1.5)

    def test_n_jets(self):
        """Test if number of n_jets correctly calculated."""
        tagger = Tagger("dummy", output_flavours=["ujets", "cjets", "bjets"])
        labels = np.concatenate([np.zeros(80), np.ones(5) * 4, np.ones(15) * 5])
        tagger.labels = np.array(labels, dtype=[("HadronConeExclTruthLabelID", "i4")])
        with self.subTest():
            self.assertEqual(tagger.n_jets("ujets"), 80)
        with self.subTest():
            self.assertEqual(tagger.n_jets("cjets"), 5)
        with self.subTest():
            self.assertEqual(tagger.n_jets("bjets"), 15)
        assert np.sum(tagger.is_flav("ujets")) == 3160


class TaggerScoreExtractionTestCase(unittest.TestCase):
    """Test extract_tagger_scores function in Tagger class."""

    def setUp(self) -> None:
        """Set up for tests."""
        self.df_dummy = pd.DataFrame({
            "dummy_pc": np.zeros(10),
            "dummy_pu": np.ones(10),
            "dummy_pb": np.zeros(10),
        })
        self.scores_expected = np.column_stack((np.ones(10), np.zeros(10), np.zeros(10)))

    def test_wrong_source_type(self):
        """Test using wrong source type."""
        tagger = Tagger("dummy")
        with self.assertRaises(ValueError):
            tagger.extract_tagger_scores(self.df_dummy, source_type="dummy")

    def test_data_frame_path_no_key(self):
        """Test passing data frame path but no key."""
        tagger = Tagger("dummy")
        with self.assertRaises(ValueError):
            tagger.extract_tagger_scores(self.df_dummy, source_type="data_frame_path")

    def test_data_frame(self):
        """Test passing data frame."""
        tagger = Tagger("dummy", output_flavours=["ujets", "cjets", "bjets"])
        tagger.extract_tagger_scores(self.df_dummy)
        np.testing.assert_array_equal(s2u(tagger.scores), self.scores_expected)

    def test_data_frame_path(self):
        """Test passing data frame path."""
        tagger = Tagger("dummy", output_flavours=["ujets", "cjets", "bjets"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_name = f"{tmp_dir}/dummy_df.h5"
            self.df_dummy.to_hdf(file_name, key="dummy_tagger")

            tagger.extract_tagger_scores(
                file_name, key="dummy_tagger", source_type="data_frame_path"
            )
        np.testing.assert_array_equal(s2u(tagger.scores), self.scores_expected)

    def test_h5_structured_numpy_path(self):
        """Test passing structured h5 path."""
        tagger = Tagger("dummy", output_flavours=["ujets", "cjets", "bjets"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_name = f"{tmp_dir}/dummy_df.h5"
            with h5py.File(file_name, "w") as f_h5:
                f_h5.create_dataset(data=self.df_dummy.to_records(), name="dummy_tagger")

            tagger.extract_tagger_scores(file_name, key="dummy_tagger", source_type="h5_file")
        np.testing.assert_array_equal(s2u(tagger.scores), self.scores_expected)

    def test_structured_array(self):
        """Test passing structured numpy array."""
        tagger = Tagger("dummy", output_flavours=["ujets", "cjets", "bjets"])
        tagger.extract_tagger_scores(
            self.df_dummy.to_records(),
            key="dummy_tagger",
            source_type="structured_array",
        )
        np.testing.assert_array_equal(s2u(tagger.scores), self.scores_expected)


class TaggerTestCase(unittest.TestCase):
    """Test class for the Tagger class."""

    def setUp(self) -> None:
        """Set up for tests."""
        scores = np.column_stack((np.ones(10), np.ones(10), np.ones(10)))
        self.scores = u2s(
            scores,
            dtype=[("dummy_pu", "f4"), ("dummy_pc", "f4"), ("dummy_pb", "f4")],
        )

    def test_errors(self):
        tagger = Tagger("dummy")
        tagger.scores = self.scores
        with self.assertRaises(ValueError):
            tagger.discriminant("qcd")
        with self.assertRaises(ValueError):
            tagger.discriminant(signal="bjets", fxs={"fc": 0.5})

    def test_disc_b_calc(self):
        """Test b-disc calculation."""
        tagger = Tagger(
            "dummy",
            fxs={"fc": 0.5},
            output_flavours=["ujets", "cjets", "bjets"],
        )
        tagger.scores = self.scores
        discs = tagger.discriminant("bjets")
        np.testing.assert_array_equal(discs, np.zeros(10))

    def test_disc_c_calc(self):
        """Test c-disc calculation."""
        tagger = Tagger(
            "dummy",
            fxs={"fb": 0.5},
            output_flavours=["ujets", "cjets", "bjets"],
        )
        tagger.scores = self.scores
        discs = tagger.discriminant("cjets")
        np.testing.assert_array_equal(discs, np.zeros(10))

    def test_disc_hbb_calc(self):
        """Test hbb-disc calculation."""
        from ftag import Flavours as F

        tagger = Tagger(
            "dummy",
            fxs={"fhcc": 0.1, "ftop": 0.1},
            output_flavours=[F["hbb"], F["hcc"], F["top"], F["qcd"]],
            category="xbb",
        )
        tagger.scores = u2s(
            np.column_stack((np.ones(10) * 2, np.ones(10), np.ones(10), np.ones(10))),
            dtype=[
                ("dummy_phbb", "f4"),
                ("dummy_phcc", "f4"),
                ("dummy_ptop", "f4"),
                ("dummy_pqcd", "f4"),
            ],
        )
        discs = tagger.discriminant("hbb")
        np.testing.assert_array_almost_equal(discs, np.ones([10]) * 0.693147)


class TaggerAuxTaskTestCase(unittest.TestCase):
    """Test class for aux task functionality in Tagger class."""

    def test_aux_variables_vertexing(self):
        """Test vertexing aux task variable retrieval. Includes special cases for SV1
        and JF taggers.
        """
        default_tagger = Tagger("dummy", aux_tasks=["vertexing"])
        SV1_tagger = Tagger("SV1", aux_tasks=["vertexing"])
        JF_tagger = Tagger("JF", aux_tasks=["vertexing"])
        self.assertEqual(default_tagger.aux_variables["vertexing"], "dummy_aux_VertexIndex")
        self.assertEqual(SV1_tagger.aux_variables["vertexing"], "SV1VertexIndex")
        self.assertEqual(JF_tagger.aux_variables["vertexing"], "JFVertexIndex")

    def test_aux_variables_track_origin(self):
        """Test track_origin aux task variable retrieval."""
        tagger = Tagger("dummy", aux_tasks=["track_origin"])
        self.assertEqual(tagger.aux_variables["track_origin"], "dummy_aux_TrackOrigin")

    def test_aux_variables_undefined(self):
        """Test undefined aux task variable retrieval."""
        tagger = Tagger("dummy", aux_tasks=["dummy"])
        with self.assertRaises(ValueError):
            tagger.aux_variables["dummy"]


class TaggerProbsTestCase(unittest.TestCase):
    """Test class for the probs method in Tagger class."""

    def setUp(self) -> None:
        """Set up for tests."""
        scores = np.column_stack((np.ones(10), np.zeros(10), np.ones(10) * 5))
        self.scores = u2s(
            scores,
            dtype=[("dummy_pu", "f4"), ("dummy_pc", "f4"), ("dummy_pb", "f4")],
        )
        self.tagger = Tagger("dummy")
        self.tagger.scores = self.scores
        self.tagger.labels = np.array(
            np.zeros(len(self.scores)), dtype=[("HadronConeExclTruthLabelID", "i4")]
        )

    def test_probs_without_label_flavour(self):
        """Test probs method without label_flavour parameter."""
        prob_flavour = "bjets"
        expected_probs = np.ones(10) * 5
        np.testing.assert_array_equal(self.tagger.probs(prob_flavour), expected_probs)

    def test_probs_with_label_flavour(self):
        """Test probs method with label_flavour parameter."""
        prob_flavour = "bjets"
        label_flavour = "cjets"
        expected_probs = np.zeros(0)
        np.testing.assert_array_equal(
            self.tagger.probs(prob_flavour, label_flavour), expected_probs
        )
        prob_flavour = "cjets"
        label_flavour = "ujets"
        expected_probs = np.zeros(10)
        np.testing.assert_array_equal(
            self.tagger.probs(prob_flavour, label_flavour), expected_probs
        )
