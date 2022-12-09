#!/usr/bin/env python

"""
Unit test script for the functions in hlplots/tagger.py
"""
# pylint: disable=no-self-use

import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd
from testfixtures import LogCapture

from puma.hlplots import Tagger
from puma.utils import logger, set_log_level

set_log_level(logger, "DEBUG")


class TaggerBasisTestCase(unittest.TestCase):
    """Test class for the Tagger class."""

    def test_empty_string_tagger_name(self):
        """Test empty string as model name."""
        tagger = Tagger("")
        self.assertEqual(tagger.model_name, "")

    def test_wrong_template(self):
        """Test wrong template."""
        template_wrong = {"test": 1}
        tagger = Tagger("dummy")
        with self.assertRaises(KeyError):
            tagger._init_from_template(template=template_wrong)  # pylint: disable=W0212

    def test_label_template(self):
        """Test template with label."""
        template_label = {"label": 1.5}
        tagger = Tagger("dummy")
        tagger._init_from_template(template=template_label)  # pylint: disable=W0212
        self.assertEqual(tagger.label, 1.5)

    def test_none_template(self):
        """Test None template."""
        tagger = Tagger("dummy")
        with LogCapture("puma") as log:
            tagger._init_from_template(template=None)  # pylint: disable=W0212
            log.check(
                (
                    "puma",
                    "DEBUG",
                    "Template initialised with template being `None` - not "
                    "doing anything.",
                )
            )

    def test_n_jets(self):
        """Test if number of n_jets correctly calculated."""
        tagger = Tagger("dummy")
        tagger.is_light = np.concatenate([np.ones(80), np.zeros(5), np.zeros(15)])
        tagger.is_c = np.concatenate([np.zeros(80), np.ones(5), np.zeros(15)])
        tagger.is_b = np.concatenate([np.zeros(80), np.zeros(5), np.ones(15)])
        with self.subTest():
            self.assertEqual(tagger.n_jets_light, 80)
        with self.subTest():
            self.assertEqual(tagger.n_jets_c, 5)
        with self.subTest():
            self.assertEqual(tagger.n_jets_b, 15)


class TaggerScoreExtractionTestCase(unittest.TestCase):
    """Test extract_tagger_scores function in Tagger class."""

    def setUp(self) -> None:
        """Set up for tests."""
        self.df_dummy = pd.DataFrame(
            {
                "dummy_pc": np.zeros(10),
                "dummy_pu": np.ones(10),
                "dummy_pb": np.zeros(10),
            }
        )
        self.scores_expected = np.column_stack(
            (np.ones(10), np.zeros(10), np.zeros(10))
        )

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
        tagger = Tagger("dummy")
        tagger.extract_tagger_scores(self.df_dummy)
        np.testing.assert_array_equal(tagger.scores, self.scores_expected)

    def test_data_frame_path(self):
        """Test passing data frame path."""

        tagger = Tagger("dummy")
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_name = f"{tmp_dir}/dummy_df.h5"
            self.df_dummy.to_hdf(file_name, key="dummy_tagger")

            tagger.extract_tagger_scores(
                file_name, key="dummy_tagger", source_type="data_frame_path"
            )
        np.testing.assert_array_equal(tagger.scores, self.scores_expected)

    def test_h5_structured_numpy_path(self):
        """Test passing data frame path."""

        tagger = Tagger("dummy")
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_name = f"{tmp_dir}/dummy_df.h5"
            with h5py.File(file_name, "w") as f_h5:
                f_h5.create_dataset(
                    data=self.df_dummy.to_records(), name="dummy_tagger"
                )

            tagger.extract_tagger_scores(
                file_name, key="dummy_tagger", source_type="numpy_structured"
            )
        np.testing.assert_array_equal(tagger.scores, self.scores_expected)


class TaggerTestCase(unittest.TestCase):
    """Test class for the Tagger class."""

    def setUp(self) -> None:
        """Set up for tests."""
        self.scores = np.column_stack((np.ones(10), np.ones(10), np.ones(10)))

    def test_disc_cut_template(self):
        """Test template with disc_cut."""
        template_disc_cut = {"disc_cut": 1.5}
        tagger = Tagger("dummy", template=template_disc_cut)
        self.assertEqual(tagger.disc_cut, 1.5)

    def test_disc_b_calc_no_fc(self):
        """Test b-disc calculation w/o f_c provided."""
        tagger = Tagger("dummy")
        tagger.scores = self.scores

        with self.assertRaises(ValueError):
            tagger.calc_disc_b()

    def test_disc_b_calc(self):
        """Test b-disc calculation."""
        tagger = Tagger("dummy")
        tagger.scores = self.scores
        tagger.f_c = 0.5
        discs = tagger.calc_disc_b()

        np.testing.assert_array_equal(discs, np.zeros(10))

    def test_sig_b_calc_no_fb(self):
        """Test c-disc calculation w/o f_c provided."""
        tagger = Tagger("dummy")
        tagger.scores = self.scores
        with self.assertRaises(ValueError):
            tagger.calc_disc_c()

    def test_sig_b_calc(self):
        """Test c-disc calculation."""
        tagger = Tagger("dummy")
        tagger.scores = self.scores
        tagger.f_b = 0.5
        discs = tagger.calc_disc_c()

        np.testing.assert_array_equal(discs, np.zeros(10))
