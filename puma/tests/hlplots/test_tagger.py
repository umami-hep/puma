#!/usr/bin/env python

"""
Unit test script for the functions in hlplots/tagger.py
"""

# import os
# import tempfile
import unittest

import numpy as np

from puma.hlplots import Tagger
from puma.utils import logger, set_log_level

set_log_level(logger, "DEBUG")


class TaggerTestCase(unittest.TestCase):
    """Test class for the puma.histogram functions."""

    def test_empty_string_tagger_name(self):
        """Test empty string as model name."""
        tagger = Tagger("")
        self.assertEqual(tagger.model_name, "")

    def test_wrong_template(self):
        """Test wrong template."""
        template_wrong = {"test": 1}
        with self.assertRaises(KeyError):
            Tagger("dummy", template=template_wrong)

    def test_disc_cut_template(self):
        """Test template with disc_cut."""
        template_disc_cut = {"disc_cut": 1.5}
        tagger = Tagger("dummy", template=template_disc_cut)
        self.assertEqual(tagger.disc_cut, 1.5)

    def test_n_jets(self):
        """Test if number of n_jets jets correctly calculated."""
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

    def test_empty_discs_rej_calc(self):
        """Test calculation of rejection without discs specified."""
        tagger = Tagger("dummy")
        with self.assertRaises(ValueError):
            tagger.calc_rej(np.linspace(0.5, 1, 10))

    def test_calc_rej_one(self):
        """Test rejection."""
        tagger = Tagger("dummy")
        rng = np.random.default_rng(seed=22)
        tagger.discs = rng.poisson(size=100)
        tagger.is_light = np.concatenate(
            [np.ones(80), np.zeros(5), np.zeros(15)]
        ).astype(bool)
        tagger.is_c = np.concatenate([np.zeros(80), np.ones(5), np.zeros(15)]).astype(
            bool
        )
        tagger.is_b = np.concatenate([np.zeros(80), np.zeros(5), np.ones(15)]).astype(
            bool
        )
        tagger.calc_rej(1.0)
        self.assertAlmostEqual(tagger.ujets_rej, 1.57, places=2)
