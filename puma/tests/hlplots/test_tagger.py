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

    def setUp(self) -> None:
        self.rng = np.random.default_rng(seed=22)
        self.discs = self.rng.poisson(size=100)

    def test_empty_tagger_name(self):
        """Test if providing no model name raises TypeError"""
        with self.assertRaises(TypeError):
            Tagger()

    def test_empty_string_tagger_name(self):
        """Test empty string as model name."""
        tagger = Tagger("")
        self.assertEqual(tagger.model_name, "")

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

    # def test_no_fc_calc_disc(self):
    #     """Test empty string as model name."""
    #     tagger = Tagger("tagger")
    #     tagger.discs = self.discs
    #     tagger.calc_discs()
