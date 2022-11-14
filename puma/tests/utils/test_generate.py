#!/usr/bin/env python
"""
Unit test script for the functions in utils/generate.py
"""

import unittest

import numpy as np

from puma.utils import logger, set_log_level
from puma.utils.generate import get_dummy_2_taggers, get_dummy_multiclass_scores

set_log_level(logger, "DEBUG")


class GetDummyMulticlassScoresTestCase(unittest.TestCase):
    """Tes case fot get_dummy_multiclass_scores function."""

    def test_size(self):
        """Check that correct size is returned."""
        output, labels = get_dummy_multiclass_scores(size=10)
        # we expect here 9 entries, since 10 is not dividable by 3, and the function
        # returns 3 classes with the same amount of stats per class
        with self.subTest("output length"):
            self.assertEqual(len(output), 9)
        with self.subTest("label length"):
            self.assertEqual(len(labels), 9)

    def test_range(self):
        """Check that correct range of output is returned"""
        output, _ = get_dummy_multiclass_scores()
        with self.subTest("max val"):
            self.assertLessEqual(np.max(output), 1)
        with self.subTest("min val"):
            self.assertGreaterEqual(np.min(output), 0)


class GetDummy2TaggersTestCase(unittest.TestCase):
    """Tes case fot get_dummy_2_taggers function."""

    def test_size(self):
        """Check that correct size is returned."""
        df_gen = get_dummy_2_taggers(size=10)
        # we expect here 9 entries, since 10 is not dividable by 3, and the function
        # returns 3 classes with the same amount of stats per class
        self.assertEqual(len(df_gen), 9)

    def test_columns(self):
        """Check correct amount of columns."""
        df_gen = get_dummy_2_taggers()
        self.assertEqual(len(df_gen.columns.values), 7)

    def test_columns_pt(self):
        """Check correct amount of columns when using pt as well."""
        df_gen = get_dummy_2_taggers(add_pt=True)
        self.assertEqual(len(df_gen.columns.values), 8)
