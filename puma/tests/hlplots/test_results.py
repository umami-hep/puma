#!/usr/bin/env python
"""
Unit test script for the functions in hlplots/tagger.py
"""
import unittest

from puma.hlplots import ResultsBase  # ,Results
from puma.utils import logger, set_log_level

set_log_level(logger, "DEBUG")


class DummyTagger:  # pylint: disable=too-few-public-methods
    """Dummy implementation of the Tagger class, to avoid boiler plate."""

    def __init__(self, model_name) -> None:
        self.model_name = model_name


class ResultsBaseTestCase(unittest.TestCase):
    """Test class for the ResultsBase class."""

    def test_add_duplicated(self):
        """Test empty string as model name."""
        dummy_tagger_1 = DummyTagger("dummy")
        dummy_tagger_2 = DummyTagger("dummy")
        results = ResultsBase()
        results.add(dummy_tagger_1)
        with self.assertRaises(KeyError):
            results.add(dummy_tagger_2)

    def test_add_2_taggers(self):
        """Test empty string as model name."""
        dummy_tagger_1 = DummyTagger("dummy")
        dummy_tagger_2 = DummyTagger("dummy_2")
        results = ResultsBase()
        results.add(dummy_tagger_1)
        results.add(dummy_tagger_2)
        self.assertEqual(results.model_names, ["dummy", "dummy_2"])

    def test_get_taggers(self):
        """Test empty string as model name."""
        dummy_tagger_1 = DummyTagger("dummy")
        dummy_tagger_2 = DummyTagger("dummy_2")
        results = ResultsBase()
        results.add(dummy_tagger_1)
        results.add(dummy_tagger_2)
        retrieved_dummy_tagger_2 = results.get("dummy_2")
        self.assertEqual(retrieved_dummy_tagger_2.model_name, dummy_tagger_2.model_name)


# class ResultsTestCase(unittest.TestCase):
#     """Test class for the Results class."""

#     def test_add_duplicated(self):
#         """Test empty string as model name."""
#         dummy_tagger_1 = DummyTagger("dummy")
#         dummy_tagger_2 = DummyTagger("dummy")
#         results = ResultsBase()
#         results.add(dummy_tagger_1)
#         with self.assertRaises(KeyError):
#             results.add(dummy_tagger_2)
