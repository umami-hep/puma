from __future__ import annotations

import unittest

import numpy as np

from puma.utils import logger, set_log_level
from puma.utils.precision_recall_scores import precision_recall_scores_per_class

set_log_level(logger, "DEBUG")


class ConfusionMatrixTestCase(unittest.TestCase):
    def test_precision_unweighted(self):
        targets = np.array([2, 0, 2, 2, 0, 1])
        predictions = np.array([0, 0, 2, 2, 0, 2])
        precision, recall = precision_recall_scores_per_class(targets, predictions)
        expected_precision = np.array([0.66666667, 0.0, 0.66666667])
        expected_recall = np.array([1.0, 0.0, 0.66666667])
        np.testing.assert_array_almost_equal(expected_precision, precision)
        np.testing.assert_array_almost_equal(expected_recall, recall)

    def test_precision_weighted(self):
        targets = np.array([2, 0, 2, 2, 0, 1])
        predictions = np.array([0, 0, 2, 2, 0, 2])
        weights = np.array([1, 0.5, 0.5, 1, 0.2, 1])
        precision, recall = precision_recall_scores_per_class(targets, predictions, weights)
        expected_precision = np.array([0.41176471, 0.0, 0.6])
        expected_recall = np.array([1.0, 0.0, 0.6])
        np.testing.assert_array_almost_equal(expected_precision, precision)
        np.testing.assert_array_almost_equal(expected_recall, recall)
