from __future__ import annotations

import unittest

import numpy as np

from puma.utils import logger, set_log_level
from puma.utils.confusion_matrix import confusion_matrix

set_log_level(logger, "DEBUG")


class ConfusionMatrixTestCase(unittest.TestCase):
    def test_confusion_matrix_unweighted(self):
        targets = np.array([2, 0, 2, 2, 0, 1])
        predictions = np.array([0, 0, 2, 2, 0, 2])
        cm = confusion_matrix(targets, predictions, normalize="all")
        expected_cm = np.array([
            [0.33333333, 0.0, 0.0],
            [0.0, 0.0, 0.16666667],
            [0.16666667, 0.0, 0.33333333],
        ])
        np.testing.assert_array_almost_equal(expected_cm, cm)

    def test_confusion_matrix_unweighted_colnorm(self):
        targets = np.array([2, 0, 2, 2, 0, 1])
        predictions = np.array([0, 0, 2, 2, 0, 1])
        cm = confusion_matrix(targets, predictions, normalize="colnorm")
        expected_cm = np.array([[0.66666667, 0.0, 0.0], [0.0, 1.0, 0.0], [0.33333333, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(expected_cm, cm)

    def test_confusion_matrix_unweighted_rownorm(self):
        targets = np.array([2, 0, 2, 2, 0, 1])
        predictions = np.array([0, 0, 2, 2, 0, 1])
        cm = confusion_matrix(targets, predictions, normalize="rownorm")
        expected_cm = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.3333333333333333, 0.0, 0.6666666666666666],
        ])
        np.testing.assert_array_almost_equal(expected_cm, cm)

    def test_confusion_matrix_unweighted_normNone(self):
        targets = np.array([2, 0, 2, 2, 0, 1])
        predictions = np.array([0, 0, 2, 2, 0, 1])
        cm = confusion_matrix(targets, predictions, normalize=None)
        expected_cm = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 2.0]])
        np.testing.assert_array_almost_equal(expected_cm, cm)

    def test_confusion_matrix_weighted(self):
        targets = np.array([2, 0, 2, 2, 0, 1])
        predictions = np.array([0, 0, 2, 2, 0, 2])
        weights = np.array([1, 0.5, 0.5, 1, 0.2, 1])
        cm = confusion_matrix(targets, predictions, sample_weights=weights, normalize="all")
        expected_cm = np.array([
            [0.16666667, 0.0, 0.0],
            [0.0, 0.0, 0.23809524],
            [0.23809524, 0.0, 0.35714286],
        ])
        np.testing.assert_array_almost_equal(expected_cm, cm)
