from __future__ import annotations

import unittest

import numpy as np

from puma.utils.logging import logger, set_log_level
from puma.utils.truth_hadron import AssociateTracksToHadron, GetOrderedHadrons, SelectHadron

set_log_level(logger, "DEBUG")


class TruthHadronTestCase(unittest.TestCase):
    """Test class for the puma.truth_hadron functions."""

    def setUp(self):
        self.hadron_barcode = np.array([[1373, 1390, 1375, -1, -1]])
        self.parent_barcode = np.array([[-1, 1373, 1373, -1, -1]])
        self.track_parent = np.array([
            [
                1390,
                1375,
                1390,
                1375,
                -2,
                -2,
                -2,
                -2,
                -2,
                -2,
                -2,
                1375,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ]
        ])
        self.hadron_mask = np.array([[1], [1], [1], [0], [0]])

        self.hadrons = np.array([
            [
                (521, 1373, -1, 47603.895),
                (-411, 1390, 1373, 29951.072),
                (431, 1375, 1373, 15795.845),
                (-1, -1, -1, np.nan),
                (-1, -1, -1, np.nan),
            ]
        ])

    def test_hadron_order(self):
        result = GetOrderedHadrons(self.hadron_barcode, self.parent_barcode, n_max_showers=1)
        known_result = [[[0, 1, 2, -1, -1]]]

        assert (result == known_result).all(), "Test failed: result does not match known_result"
        print("Test passed!")

    def test_track_to_hadron(self):
        result_a, result_b, result_c = AssociateTracksToHadron(
            self.track_parent, self.hadron_barcode, self.hadron_mask
        )

        known_result_a = np.array([
            [
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ],
            [
                [
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ],
            [
                [
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ],
            [
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ],
            [
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ],
        ])

        known_result_b = np.array([
            [
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ])

        assert (result_a == known_result_a).all(), "Test failed: result does not match known_result"
        print("Test passed!")

        assert (result_b == known_result_b).all(), "Test failed: result does not match known_result"
        print("Test passed!")

        assert (result_c == known_result_b).all(), "Test failed: result does not match known_result"
        print("Test passed!")

        def test_select_hadron(self):
            result = SelectHadron(self.hadrons, np.array([[1]]))

            known_result = np.array([[[-411, 1390, 1373, 29951.072]]])

            assert (result == known_result).all(), "Test failed: result does not match known_result"
            print("Test passed!")
