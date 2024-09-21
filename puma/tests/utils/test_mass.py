"""Unit test script for the functions in utils/vertexing.py."""

from __future__ import annotations

import unittest

import numpy as np

from puma.utils import logger, set_log_level
from puma.utils.mass import (
    calculate_vertex_mass,
)

set_log_level(logger, "DEBUG")


class VertexMassTestCase(unittest.TestCase):
    """Test case for clean_indices function."""

    def test_output_shape(self):
        """Check that output has the right shape."""
        track_pt = np.zeros((5, 10))
        track_eta = np.zeros((5, 10))
        track_phi = np.zeros((5, 10))
        vtx_idx = np.zeros((5, 10))
        vtx_masses = calculate_vertex_mass(track_pt, track_eta, track_phi, vtx_idx)
        expected_shape = (5, 10)
        np.testing.assert_array_equal(vtx_masses.shape, expected_shape)

    def test_single_track_only(self):
        """Check case where only single track vertices exist."""
        track_pt = np.zeros((1, 5))
        track_eta = np.zeros((1, 5))
        track_phi = np.zeros((1, 5))
        vtx_idx = np.array([[0, 1, 2, 3, 4]])
        vtx_masses = calculate_vertex_mass(track_pt, track_eta, track_phi, vtx_idx, particle_mass=2)
        expected_result = np.array([[2, 2, 2, 2, 2]])
        np.testing.assert_array_equal(vtx_masses, expected_result)

    def test_multiple_vertices(self):
        """Check more complicated case with two vertices."""
        track_pt = np.random.rand(1, 5) * 1000
        track_eta = np.random.rand(1, 5) * 2
        track_phi = np.random.rand(1, 5) * 2 * np.pi - np.pi
        vtx_idx = np.array([[0, 0, 1, 1, 1]])

        px = track_pt * np.cos(track_phi)
        py = track_pt * np.sin(track_phi)
        pz = track_pt * np.sinh(track_eta)
        e = np.sqrt(px**2 + py**2 + pz**2 + 0.13957**2)

        vtx_mass_1 = np.sqrt(
            (e[0, 0] + e[0, 1]) ** 2
            - (px[0, 0] + px[0, 1]) ** 2
            - (py[0, 0] + py[0, 1]) ** 2
            - (pz[0, 0] + pz[0, 1]) ** 2
        )
        vtx_mass_2 = np.sqrt(
            (e[0, 2] + e[0, 3] + e[0, 4]) ** 2
            - (px[0, 2] + px[0, 3] + px[0, 4]) ** 2
            - (py[0, 2] + py[0, 3] + py[0, 4]) ** 2
            - (pz[0, 2] + pz[0, 3] + pz[0, 4]) ** 2
        )

        vtx_masses = calculate_vertex_mass(
            track_pt, track_eta, track_phi, vtx_idx, particle_mass=0.13957
        )
        expected_result = np.array([[vtx_mass_1, vtx_mass_1, vtx_mass_2, vtx_mass_2, vtx_mass_2]])
        np.testing.assert_array_equal(vtx_masses, expected_result)
