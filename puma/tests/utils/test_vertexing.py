#!/usr/bin/env python

"""Unit test script for the functions in utils/vertexing.py."""
from __future__ import annotations

import unittest

import numpy as np

from puma.utils import logger, set_log_level
from puma.utils.vertexing import (
    associate_vertices,
    build_vertices,
    calculate_vertex_metrics,
)

set_log_level(logger, "DEBUG")


class BuildVerticesTestCase(unittest.TestCase):
    """Test case for build_vertices function."""

    def test_no_vertices(self):
        """Check case where there are no vertices."""
        indices = np.array([0, 1, 2, 3, 4])
        vertices = build_vertices(indices)
        self.assertEqual(vertices.shape, (0, indices.shape[0]))

    def test_simple_case(self):
        """Check simple case with one vertex."""
        indices = np.array([0, 1, 2, 2, 3])
        expected_result = np.array([[False, False, True, True, False]])
        vertices = build_vertices(indices)
        np.testing.assert_array_equal(vertices, expected_result)

    def test_complex_case(self):
        """Check complex case with multiple vertices."""
        indices = np.array([0, 1, 2, 2, 3, 1, 3, 3, 4, 1])
        expected_result = np.array(
            [
                [False, True, False, False, False, True, False, False, False, True],
                [False, False, True, True, False, False, False, False, False, False],
                [False, False, False, False, True, False, True, True, False, False],
            ]
        )
        vertices = build_vertices(indices)
        np.testing.assert_array_equal(vertices, expected_result)

    def test_arbitrary_indices(self):
        """Check case where vertex indices are arbitrary."""
        indices = np.array([22, 19, 22, 103, 103])
        expected_result = np.array(
            [[True, False, True, False, False], [False, False, False, True, True]]
        )
        vertices = build_vertices(indices)
        np.testing.assert_array_equal(vertices, expected_result)

    def test_ignore_negatives(self):
        """Check case where negative indices are present."""
        indices = np.array([0, -2, 0, -2, -1, -1, -1])
        expected_result = np.array([[True, False, True, False, False, False, False]])
        vertices = build_vertices(indices)
        np.testing.assert_array_equal(vertices, expected_result)

    def test_ignore_indices(self):
        """Check case where ignore_indices is given."""
        indices = np.array([0, 0, 1, 2, 1, 2])
        expected_result = np.array([[False, False, False, True, False, True]])
        vertices = build_vertices(indices, ignore_indices=[0, 1])
        np.testing.assert_array_equal(vertices, expected_result)


class AssociateVerticesTestCase(unittest.TestCase):
    """Test case for associate_vertices function."""

    def test_no_associations(self):
        """Check case where there are no associations to make."""
        vertices1 = np.array(
            [
                [True, True, False, False, False],
            ]
        )
        vertices2 = np.array(
            [
                [False, False, True, True, False],
            ]
        )
        expected_assoc = np.array([[False]])
        expected_common = np.array([[0]])
        assoc, common = associate_vertices(vertices1, vertices2)
        np.testing.assert_array_equal(assoc, expected_assoc)
        np.testing.assert_array_equal(common, expected_common)

    def test_association_condition1(self):
        """Check case where associations are made based on most common tracks."""
        vertices1 = np.array(
            [
                [True, False, True, False, False, False, True],
                [False, True, False, True, False, False, False],
            ]
        )
        vertices2 = np.array(
            [
                [True, True, True, False, False, False, False],
            ]
        )
        expected_assoc = np.array(
            [
                [True],
                [False],
            ]
        )
        expected_common = np.array([[2], [1]])
        assoc, common = associate_vertices(vertices1, vertices2)
        np.testing.assert_array_equal(assoc, expected_assoc)
        np.testing.assert_array_equal(common, expected_common)

    def test_association_condition2(self):
        """Check case where associations are made based on
        highest efficiency (tiebreaker 1).
        """
        vertices1 = np.array(
            [
                [True, False, True, False, False, False, True],
                [False, True, False, True, False, False, False],
            ]
        )
        vertices2 = np.array(
            [
                [True, True, True, True, False, False, False],
            ]
        )
        expected_assoc = np.array(
            [
                [False],
                [True],
            ]
        )
        expected_common = np.array([[2], [2]])
        assoc, common = associate_vertices(vertices1, vertices2)
        np.testing.assert_array_equal(assoc, expected_assoc)
        np.testing.assert_array_equal(common, expected_common)

    def test_association_condition3(self):
        """Check case where associations are made based on
        lowest fake rate (tiebreaker 2).
        """
        vertices1 = np.array(
            [
                [True, False, True, False, True, False, True],
            ]
        )
        vertices2 = np.array(
            [
                [True, False, True, False, False, False, False],
                [False, True, False, True, True, False, True],
            ]
        )
        expected_assoc = np.array([[True, False]])
        expected_common = np.array([[2, 2]])
        assoc, common = associate_vertices(vertices1, vertices2)
        np.testing.assert_array_equal(assoc, expected_assoc)
        np.testing.assert_array_equal(common, expected_common)

    def test_association_condition4(self):
        """Check case where associations are made based on
        first match (tiebreaker 3).
        """
        vertices1 = np.array(
            [
                [True, False, True, False, False, False, False],
                [False, True, False, True, False, False, False],
            ]
        )
        vertices2 = np.array(
            [
                [True, True, True, True, False, False, False],
            ]
        )
        expected_assoc = np.array(
            [
                [True],
                [False],
            ]
        )
        expected_common = np.array([[2], [2]])
        assoc, common = associate_vertices(vertices1, vertices2)
        np.testing.assert_array_equal(assoc, expected_assoc)
        np.testing.assert_array_equal(common, expected_common)

    def test_full_associations(self):
        """Check case where associations are a perfect match."""
        vertices1 = np.array(
            [
                [True, True, False, False, False],
            ]
        )
        vertices2 = np.array(
            [
                [True, True, False, False, False],
            ]
        )
        expected_assoc = np.array([[True]])
        expected_common = np.array([[2]])
        assoc, common = associate_vertices(vertices1, vertices2)
        np.testing.assert_array_equal(assoc, expected_assoc)
        np.testing.assert_array_equal(common, expected_common)


class CalculateVertexMetricsTestCase(unittest.TestCase):
    """Test case for calculate_vertex_metrics function."""

    def test_no_vertices(self):
        """Check case where there are no vertices."""
        indices1 = np.array([[0, 1, 2, 3, 4]])
        indices2 = np.array([[0, 1, 2, 3, 4]])
        metrics = calculate_vertex_metrics(indices1, indices2, max_vertices=2)
        np.testing.assert_array_equal(metrics["n_match"], [0])
        np.testing.assert_array_equal(metrics["n_test"], [0])
        np.testing.assert_array_equal(metrics["n_ref"], [0])
        np.testing.assert_array_equal(metrics["track_overlap"], [[-1, -1]])
        np.testing.assert_array_equal(metrics["test_vertex_size"], [[-1, -1]])
        np.testing.assert_array_equal(metrics["ref_vertex_size"], [[-1, -1]])

    def test_no_reco_vertices(self):
        """Check case where there are no reco vertices."""
        indices1 = np.array([[0, 1, 2, 3, 4]])
        indices2 = np.array([[0, 1, 2, 2, 3]])
        metrics = calculate_vertex_metrics(indices1, indices2, max_vertices=2)
        np.testing.assert_array_equal(metrics["n_match"], [0])
        np.testing.assert_array_equal(metrics["n_test"], [0])
        np.testing.assert_array_equal(metrics["n_ref"], [1])
        np.testing.assert_array_equal(metrics["track_overlap"], [[-1, -1]])
        np.testing.assert_array_equal(metrics["test_vertex_size"], [[-1, -1]])
        np.testing.assert_array_equal(metrics["ref_vertex_size"], [[-1, -1]])

    def test_no_truth_vertices(self):
        """Check case where there are no truth vertices."""
        indices1 = np.array([[0, 0, 1, 2, 3]])
        indices2 = np.array([[0, 1, 2, 3, 4]])
        metrics = calculate_vertex_metrics(indices1, indices2, max_vertices=2)
        np.testing.assert_array_equal(metrics["n_match"], [0])
        np.testing.assert_array_equal(metrics["n_test"], [1])
        np.testing.assert_array_equal(metrics["n_ref"], [0])
        np.testing.assert_array_equal(metrics["track_overlap"], [[-1, -1]])
        np.testing.assert_array_equal(metrics["test_vertex_size"], [[-1, -1]])
        np.testing.assert_array_equal(metrics["ref_vertex_size"], [[-1, -1]])

    def test_no_match(self):
        """Check case where there are no vertex matches."""
        indices1 = np.array([[0, 0, 1, 2, 3]])
        indices2 = np.array([[0, 1, 2, 2, 3]])
        metrics = calculate_vertex_metrics(indices1, indices2, max_vertices=2)
        np.testing.assert_array_equal(metrics["n_match"], [0])
        np.testing.assert_array_equal(metrics["n_test"], [1])
        np.testing.assert_array_equal(metrics["n_ref"], [1])
        np.testing.assert_array_equal(metrics["track_overlap"], [[-1, -1]])
        np.testing.assert_array_equal(metrics["test_vertex_size"], [[-1, -1]])
        np.testing.assert_array_equal(metrics["ref_vertex_size"], [[-1, -1]])

    def test_one_match(self):
        """Check case where there is one vertex match."""
        indices1 = np.array([[0, 0, 1, 2, 2]])
        indices2 = np.array([[0, 0, 0, 1, 2]])
        metrics = calculate_vertex_metrics(indices1, indices2, max_vertices=2)
        np.testing.assert_array_equal(metrics["n_match"], [1])
        np.testing.assert_array_equal(metrics["n_test"], [2])
        np.testing.assert_array_equal(metrics["n_ref"], [1])
        np.testing.assert_array_equal(metrics["track_overlap"], [[2, -1]])
        np.testing.assert_array_equal(metrics["test_vertex_size"], [[2, -1]])
        np.testing.assert_array_equal(metrics["ref_vertex_size"], [[3, -1]])

    def test_mult_matches(self):
        """Check case wehre there are multiple vertex matches."""
        indices1 = np.array([[0, 0, 1, 1, 0, 1, 2, 2, 3]])
        indices2 = np.array([[0, 1, 0, 2, 0, 2, 3, 3, 1]])
        metrics = calculate_vertex_metrics(indices1, indices2, max_vertices=3)
        np.testing.assert_array_equal(metrics["n_match"], [3])
        np.testing.assert_array_equal(metrics["n_test"], [3])
        np.testing.assert_array_equal(metrics["n_ref"], [4])
        np.testing.assert_array_equal(metrics["track_overlap"], [[2, 2, 2]])
        np.testing.assert_array_equal(metrics["test_vertex_size"], [[3, 3, 2]])
        np.testing.assert_array_equal(metrics["ref_vertex_size"], [[3, 2, 2]])
