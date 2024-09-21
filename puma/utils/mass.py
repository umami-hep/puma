"""Support functions for SV mass calculation."""

from __future__ import annotations

import numpy as np


def calculate_vertex_mass(pt, eta, phi, vtx_idx, particle_mass=0.13957):
    """
    Calculate the invariant mass of secondary vertices from the track 4-momenta,
    assuming a pion mass hypothesis.

    Parameters
    ----------
    pt: np.ndarray
        Array of shape (n_jets, n_tracks) containing the track transverse momenta (in GeV).
    eta: np.ndarray
        Array of shape (n_jets, n_tracks) containing the track pseudorapidities.
    phi: np.ndarray
        Array of shape (n_jets, n_tracks) containing the track azimuthal angles.
    vtx_idx: np.ndarray
        Array of shape (n_jets, n_tracks) containing the vertex indices for each track.
    particle_mass: float, optional
        Mass hypothesis for each particle in GeV. Default is the pion mass.

    Returns
    -------
    mass: np.ndarray
        Array of shape (n_jets, n_tracks) containing the invariant vertex mass for each track.
    """
    n_jets = pt.shape[0]
    sv_masses = np.full(pt.shape, -1.0, dtype=float)

    for i in range(n_jets):
        vtx_idx_i = vtx_idx[i]
        pt_i = pt[i]
        eta_i = eta[i]
        phi_i = phi[i]

        unique_idx = np.unique(vtx_idx_i)
        unique_idx = unique_idx[unique_idx >= 0]  # remove tracks with negative indices
        vertices = np.tile(vtx_idx_i, (unique_idx.size, 1))
        comparison_ids = np.tile(unique_idx, (vtx_idx_i.size, 1)).T
        vertices = (vertices == comparison_ids).astype(int)

        pt_i = pt_i * vertices
        eta_i = eta_i * vertices
        phi_i = phi_i * vertices
        m_i = particle_mass * vertices

        px_i = pt_i * np.cos(phi_i)
        py_i = pt_i * np.sin(phi_i)
        pz_i = pt_i * np.sinh(eta_i)
        e_i = np.sqrt(px_i**2 + py_i**2 + pz_i**2 + m_i**2)

        px = np.nansum(px_i, axis=1)
        py = np.nansum(py_i, axis=1)
        pz = np.nansum(pz_i, axis=1)
        e = np.nansum(e_i, axis=1)

        m = np.sqrt(e**2 - px**2 - py**2 - pz**2)
        sv_masses[i] = np.sum(np.tile(m, (vtx_idx_i.size, 1)).T * vertices, axis=0)

    return sv_masses
