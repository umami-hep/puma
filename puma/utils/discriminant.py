"""Discriminant calculation for flavour tagging."""
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as s2u

from puma.utils.histogram import save_divide


def calc_disc(
    scores: np.ndarray,
    flvs: list = None,
    flv_map: dict = None,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Calculate arbitrary flavour tagging score.

    Parameters
    ----------
    scores : np.ndarray
        tagger scores in the shape (n_jets, n_flavours)
    flvs : list, optional
        List of flavours corresponding to the order in the scores array, by default None
    flv_map : dict, optional
        flavour map containing signal and background mapping from `flv` and their
        fractions, by default None
    epsilon : float, optional
        adds a small epsilon to the numerator and denominator to avoid infinities,
        by default 1e-10

    Returns
    -------
    np.ndarray
        discriminant values

    Raises
    ------
    ValueError
        if scores and shapes have different shapes

    Examples
    --------
    The `flv_map` can e.g. look like this together with the `flvs`
    >>> flv_map ={
    ...     "sig": {"b": 1.0},
    ...     "bkg": {"l": 1 - 0.5, "c": 0.5},
    ... }
    >>> flvs = ["l", "c", "b"]
    >>> scores = np.column_stack((np.ones(10), np.ones(10), np.ones(10)))
    >>> calc_disc(scores, flvs=flvs, flv_map=flv_map)
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    """
    flvs = ["l", "c", "b"] if flvs is None else flvs

    if scores.dtype.names is not None:
        scores = s2u(scores)

    if len(flvs) != scores.shape[1]:
        raise ValueError("`flvs` and `scores` have incompatible shapes.")
    flv_map = (
        {
            "sig": {"b": 1.0},
            "bkg": {"l": 1 - 0.5, "c": 0.5},
        }
        if flv_map is None
        else flv_map
    )
    numerator = None
    for sig_name, frac in flv_map["sig"].items():
        sig_arr = scores[:, flvs.index(sig_name)]
        numerator = frac * sig_arr if numerator is None else numerator + frac * sig_arr
    denominator = None
    for bkg_name, frac in flv_map["bkg"].items():
        bkg_arr = scores[:, flvs.index(bkg_name)]
        denominator = (
            frac * bkg_arr if denominator is None else denominator + frac * bkg_arr
        )
    discs = np.log(
        save_divide(
            numerator + epsilon,
            denominator + epsilon,
            default=np.infty,
        )
    )
    return discs


def calc_disc_b(
    arr_pu: np.ndarray,
    arr_pc: np.ndarray,
    arr_pb: np.ndarray,
    fc_par: float,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Calculate b-tagging discriminant with one fc parameter.

    Parameters
    ----------
    arr_pu : np.ndarray
        Light prediction scores
    arr_pc : np.ndarray
        c prediction scores
    arr_pb : np.ndarray
        b prediction scores
    fc_par : float
        fc parameter for b-jet discriminant
    epsilon : float, optional
        adds a small epsilon to the numerator and denominator to avoid infinities,
        by default 1e-10

    Returns
    -------
    np.ndarray
        b-tagging discriminant

    Raises
    ------
    ValueError
        if inputs has not the same length
    """
    if len(arr_pu) != len(arr_pc) or len(arr_pu) != len(arr_pb):
        raise ValueError("arr_pu, arr_pc and arr_pb don't have the same length.")
    return np.log(
        save_divide(
            arr_pb + epsilon,
            fc_par * arr_pc + (1 - fc_par) * arr_pu + epsilon,
            default=np.infty,
        )
    )


def calc_disc_c(
    arr_pu: np.ndarray,
    arr_pc: np.ndarray,
    arr_pb: np.ndarray,
    fb_par: float,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Calculate c-tagging discriminant with one fb parameter.

    Parameters
    ----------
    arr_pu : np.ndarray
        Light prediction scores
    arr_pc : np.ndarray
        c prediction scores
    arr_pb : np.ndarray
        b prediction scores
    fb_par : float
        fb parameter for c-jet discriminant
    epsilon : float, optional
        adds a small epsilon to the numerator and denominator to avoid infinities,
        by default 1e-10

    Returns
    -------
    np.ndarray
        b-tagging discriminant
    Raises
    ------
    ValueError
        if inputs has not the same length
    """
    if len(arr_pu) != len(arr_pc) or len(arr_pu) != len(arr_pb):
        raise ValueError("arr_pu, arr_pc and arr_pb don't have the same length.")
    return np.log(
        save_divide(
            arr_pc + epsilon,
            fb_par * arr_pb + (1 - fb_par) * arr_pu + epsilon,
            default=np.infty,
        )
    )
