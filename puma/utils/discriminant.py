"""Discriminant calculation for flavour tagging."""
import numpy as np

from puma.utils.histogram import save_divide


def calc_disc_b(
    arr_pu: np.ndarray, arr_pc: np.ndarray, arr_pb: np.ndarray, fc_par: float
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

    Returns
    -------
    np.ndarray
        b-tagging discriminant
    """
    if len(arr_pu) != len(arr_pc) or len(arr_pu) != len(arr_pb):
        raise ValueError("arr_pu, arr_pc and arr_pb don't have the same length.")
    return np.log(
        save_divide(
            arr_pb,
            fc_par * arr_pc + (1 - fc_par) * arr_pu,
            default=np.infty,
        )
    )


def calc_disc_c(
    arr_pu: np.ndarray, arr_pc: np.ndarray, arr_pb: np.ndarray, fb_par: float
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

    Returns
    -------
    np.ndarray
        b-tagging discriminant
    """
    if len(arr_pu) != len(arr_pc) or len(arr_pu) != len(arr_pb):
        raise ValueError("arr_pu, arr_pc and arr_pb don't have the same length.")
    return np.log(
        save_divide(
            arr_pc,
            fb_par * arr_pb + (1 - fb_par) * arr_pu,
            default=np.infty,
        )
    )
