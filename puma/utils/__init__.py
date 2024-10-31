"""Module for usefule tools in puma."""

# flake8: noqa

import numpy as np
import pandas as pd
from palettable.colorbrewer.qualitative import Dark2_8

from puma.utils.aux import get_aux_labels
from puma.utils.generate import (
    get_dummy_2_taggers,
    get_dummy_tagger_aux,
    get_dummy_multiclass_scores,
)
from puma.utils.logging import logger, set_log_level


def set_xaxis_ticklabels_invisible(ax):
    """Helper function to set the ticklabels of the xaxis invisible

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis you want to modify
    """

    for label in ax.get_xticklabels():
        label.set_visible(False)


def get_good_pie_colours(colour_scheme=None):
    """Helper function to get good colours for a pie chart. You can
    choose between a specific colour scheme or use the default colours
    for a pie chart

    Parameters
    ----------
    colour_scheme : string, optional
        colour scheme for the pie chart. Can be None to use default colours
        or blue, red, green or yellow to use a specific colour scheme

    Returns
    -------
    list
        returns a list of colours in the specified colour scheme

    Raises
    ------
    KeyError
        If colour_scheme is not in ["blue", "red", "green", "yellow", None]
    """
    if colour_scheme is None:
        return [
            "#3f90da",
            "#ffa90e",
            "#bd1f01",
            "#94a4a2",
            "#832db6",
            "#a96b59",
            "#e76300",
            "#b9ac70",
            "#717581",
            "#92dadd",
        ]
    if colour_scheme == "red":
        return [
            "#750000",
            "#A30000",
            "#D10000",
            "#FF0000",
            "#FF2E2E",
            "#FF5C5C",
            "#FF8A8A",
        ]
    if colour_scheme == "blue":
        return [
            "#000075",
            "#0000A3",
            "#0000D1",
            "#0000FF",
            "#2E2EFF",
            "#5C5CFF",
            "#8A8AFF",
        ]
    if colour_scheme == "green":
        return [
            "#1E5631",
            "#A4DE02",
            "#76BA1B",
            "#4C9A2A",
            "#ACDF87",
            "#68BB59",
            "#CEF3CE",
        ]
    if colour_scheme == "yellow":
        return [
            "#755800",
            "#A37A00",
            "#D19D00",
            "#FFBF00",
            "#FFCB2E",
            "#FFD65C",
            "#FFE28A",
        ]
    raise KeyError(
        f"Given colour scheme is {colour_scheme} but it has to be blue, red, green,"
        " yellow or None"
    )


def get_good_colours(colour_scheme: str | None = None):
    """List of colours adequate for plotting

    Parameters
    ----------
    colour_scheme : string, optional
        colour scheme for line plots, by default None

    Returns
    -------
    list
        list with colours
    """

    # If no colour scheme is selected, return colour-blind friendly colours
    # See https://arxiv.org/pdf/2107.02270 Page 15 (10 colours)
    if colour_scheme is None:
        return [
            "#3f90da",
            "#ffa90e",
            "#bd1f01",
            "#94a4a2",
            "#832db6",
            "#a96b59",
            "#e76300",
            "#b9ac70",
            "#717581",
            "#92dadd",
        ] + Dark2_8.mpl_colors

    elif colour_scheme == "Dark2_8":
        return Dark2_8.mpl_colors


def get_good_markers():
    """List of markers adequate for plotting

    Returns
    -------
    list
        list with markers
    """
    return [
        "o",  # Circle
        "x",  # x
        "v",  # Triangle down
        "^",  # Triangle up
        "D",  # Diamond
        "p",  # Pentagon
        "s",  # Square
    ]


def get_good_linestyles(names=None):
    """Returns a list of good linestyles

    Parameters
    ----------
    names : list or str, optional
        List or string of the name(s) of the linestyle(s) you want to retrieve, e.g.
        "densely dotted" or ["solid", "dashdot", "densely dashed"], by default None

    Returns
    -------
    list
        List of good linestyles. Either the specified selection or the whole list in
        the predefined order.

    Raises
    ------
    ValueError
        If `names` is not a str or list.
    """
    linestyle_tuples = {
        "solid": "solid",
        "densely dashdotted": (0, (3, 1, 1, 1)),
        "densely dashed": (0, (5, 2)),
        "densely dotted": (0, (1, 1)),
        "densely dashdotdotted": (0, (3, 1, 1, 1, 1, 1)),
        "dashed": (0, (5, 5)),
        "dotted": (0, (2, 2)),
        "dashdot": "dashdot",
        "loosely dashed": (0, (5, 10)),
        "loosely dotted": (0, (1, 10)),
        "loosely dashdotted": (0, (3, 10, 1, 10)),
        "loosely dashdotdotted": (0, (3, 10, 1, 10, 1, 10)),
        "dashdotted": (0, (3, 5, 1, 5)),
        "dashdotdotted": (0, (3, 5, 1, 5, 1, 5)),
    }

    default_order = [
        "solid",
        "densely dashdotted",
        "densely dotted",
        "densely dashed",
        "densely dashdotdotted",
        "dashed",
        "dotted",
        "dashdot",
        "loosely dashed",
        "loosely dotted",
        "loosely dashdotted",
        "loosely dashdotdotted",
        "dashdotted",
        "dashdotdotted",
    ]
    if names is None:
        names = default_order
    elif isinstance(names, str):
        return linestyle_tuples[names]
    elif not isinstance(names, list):
        raise ValueError("Invalid type of `names`, has to be a list of strings or a sting.")
    return [linestyle_tuples[name] for name in names]
