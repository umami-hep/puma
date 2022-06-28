"""Module for usefule tools in puma."""

# flake8: noqa
# pylint: skip-file

import numpy as np
import pandas as pd
from scipy.special import softmax

from puma.utils.generate import get_dummy_2_taggers, get_dummy_multiclass_scores
from puma.utils.logging import logger, set_log_level  # noqa: F401


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
    # TODO change in python 3.10 -> case syntax
    if colour_scheme is None:
        return [
            "#1F77B4",
            "#FF7F0E",
            "#2CA02C",
            "#D62728",
            "#9467BD",
            "#8C564B",
            "#E377C2",
            "#7F7F7F",
            "#808000",
            "#000080",
            "#800080",
            "#80B9A1",
            "#205522",
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
        f"Given colour scheme is {colour_scheme} but it has to "
        "be blue, red, green, yellow or None"
    )


def get_good_colours():
    """List of colours adequate for plotting

    Returns
    -------
    list
        list with colours
    """
    return ["#AA3377", "#228833", "#4477AA", "#CCBB44", "#EE6677", "#BBBBBB"]


global_config = {
    "flavour_categories": {
        "bjets": {
            "colour": "#1f77b4",  # blue
            "legend_label": "$b$-jets",
        },
        "cjets": {
            "colour": "#ff7f0e",  # orange
            "legend_label": "$c$-jets",
        },
        "ujets": {
            "colour": "#2ca02c",  # green
            "legend_label": "light-flavour jets",
        },
        "taujets": {
            "colour": "#7c5295",  # purple
            "legend_label": "$\\tau$-jets",
        },
        "singlebjets": {
            "colour": "#1f77b4",  # blue (like b-jets)
            "legend_label": "single-$b$ jets",
        },
        "bbjets": {
            "colour": "#8E0024",  # dark red
            "legend_label": "$bb$-jets",
        },
        "singlecjets": {
            "colour": "#ff7f0e",  # orange (like c-jets)
            "legend_label": "single-$c$ jets",
        },
        "ccjets": {
            "colour": "#ad4305",
            "legend_label": "$cc$-jets",
        },
        "upjets": {
            "colour": "#9ed670",
            "legend_label": "$u$-jets",
        },
        "djets": {
            "colour": "#274e13",
            "legend_label": "$d$-jets",
        },
        "sjets": {
            "colour": "#00bfaf",
            "legend_label": "$s$-jets",
        },
        "gluonjets": {
            "colour": "#7b4e24",
            "legend_label": "gluon-jets",
        },
        "lquarkjets": {
            "colour": "#A05252",
            "legend_label": "light-fl. jets w/o gluons",
        },
        "hadrcbjets": {
            "colour": "#264653",
            "legend_label": "hadronic $b$-hadron decay",
        },
        "lepcbjets": {
            "colour": "#190099",
            "legend_label": "leptonic $b$-hadron decay",
        },
        "singleebdecay": {
            "colour": "#e9c46a",
            "legend_label": "$e$'s in $b$- or $c$-hadron decay",
        },
        "singlemubdecay": {
            "colour": "#f4a261",
            "legend_label": "$\\mu$'s in $b$- or $c$-hadron decay",
        },
        "singletaubdecay": {
            "colour": "#e76f51",
            "legend_label": "\u03C4's in $b$- or $c$-hadron decay",
        },
    },
    "hist_err_style": {
        "fill": False,
        "linewidth": 0,
        "hatch": "/////",
        "edgecolor": "#666666",
    },
}
