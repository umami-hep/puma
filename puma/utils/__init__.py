"""Module for usefule tools in puma."""

# flake8: noqa
# pylint: skip-file

import numpy as np
import pandas as pd
from scipy.special import softmax

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


def get_dummy_multiclass_scores(
    size: int = 10_000, bjets_mean: float = 1.4, seed: int = 42
):
    """
    Generate dummy data representing output of 3 class classifier.
    Adapted to light-, c- and b-jets and values of `HadronConeExclTruthLabelID`.

    Parameters
    ----------
    size : int, optional
        size of dummy data. For each of the 3 classes, the same amount of is produces,
        by default 10_000
    bjets_mean : float, optional
        mean value of the b-jets 3D gaussian, the more away from 0, the better the
        b-tagging performance, by default 1.4
    seed : int, optional
        random seed for number generation, by default 42

    Returns
    -------
    np.ndarray
        output scores of the shape (size, 3)
    np.ndarray
        labels of shape (size,). The order of the output is light-jets, c-jets, b-jets

    """
    size_class = int(size / 3)
    rng = np.random.default_rng(seed=seed)
    ujets = softmax(rng.normal(loc=[0, 1, 0], scale=2.5, size=(size_class, 3)), axis=1)
    cjets = softmax(rng.normal(loc=[-1, 0, 0], scale=1, size=(size_class, 3)), axis=1)
    bjets = softmax(
        rng.normal(loc=[0, 0, bjets_mean], scale=2, size=(size_class, 3)), axis=1
    )
    output = np.concatenate((ujets, cjets, bjets))
    labels = np.concatenate(
        (np.zeros(size_class), np.ones(size_class) * 4, np.ones(size_class) * 5)
    )
    return output, labels


def get_dummy_2_taggers(
    size: int = 10_000, shuffle: bool = True, seed: int = 42, add_pt: bool = False
):
    """
    Wrapper function of `get_dummy_multiclass_scores` to generate classifier output
    for 2 taggers, in this case rnnip and dips as well as HadronConeExclTruthLabelID.


    Parameters
    ----------
    size : int, optional
        size of dummy data, by default 10_000
    shuffle : bool, optional
        if True shuffles the dummy data, by default True
    seed : int, optional
        random seed for number generation (will count +10 for second tagger),
        by default 42
    add_pt : bool, optional
        specify if pt column should be added as well, by default False

    Returns
    -------
    pd.DataFrame
        pandas DataFrame with columns
        [HadronConeExclTruthLabelID, rnnip_pu, rnnip_pc, rnnip_pb, dips_pu, dips_pc,
        dips_pb]
    """
    output_rnnip, labels = get_dummy_multiclass_scores(
        bjets_mean=0.9, size=size, seed=seed
    )
    df = pd.DataFrame(output_rnnip, columns=["rnnip_pu", "rnnip_pc", "rnnip_pb"])
    df["HadronConeExclTruthLabelID"] = labels
    output_dips, _ = get_dummy_multiclass_scores(
        bjets_mean=1.4, size=size, seed=seed + 10
    )
    df2 = pd.DataFrame(output_dips, columns=["dips_pu", "dips_pc", "dips_pb"])
    df = pd.concat([df, df2], axis=1)
    if add_pt:
        rng = np.random.default_rng(seed=seed)
        df["pt"] = rng.exponential(100_000, size=len(df))
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    return df


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
        "bjets": {"colour": "#1f77b4", "legend_label": "$b$-jets"},
        "cjets": {"colour": "#ff7f0e", "legend_label": "$c$-jets"},
        "ujets": {"colour": "#2ca02c", "legend_label": "light-flavour jets"},
        "taujets": {"colour": "#7c5295", "legend_label": "$\\tau$-jets"},
        "singlebjets": {"colour": "#1f77b4", "legend_label": "$b$-jets"},
        "bbjets": {"colour": "#012F51", "legend_label": "$bb$-jets"},
        "singlecjets": {"colour": "#ff7f0e", "legend_label": "$c$-jets"},
    },
    "hist_err_style": {
        "fill": False,
        "linewidth": 0,
        "hatch": "/////",
        "edgecolor": "#666666",
    },
}
