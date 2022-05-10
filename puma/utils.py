"""Helper functions for the plotting API"""
import numpy as np
import pandas as pd
from scipy.special import softmax

from umami.configuration import logger  # isort:skip


def translate_kwargs(kwargs):
    """Maintaining backwards compatibility for the kwargs and the new plot_base syntax.

    Parameters
    ----------
    kwargs : dict
        dictionary with kwargs

    Returns
    -------
    dict
        kwargs compatible with new naming.
    """
    mapping = {
        "ApplyATLASStyle": "apply_atlas_style",
        "AtlasTag": "atlas_first_tag",
        "Bin_Width_y_axis": "bin_width_in_ylabel",
        "labelFontSize": "fontsize",
        "legcols": "leg_ncol",
        "legFontSize": "leg_fontsize",
        "loc_legend": "leg_loc",
        "legend_loc": "leg_loc",
        "Log": "logy",
        "n_Leading": "n_leading",
        "ncol": "leg_ncol",
        "nJets": "n_jets",
        "normalise": "norm",
        "ratio_cut": ["ymin_ratio_1", "ymax_ratio_1"],
        "Ratio_Cut": ["ymin_ratio_1", "ymax_ratio_1"],
        "SecondTag": "atlas_second_tag",
        "set_logy": "logy",
        "UseAtlasTag": "use_atlas_tag",
        "yAxisIncrease": "y_scale",
    }
    deprecated_args = ["yAxisAtlasTag"]
    for key, elem in mapping.items():
        if key in kwargs:
            # if old naming is used, translate to new naming
            logger.debug(f"Mapping keyword argument {key} -> {elem}")
            if isinstance(elem, str):
                # print warning if old AND new convention are used
                if elem in kwargs:
                    logger.warning(
                        "You specified two keyword arguments which mean the same: "
                        f"{key}, {elem} --> using the new naming convention {elem}"
                    )
                else:
                    kwargs[elem] = kwargs[key]

            elif isinstance(elem, list):
                for i, key_new in enumerate(elem):
                    kwargs[key_new] = kwargs[key][i]
            kwargs.pop(key)

    # Remove deprecated arguments from kwargs
    for dep_key in deprecated_args:
        if dep_key in kwargs:
            logger.warning(
                f"You specified the argument {dep_key}, which is no longer"
                " supported and will be ignored."
            )
            kwargs.pop(dep_key)
    return kwargs


def translate_binning(
    binning,
    variable_name: str = None,
):
    """Helper function to translate binning used in some configs to an integer that
    represents the number of bins or an array representing the bin edges

    Parameters
    ----------
    binning : int, list or None
        Binning
    variable_name : str, optional
        Name of the variable. If provided, and the name contains "number", the binning
        will be created such that integer numbers are at the center of the bins,
        by default None

    Returns
    -------
    int or np.ndarray
        Number of bins or array of bin edges

    Raises
    ------
    ValueError
        If unsupported type is provided
    """
    if isinstance(binning, list):
        if len(binning) != 3:
            raise ValueError(
                "The list given for binning has to be of length 3, representing "
                "[x_min, x_max, bin_width]"
            )
        bins = np.arange(binning[0], binning[1], binning[2])
        if variable_name is not None:
            if variable_name.startswith("number"):
                bins = np.arange(binning[0] - 0.5, binning[1] - 0.5, binning[2])

    # If int, set to the given numbers
    elif isinstance(binning, int):
        bins = binning

    # If None, set number of bins to 100
    elif binning is None:
        bins = 100
    else:
        raise ValueError(f"Type {type(binning)} is not supported!")

    return bins


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
    ujets = softmax(rng.normal(loc=[-1, 0, 0], scale=1, size=(size_class, 3)), axis=1)
    cjets = softmax(rng.normal(loc=[0, 1, 0], scale=2.5, size=(size_class, 3)), axis=1)
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
