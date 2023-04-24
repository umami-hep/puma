"""Dummy data generation for plotting."""
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import pandas as pd
from numpy.lib.recfunctions import structured_to_unstructured as s2u
from numpy.lib.recfunctions import unstructured_to_structured as u2s
from scipy.special import softmax


def get_dummy_multiclass_scores(
    size: int = 9_999,
    bjets_mean: float = 1.4,
    seed: int = 42,
):
    """
    Generate dummy data representing output of 3 class classifier.
    Adapted to light-, c- and b-jets and values of `HadronConeExclTruthLabelID`.

    Parameters
    ----------
    size : int, optional
        Size of dummy data. For each of the 3 classes, the same amount is produced,
        if size cannot be divided by 3, next smaller number is taken,
        by default 9_999
    bjets_mean : float, optional
        Mean value of the b-jets 3D gaussian, the more away from 0, the better the
        b-tagging performance, by default 1.4
    seed : int, optional
        Random seed for number generation, by default 42

    Returns
    -------
    np.ndarray
        Output scores of the shape (size, 3)
    np.ndarray
        Labels of shape (size,). The order of the output is light-jets, c-jets, b-jets

    """
    size_class = int(size / 3)
    rng = np.random.default_rng(seed=seed)
    ujets = softmax(rng.normal(loc=[-1, 0, 0], scale=1, size=(size_class, 3)), axis=1)
    cjets = softmax(rng.normal(loc=[0, 1, 0], scale=2.5, size=(size_class, 3)), axis=1)
    bjets = softmax(
        rng.normal(loc=[0, 0, bjets_mean], scale=2, size=(size_class, 3)), axis=1
    )
    output = np.concatenate((ujets, cjets, bjets))
    output = u2s(output, dtype=[("ujets", "f4"), ("cjets", "f4"), ("bjets", "f4")])
    labels = np.concatenate(
        (np.zeros(size_class), np.ones(size_class) * 4, np.ones(size_class) * 5)
    )
    return output, labels


def get_dummy_2_taggers(
    size: int = 9_999,
    shuffle: bool = True,
    seed: int = 42,
    add_pt: bool = False,
    label: str = "HadronConeExclTruthLabelID",
    return_file: bool = False,
):
    """
    Wrapper function of `get_dummy_multiclass_scores` to generate classifier output
    for 2 taggers, in this case rnnip and dips as well as HadronConeExclTruthLabelID.


    Parameters
    ----------
    size : int, optional
        Size of dummy data, by default 9_999
    shuffle : bool, optional
        If True shuffles the dummy data, by default True
    seed : int, optional
        Random seed for number generation (will count +10 for second tagger),
        by default 42
    add_pt : bool, optional
        Specify if pt column should be added as well, by default False
    label : str, optional
        Name of the label column, by default "HadronConeExclTruthLabelID"
    return_file : bool, optional
        If True returns a file object instead of a pandas.DataFrame,

    Returns
    -------
    df_gen : pandas.DataFrame or file
        Dataframe with columns
        [label, rnnip_pu, rnnip_pc, rnnip_pb, dips_pu, dips_pc,
        dips_pb] if `add_pt` is True also pt is added
    """
    output_rnnip, labels = get_dummy_multiclass_scores(
        bjets_mean=0.9, size=size, seed=seed
    )
    df_gen = pd.DataFrame(
        s2u(output_rnnip), columns=["rnnip_pu", "rnnip_pc", "rnnip_pb"]
    )
    df_gen[label] = labels
    output_dips, _ = get_dummy_multiclass_scores(
        bjets_mean=1.4, size=size, seed=seed + 10
    )
    df_gen2 = pd.DataFrame(s2u(output_dips), columns=["dips_pu", "dips_pc", "dips_pb"])
    df_gen = pd.concat([df_gen, df_gen2], axis=1)
    if add_pt:
        rng = np.random.default_rng(seed=seed)
        df_gen["pt"] = rng.exponential(100_000, size=len(df_gen))
    if shuffle:
        df_gen = df_gen.sample(frac=1).reset_index(drop=True)

    df_gen["n_truth_promptLepton"] = 0

    if return_file:
        fname = NamedTemporaryFile(  # pylint: disable=R1732
            mode="w", suffix=".h5", delete=False
        ).name
        file = h5py.File(fname, "w")
        file.create_dataset(name="jets", data=df_gen.to_records())
        return file

    return df_gen
