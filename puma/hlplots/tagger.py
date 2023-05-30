"""Tagger module for high level API."""
from __future__ import annotations

from dataclasses import dataclass, field

import h5py
import numpy as np
import pandas as pd
from ftag import Flavour, Flavours, get_discriminant

from puma.utils import logger


@dataclass
class Tagger:
    """Class storing information and results for a tagger."""

    # commonly passed to the constructor
    name: str
    label: str = None
    reference: bool = False
    colour: str = None

    # commonly set by the Results class
    scores: np.ndarray = None
    labels: np.ndarray = None
    perf_var: np.ndarray = None
    output_nodes: list = field(
        default_factory=lambda: [Flavours.ujets, Flavours.cjets, Flavours.bjets]
    )

    disc_cut: float = None
    working_point: float = None
    f_c: float = None
    f_b: float = None

    def __post_init__(self):
        if self.label is None:
            self.label = self.name

    def __repr__(self):
        return f"{self.name} ({self.label})"

    def is_flav(self, flavour: Flavour | str):
        """Return indices of jets of given flavour.

        Parameters
        ----------
        flavour : str
            Flavour to select

        Returns
        -------
        np.ndarray
            Array of indices of the given flavour
        """
        flavour = Flavours[flavour] if isinstance(flavour, str) else flavour
        return flavour.cuts(self.labels).idx

    @property
    def probabilities(self):
        """Return the probabilities of the tagger.

        Returns
        -------
        list
            List of probability names
        """
        return [flav.px for flav in self.output_nodes]

    @property
    def variables(self):
        """Return a list of the outputs of the tagger.

        Returns
        -------
        list
            List of the outputs variable names of the tagger
        """
        return [f"{self.name}_{prob}" for prob in self.probabilities]

    def extract_tagger_scores(
        self, source: object, source_type: str = "data_frame", key: str = None
    ):
        """Extract tagger scores from data frame or file.

        Parameters
        ----------
        source : object
            pd.DataFrame or file path to h5 file containing pd.DataFrame or structured
            numpy array
        source_type : str, optional
            Indicates from which source scores should be extracted. Possible options are
            `data_frame` when passing a pd.DataFrame, `data_frame_path` when passing a
            file path to a h5 file with a pd.DataFrame, `h5_file` when
            passing a file path to a h5 file with a structured numpy array, or
            `strucuted_array` when passing a structured numpy array,
            by default "data_frame"
        key : str, optional
            Key within h5 file, needs to be provided when using the `source_type`
            `data_frame_path` or `numpy_structured`, by default None
        Raises
        ------
        ValueError
            if source_type is wrongly specified
        """
        if source_type == "data_frame":
            logger.debug("Retrieving tagger `%s` from data frame.", self.name)
            self.scores = source[self.variables]
            return
        if source_type == "structured_array":
            logger.debug(
                "Retrieving tagger %s from h5py fields %s.",
                self.name,
                source,
            )
            self.scores = source[self.variables]
            return
        if key is None:
            raise ValueError(
                "When using a `source_type` other than `data_frame`, you need to"
                " specify the `key`."
            )
        if source_type == "data_frame_path":
            logger.debug(
                "Retrieving tagger %s in data frame from file %s.",
                self.name,
                source,
            )
            df_in = pd.read_hdf(source, key=key)
            self.scores = df_in[self.variables]

        elif source_type == "h5_file":
            logger.debug(
                "Retrieving tagger %s from structured h5 file %s.",
                self.name,
                source,
            )
            with h5py.File(source, "r") as f_h5:
                self.scores = f_h5[key].fields(self.variables)[:]

        else:
            raise ValueError(f"{source_type} is not a valid value for `source_type`.")

    def n_jets(self, flavour: Flavour | str):
        """Retrieve number of jets of a given flavour.

        Parameters
        ----------
        flavour : Flavour | str
            Flavour of jets to count

        Returns
        -------
        int
            Number of jets of given flavour
        """
        flavour = Flavours[flavour] if isinstance(flavour, str) else flavour
        return len(flavour.cuts(self.labels).values)

    def probs(self, prob_flavour: Flavour, label_flavour: Flavour = None):
        """Retrieve probabilities for a given flavour.

        Parameters
        ----------
        prob_flavour : Flavour
            Return probabilities for this flavour class
        label_flavour : Flavour, optional
            Only return jets of the given truth flavour, by default None

        Returns
        -------
        np.ndarray
            Probabilities for given flavour
        """
        return self.scores[self.is_flav(label_flavour)][
            f"{self.name}_{prob_flavour.px}"
        ]

    def discriminant(self, signal: Flavour, fx: float = None):
        """Retrieve the discriminant for a given signal class.

        Parameters
        ----------
        signal : Flavour
            Signal class for which the discriminant should be retrieved
        fx : float, optional
            fc or fb value, by default None

        Returns
        -------
        np.ndarray
            Discriminant for given signal class

        Raises
        ------
        ValueError
            If no discriminant is defined for given signal class
        """
        if fx is not None and signal not in (Flavours.bjets, Flavours.cjets):
            raise ValueError("fx only valid for bjets and cjets.")
        if fx is None:
            fx = self.f_c if Flavours[signal] == Flavours.bjets else self.f_b
        if Flavours[signal] == Flavours.bjets:
            return get_discriminant(self.scores, self.name, signal, fx)
        if Flavours[signal] == Flavours.cjets:
            return get_discriminant(self.scores, self.name, signal, fx)
        if Flavours[signal] in (Flavours.hbb, Flavours.hcc):
            return self.scores[signal.px]
        raise ValueError(f"No discriminant defined for {signal} signal.")
