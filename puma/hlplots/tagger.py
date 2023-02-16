"""Tagger module for high level API."""
from dataclasses import dataclass, field

import h5py
import numpy as np
import pandas as pd
from numpy.lib.recfunctions import structured_to_unstructured

from puma.utils import calc_disc, logger


@dataclass
class Tagger:  # pylint: disable=too-many-instance-attributes
    """Class storing information and results for a tagger."""

    name: str
    label: str = None
    reference: bool = False

    scores = None
    perf_var = None
    output_nodes: list = field(default_factory=lambda: ["pu", "pc", "pb"])

    is_b: bool = None
    is_light: bool = None
    is_c: bool = None

    colour: str = None

    disc_cut: float = None
    working_point: float = None
    f_c: float = None
    f_b: float = None

    def __post_init__(self):
        if self.label is None:
            self.label = self.name

    @property
    def variables(self):
        """Return a list of the outputs of the tagger.

        Returns
        -------
        list
            List of the outputs variable names of the tagger
        """

        return [f"{self.name}_{flv}" for flv in self.output_nodes]

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
        # TODO: change to case syntax in python 3.10
        if source_type == "data_frame":
            logger.debug("Retrieving tagger `%s` from data frame.", self.name)
            self.scores = source[self.variables].values
            return
        if source_type == "structured_array":
            logger.debug(
                "Retrieving tagger %s from h5py fields %s.",
                self.name,
                source,
            )
            self.scores = structured_to_unstructured(source[self.variables])
            return
        if key is None:
            raise ValueError(
                "When using a `source_type` other than `data_frame`, you need to "
                "specify the `key`."
            )
        if source_type == "data_frame_path":
            logger.debug(
                "Retrieving tagger %s in data frame from file %s.",
                self.name,
                source,
            )
            df_in = pd.read_hdf(source, key=key)
            self.scores = df_in[self.variables].values

        elif source_type == "h5_file":
            logger.debug(
                "Retrieving tagger %s from structured h5 file %s.",
                self.name,
                source,
            )
            with h5py.File(source, "r") as f_h5:
                self.scores = structured_to_unstructured(
                    f_h5[key].fields(self.variables)[:]
                )

        else:
            raise ValueError(f"{source_type} is not a valid value for `source_type`.")

    @property
    def n_jets_light(self):
        """Retrieve number of light jets.

        Returns
        -------
        int
            number of light jets
        """
        return int(np.sum(self.is_light))

    @property
    def n_jets_c(self):
        """Retrieve number of c jets.

        Returns
        -------
        int
            number of c jets
        """
        return int(np.sum(self.is_c))

    @property
    def n_jets_b(self):
        """Retrieve number of b jets.

        Returns
        -------
        int
            number of b jets
        """
        return int(np.sum(self.is_b))

    def calc_disc_b(self) -> np.ndarray:
        """Calculate b-tagging discriminant

        Returns
        -------
        np.ndarray
            b-tagging discriminant

        Raises
        ------
        ValueError
            if f_c parameter is not specified for tagger
        """
        if self.f_c is None:
            raise ValueError(
                "Before calculating the b-tagging discriminant, specify `f_c`"
            )
        flv_map = {
            "sig": {"pb": 1.0},
            "bkg": {"pu": 1 - self.f_c, "pc": self.f_c},
        }
        return calc_disc(
            scores=self.scores,
            flvs=self.output_nodes,
            flv_map=flv_map,
        )

    def calc_disc_c(self) -> np.ndarray:
        """Calculate c-tagging discriminant

        Returns
        -------
        np.ndarray
            c-tagging discriminant

        Raises
        ------
        ValueError
            if f_b parameter is not specified for tagger
        """
        if self.f_b is None:
            raise ValueError(
                "Before calculating the c-tagging discriminant, specify `f_b`"
            )
        flv_map = {
            "sig": {"pc": 1.0},
            "bkg": {"pu": 1 - self.f_b, "pb": self.f_b},
        }
        return calc_disc(
            scores=self.scores,
            flvs=self.output_nodes,
            flv_map=flv_map,
        )
