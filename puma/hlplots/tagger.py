"""Tagger module for high level API."""
import h5py
import numpy as np
import pandas as pd
from numpy.lib.recfunctions import structured_to_unstructured

from puma.utils import calc_disc, logger


class Tagger:  # pylint: disable=too-many-instance-attributes
    """Class storing tagger results."""

    def __init__(self, model_name: str, template: dict = None) -> None:
        """Init Tagger class.

        Parameters
        ----------
        model_name : str
            Name of the model, also correspondinng to the pre-fix of the tagger
            variables.
        template : dict
            Template dictionary which keys are directly set as class variables
        """

        self.model_name = model_name
        self.label = None
        self.reference = False

        self.scores = None
        self.perf_var = None
        self.output_nodes = ["pu", "pc", "pb"]

        self.is_b = None
        self.is_light = None
        self.is_c = None

        self.colour = None

        self.disc_cut = None
        self.working_point = None
        self.f_c = None
        self.f_b = None

        self._init_from_template(template)

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
            file path to a h5 file with a pd.DataFrame or `numpy_structured` when
            passing a file path to a h5 file with a structured numpy array,
            by default "data_frame"
        key : str, optional
            Key within h5 file, needs to be provided when using the `source_type`
            `data_frame_path` or `numpy_structured`, by default None
        Raises
        ------
        ValueError
            if source_type is wrongly specified
        """
        # list tagger variables
        tagger_vars = [f"{self.model_name}_{flv}" for flv in self.output_nodes]
        # TODO: change to case syntax in python 3.10
        if source_type == "data_frame":
            logger.debug("Retrieving tagger `%s` from data frame.", self.model_name)
            self.scores = source[tagger_vars].values
            return
        if key is None:
            raise ValueError(
                "When using a `source_type` other than `data_frame`, you need to "
                "specify the `key`."
            )
        if source_type == "data_frame_path":
            logger.debug(
                "Retrieving tagger %s in data frame from file %s.",
                self.model_name,
                source,
            )
            df_in = pd.read_hdf(source, key=key)
            self.scores = df_in[tagger_vars].values

        elif source_type == "numpy_structured":
            logger.debug(
                "Retrieving tagger %s from structured numpy file %s.",
                self.model_name,
                source,
            )
            with h5py.File(source, "r") as f_h5:
                self.scores = structured_to_unstructured(
                    f_h5[key].fields(tagger_vars)[:]
                )
        else:
            raise ValueError(f"{source_type} is not a valid value for `source_type`.")

    def _init_from_template(self, template):
        if template is not None:
            for key, val in template.items():
                if hasattr(self, key):
                    setattr(self, key, val)
                else:
                    raise KeyError(f"`{key}` is not an attribute of the Tagger class.")
        else:
            logger.debug(
                "Template initialised with template being `None` - not doing anything."
            )

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
