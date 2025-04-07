"""Tagger module for high level API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from ftag import Cuts, Flavours, Label
from ftag.utils import get_discriminant

from puma.utils import logger
from puma.utils.aux import get_aux_labels
from puma.utils.vertexing import clean_reco_vertices, clean_truth_vertices


@dataclass
class Tagger:
    """Class storing information and results for a tagger."""

    # commonly passed to the constructor
    name: str
    label: str = None
    reference: bool = False
    colour: str = None
    fxs: dict[str, float] = field(default_factory=lambda: {"fc": 0.1, "fb": 0.2})
    aux_tasks: list = field(default_factory=lambda: list(get_aux_labels().keys()))
    sample_path: Path = None
    category: str = "single-btag"

    # this is only read by the Results class
    cuts: Cuts | list | None = None

    # commonly set by the Results or AuxResults class
    scores: np.ndarray = None
    labels: np.ndarray = None
    aux_scores: dict = None
    aux_labels: dict = None
    perf_vars: dict = None
    aux_perf_vars: dict = None
    output_flavours: list = None
    disc_cut: float = None
    working_point: float = None
    vertexing_require_hf_track: bool = True

    # Used only by YUMA
    yaml_name: str = None

    def __post_init__(self):
        if self.label is None:
            self.label = self.name
        if isinstance(self.cuts, list):
            self.cuts = Cuts.from_list(self.cuts)
        if self.aux_tasks is not None:
            self.aux_scores = dict.fromkeys(self.aux_tasks)
            self.aux_labels = dict.fromkeys(self.aux_tasks)
        if self.sample_path is not None:
            self.sample_path = Path(self.sample_path)
        if self.output_flavours is None:
            logger.info(
                f"No output flavours were given. Using flavours of category {self.category}"
            )
            self.output_flavours = Flavours.by_category(self.category)

        # Ensure output_flavours is a list of Label instances
        self.output_flavours = [Flavours[iter_flav] for iter_flav in self.output_flavours]

        # Check used flavours and check that they are in the chosen label category
        for iter_flav in self.output_flavours:
            if iter_flav not in Flavours.by_category(category=self.category):
                raise ValueError(
                    f"Given output flavour {iter_flav.name} is not supported in label category "
                    f"{self.category}"
                )

        # Check if some flavours from the category are not used. Set their fraction values to 0
        for iter_ref_flav in Flavours.by_category(category=self.category):
            if iter_ref_flav not in self.output_flavours:
                logger.debug(
                    f"Flavour {iter_ref_flav} in category {self.category} but not in "
                    f"output_flavours. Setting {iter_ref_flav.frac_str} to 0."
                )
                self.fxs[iter_ref_flav.frac_str] = 0

    def __repr__(self):
        return f"{self.name} ({self.label})"

    def is_flav(self, flavour: Label | str):
        """Return indices of jets of given flavour.

        Parameters
        ----------
        flavour : Label | str
            Flavour label to select

        Returns
        -------
        np.ndarray
            Array of indices of the given flavour
        """
        flavour = Flavours[flavour]
        return flavour.cuts(self.labels).idx

    @property
    def probabilities(self):
        """Return the probabilities of the tagger.

        Returns
        -------
        list
            List of probability names
        """
        return [flav.px for flav in self.output_flavours]

    @property
    def variables(self):
        """Return a list of the outputs of the tagger.

        Returns
        -------
        list
            List of the outputs variable names of the tagger
        """
        return [f"{self.name}_{prob}" for prob in self.probabilities]

    @property
    def aux_variables(self):
        """Return a dict of the auxiliary outputs of the tagger for each task.

        Returns
        -------
        aux_outputs: dict
            Dictionary of auxiliary output variables of the tagger

        Raises
        ------
        ValueError
            If element in self.aux_tasks is unrecognized
        """
        aux_outputs = {}

        for aux_type in self.aux_tasks:
            if aux_type == "vertexing":
                if self.name in {"SV1", "JF"}:
                    aux_outputs[aux_type] = f"{self.name}VertexIndex"
                else:
                    aux_outputs[aux_type] = f"{self.name}_aux_VertexIndex"
            elif aux_type == "track_origin":
                aux_outputs[aux_type] = f"{self.name}_aux_TrackOrigin"
            else:
                raise ValueError(f"{aux_type} is not a recognized aux task.")

        return aux_outputs

    def extract_tagger_scores(
        self,
        source: object,
        source_type: str = "data_frame",
        key: str | None = None,
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
            self.scores = source[self.variables].to_records(index=False)
            return
        if source_type == "structured_array":
            self.scores = source[self.variables]
            return
        if key is None:
            raise ValueError(
                "When using a `source_type` other than `data_frame`, you need to"
                " specify the `key`."
            )
        if source_type == "data_frame_path":
            df_in = pd.read_hdf(source, key=key)
            self.scores = df_in[self.variables].to_records(index=False)

        elif source_type == "h5_file":
            with h5py.File(source, "r") as f_h5:
                self.scores = f_h5[key].fields(self.variables)[:]

        else:
            raise ValueError(f"{source_type} is not a valid value for `source_type`.")

    def n_jets(self, flavour: Label | str):
        """Retrieve number of jets of a given flavour.

        Parameters
        ----------
        flavour : Label | str
            Flavour of jets to count

        Returns
        -------
        int
            Number of jets of given flavour
        """
        flavour = Flavours[flavour]
        return len(flavour.cuts(self.labels).values)

    def probs(self, prob_flavour: Label, label_flavour: Label = None):
        """Retrieve probabilities for a given flavour.

        Parameters
        ----------
        prob_flavour : Label
            Return probabilities for this flavour class
        label_flavour : Label, optional
            Only return jets of the given truth flavour, by default None

        Returns
        -------
        np.ndarray
            Probabilities for given flavour
        """
        prob_flavour = Flavours[prob_flavour]
        scores = self.scores
        if label_flavour is not None:
            scores = scores[self.is_flav(label_flavour)]
        return scores[f"{self.name}_{prob_flavour.px}"]

    def discriminant(self, signal: Label, fxs: dict | None = None):
        """Retrieve the discriminant for a given signal class.

        Parameters
        ----------
        signal : Label
            Signal class for which the discriminant should be retrieved
        fxs : dict, optional
            dict of fractions to use instead of the default ones, by default None

        Returns
        -------
        np.ndarray
            Discriminant for given signal class
        """
        # Ensure Label instance
        if isinstance(signal, str):
            signal = Flavours[signal]

        # Check that the given flavour is in output flavours
        if signal not in self.output_flavours:
            raise ValueError(
                f"Given signal flavour {signal.name} is not available in given output flavours!"
            )

        # Get fraction values from class if not provided
        if fxs is None:
            fxs = self.fxs

        # Remove signal fraction value if present
        fxs = {k: v for k, v in fxs.items() if k != signal.frac_str}

        # Init a counter to count, how many flavour have no fraction value given
        mis_frac_list = []

        # Check that all fraction values for the flavours are given
        for iter_flav in self.output_flavours:
            # Skip signal flavour
            if iter_flav.name == signal.name:
                continue

            # Check which flavours are missing a fraction value
            if iter_flav.frac_str not in fxs:
                mis_frac_list.append(iter_flav.frac_str)

        # Ensure only one fraction value is missing and calculate it
        if len(mis_frac_list) == 1:
            tmp_fx_value = 0
            for frac_value in fxs.values():
                tmp_fx_value += frac_value
            fxs[mis_frac_list[0]] = round(1 - tmp_fx_value, 3)

        elif len(mis_frac_list) >= 2:
            raise ValueError(
                "More than one fraction value is missing from the fraction dict! Please check"
                f"{mis_frac_list}"
            )

        # Calculate discs
        return get_discriminant(
            jets=self.scores,
            tagger=self.name,
            signal=signal,
            flavours=self.output_flavours,
            fraction_values=fxs,
        )

    def vertex_indices(self, incl_vertexing=False):
        """Retrieve cleaned vertex indices for the tagger.

        Parameters
        ----------
        incl_vertexing : bool, optional
            Whether to merge all vertex indices, by default False.

        Returns
        -------
        truth_indices : np.ndarray
            Cleaned truth vertex indices for the tagger.
        reco_indices : np.ndarray
            Cleaned reco vertex indices for the tagger.
        """
        if "vertexing" not in self.aux_tasks:
            raise ValueError("Vertexing aux task not available for this tagger.")
        if "vertexing" not in self.aux_labels:
            raise ValueError("Vertexing labels not found.")
        if "track_origin" not in self.aux_labels:
            raise ValueError("Track origin labels not found.")

        truth_indices = np.copy(self.aux_labels["vertexing"])
        reco_indices = np.copy(self.aux_scores["vertexing"])

        # clean truth and reco indices for each jet
        for i in range(truth_indices.shape[0]):
            truth_indices[i] = clean_truth_vertices(
                truth_indices[i], self.aux_labels["track_origin"][i], incl_vertexing=incl_vertexing
            )
            reco_track_origin = (
                None
                if "track_origin" not in self.aux_scores
                else self.aux_scores["track_origin"][i]
            )
            reco_indices[i] = clean_reco_vertices(
                reco_indices[i],
                reco_track_origin,
                incl_vertexing=incl_vertexing,
                require_hf_track=self.vertexing_require_hf_track,
            )

        return truth_indices, reco_indices
