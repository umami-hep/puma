"""Tagger module for high level API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from ftag import Cuts, Flavours, Label
from ftag.utils import get_discriminant

from puma.utils import logger
from puma.utils.auxiliary import get_aux_labels
from puma.utils.vertexing import clean_reco_vertices, clean_truth_vertices


@dataclass
class Tagger:
    """Class storing information and results for a tagger."""

    # commonly passed to the constructor
    name: str
    label: str | None = None
    reference: bool = False
    colour: str | None = None
    fxs: dict[str, float] = field(default_factory=lambda: {"fc": 0.1, "fb": 0.2})
    aux_tasks: list[str] = field(default_factory=lambda: list(get_aux_labels().keys()))
    sample_path: Path | str | None = None
    category: str = "single-btag"

    # this is only read by the Results class
    cuts: Cuts | list[str] | None = None

    # commonly set by the Results or AuxResults class
    scores: np.ndarray | None = None
    labels: np.ndarray | None = None
    aux_scores: dict[str, Any] | None = None
    aux_labels: dict[str, Any] | None = None
    perf_vars: dict[str, Any] | None = None
    aux_perf_vars: dict[str, Any] | None = None
    output_flavours: list[Label] | list[str] | None = None
    disc_cut: float | None = None
    working_point: float | None = None
    vertexing_require_hf_track: bool = True

    # Filepaths of stored Histogram and ROC objects
    prob_path: dict[str, Path] | None = None
    roc_path: dict[str, Path] | None = None
    disc_path: dict[str, Path] | None = None

    # Used only by YUMA
    yaml_name: str | None = None

    def __post_init__(self) -> None:
        """Run post init checks of the inputs.

        Raises
        ------
        ValueError
            If a given output flavour is not supported by the label category
        """
        if self.label is None:
            self.label = self.name

        if isinstance(self.cuts, list):
            self.cuts = Cuts.from_list(self.cuts)

        if self.aux_tasks is not None:
            # create dicts keyed by task names
            self.aux_scores = dict.fromkeys(self.aux_tasks)  # type: ignore[assignment]
            self.aux_labels = dict.fromkeys(self.aux_tasks)  # type: ignore[assignment]

        if self.sample_path is not None:
            self.sample_path = Path(self.sample_path)

        if not self.output_flavours:
            logger.info(
                f"No output flavours were given. Using flavours of category {self.category}"
            )
            self.output_flavours = Flavours.by_category(self.category)

        # Ensure output_flavours is a list of Label instances
        # (allowing that user may have passed strings)
        self.output_flavours = [
            Flavours[f] if isinstance(f, str) else f  # type: ignore[index]
            for f in self.output_flavours
        ]

        # Check used flavours are in the chosen label category
        cat_flavs = Flavours.by_category(category=self.category)
        for iter_flav in self.output_flavours:
            if iter_flav not in cat_flavs:
                raise ValueError(
                    f"Given output flavour {iter_flav.name} is not supported in label category "
                    f"{self.category}"
                )

        # For flavours in the category but not used: set their fractions to 0
        for iter_ref_flav in cat_flavs:
            if iter_ref_flav not in self.output_flavours:
                logger.debug(
                    f"Flavour {iter_ref_flav} in category {self.category} but not in "
                    f"output_flavours. Setting {iter_ref_flav.frac_str} to 0."
                )
                self.fxs[iter_ref_flav.frac_str] = 0

    def __repr__(self) -> str:
        """Return the name and label of the tagger.

        Returns
        -------
        str
            Name and label of the tagger.
        """
        return f"{self.name} ({self.label})"

    def is_flav(self, flavour: Label | str) -> np.ndarray:
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
        assert self.labels is not None, "labels must be set before calling is_flav()"
        return flavour.cuts(self.labels).idx

    @property
    def probabilities(self) -> list[str]:
        """Return the probabilities of the tagger.

        Returns
        -------
        list
            List of probability names
        """
        assert self.output_flavours is not None, "output_flavours not initialized"
        return [flav.px for flav in self.output_flavours]

    @property
    def variables(self) -> list[str]:
        """Return a list of the outputs of the tagger.

        Returns
        -------
        list
            List of the outputs variable names of the tagger
        """
        return [f"{self.name}_{prob}" for prob in self.probabilities]

    @property
    def aux_variables(self) -> dict[str, str]:
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
        aux_outputs: dict[str, str] = {}
        for aux_type in self.aux_tasks:
            if aux_type == "vertexing":
                aux_outputs[aux_type] = (
                    f"{self.name}VertexIndex"
                    if self.name in {"SV1", "JF"}
                    else f"{self.name}_aux_VertexIndex"
                )
            elif aux_type == "track_origin":
                aux_outputs[aux_type] = f"{self.name}_aux_TrackOrigin"
            else:
                raise ValueError(f"{aux_type} is not a recognized aux task.")
        return aux_outputs

    def extract_tagger_scores(
        self,
        source: pd.DataFrame | np.ndarray | str | Path,
        source_type: str = "data_frame",
        key: str | None = None,
    ) -> None:
        """Extract tagger scores from data frame or file.

        Parameters
        ----------
        source : pd.DataFrame | np.ndarray | str | Path
            pd.DataFrame or file path to h5 file containing pd.DataFrame or structured
            numpy array
        source_type : str, optional
            Indicates from which source scores should be extracted. Possible options are
            `data_frame` when passing a pd.DataFrame, `data_frame_path` when passing a
            file path to a h5 file with a pd.DataFrame, `h5_file` when
            passing a file path to a h5 file with a structured numpy array, or
            `strucuted_array` when passing a structured numpy array,
            by default "data_frame"
        key : str | None, optional
            Key within h5 file, needs to be provided when using the `source_type`
            `data_frame_path` or `numpy_structured`, by default None

        Raises
        ------
        ValueError
            if source_type is wrongly specified
        """
        if source_type == "data_frame":
            assert isinstance(source, pd.DataFrame)
            self.scores = source[self.variables].to_records(index=False)
            return
        if source_type == "structured_array":
            assert isinstance(source, np.ndarray)
            self.scores = source[self.variables]
            return
        if key is None:
            raise ValueError(
                "When using a `source_type` other than `data_frame`, you need to specify `key`."
            )
        if source_type == "data_frame_path":
            df_in = pd.read_hdf(source, key=key)
            self.scores = df_in[self.variables].to_records(index=False)
        elif source_type == "h5_file":
            with h5py.File(source, "r") as f_h5:
                self.scores = f_h5[key].fields(self.variables)[:]
        else:
            raise ValueError(f"{source_type} is not a valid value for `source_type`.")

    def n_jets(self, flavour: Label | str) -> int:
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
        assert self.labels is not None, "labels must be set before calling n_jets()"
        return len(flavour.cuts(self.labels).values)

    def probs(
        self, prob_flavour: Label | str, label_flavour: Label | str | None = None
    ) -> np.ndarray:
        """Retrieve probabilities for a given flavour.

        Parameters
        ----------
        prob_flavour : Label | str
            Return probabilities for this flavour class
        label_flavour : Label | str | None, optional
            Only return jets of the given truth flavour, by default None

        Returns
        -------
        np.ndarray
            Probabilities for given flavour
        """
        prob_flavour = Flavours[prob_flavour]
        assert self.scores is not None, "scores must be set before calling probs()"
        scores = self.scores
        if label_flavour is not None:
            scores = scores[self.is_flav(label_flavour)]
        return scores[f"{self.name}_{prob_flavour.px}"]

    def discriminant(self, signal: Label | str, fxs: dict[str, float] | None = None) -> np.ndarray:
        """Retrieve the discriminant for a given signal class.

        Parameters
        ----------
        signal : Label | str
            Signal class for which the discriminant should be retrieved
        fxs : dict[str, float] | None, optional
            dict of fractions to use instead of the default ones, by default None

        Returns
        -------
        np.ndarray
            Discriminant for given signal class

        Raises
        ------
        ValueError
            If the given signal flavour is not available in the given output flavours
        """
        signal = Flavours[signal]
        assert self.scores is not None, "scores must be set before calling discriminant()"
        assert self.output_flavours is not None, "output_flavours not initialized"

        if signal not in self.output_flavours:
            raise ValueError(
                f"Given signal flavour {signal.name} is not available in given output flavours!"
            )

        use_fxs = dict(self.fxs if fxs is None else fxs)
        # Remove signal fraction value if present
        use_fxs.pop(signal.frac_str, None)

        # Find missing flavour fractions
        missing: list[str] = []

        # Check that all fraction values for the flavours are given
        for iter_flav in self.output_flavours:
            # Skip signal flavour
            if iter_flav.name == signal.name:
                continue

            # Check which flavours are missing a fraction value
            if iter_flav.frac_str not in use_fxs:
                missing.append(iter_flav.frac_str)

        # Ensure only one fraction value is missing and calculate it
        if len(missing) == 1:
            total = sum(use_fxs.values())
            use_fxs[missing[0]] = round(1 - total, 3)
        elif len(missing) >= 2:
            raise ValueError(
                "More than one fraction value is missing from the fraction dict! Please check "
                f"{missing}"
            )

        # Calculate discs
        return get_discriminant(
            jets=self.scores,
            tagger=self.name,
            signal=signal,
            flavours=self.output_flavours,
            fraction_values=use_fxs,
        )

    def vertex_indices(self, incl_vertexing: bool = False) -> tuple[np.ndarray, np.ndarray]:
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

        Raises
        ------
        ValueError
            If the vertexing aux task is not available for a given tagger
            If the vertexing labels weren't found
            If the track origin labels weren't found
        """
        if "vertexing" not in self.aux_tasks:
            raise ValueError("Vertexing aux task not available for this tagger.")
        assert self.aux_labels is not None, "aux_labels not set"
        assert self.aux_scores is not None, "aux_scores not set"
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
