"""Tagger module for high level API."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from ftag import Cuts, Flavour, Flavours, get_discriminant

from puma.utils.aux import get_aux_labels
from puma.utils.vertexing import clean_indices


@dataclass
class Tagger:
    """Class storing information and results for a tagger."""

    # commonly passed to the constructor
    name: str
    label: str = None
    reference: bool = False
    colour: str = None
    f_c: float = None
    f_b: float = None
    aux_tasks: list = field(default_factory=lambda: list(get_aux_labels().keys()))
    sample_path: Path = None

    # this is only read by the Results class
    cuts: Cuts | list | None = None

    # commonly set by the Results or AuxResults class
    scores: np.ndarray = None
    labels: np.ndarray = None
    aux_scores: dict = None
    aux_labels: dict = None
    perf_vars: dict = None
    output_flavours: list = field(
        default_factory=lambda: [Flavours.ujets, Flavours.cjets, Flavours.bjets]
    )
    disc_cut: float = None
    working_point: float = None

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
                if self.name == "SV1" or self.name == "JF":
                    aux_outputs[aux_type] = f"{self.name}VertexIndex"
                else:
                    aux_outputs[aux_type] = f"{self.name}_VertexIndex"
            elif aux_type == "track_origin":
                aux_outputs[aux_type] = f"{self.name}_TrackOrigin"
            else:
                raise ValueError(f"{aux_type} is not a recognized aux task.")

        return aux_outputs

    def extract_tagger_scores(
        self, source: object, source_type: str = "data_frame", key: str | None = None
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
            self.scores = source[self.variables]
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
            self.scores = df_in[self.variables]

        elif source_type == "h5_file":
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

    def discriminant(self, signal: Flavour, fx: float | None = None):
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
            sig_var = self.variables[self.output_flavours.index(Flavours[signal])]
            return self.scores[sig_var]
        raise ValueError(f"No discriminant defined for {signal} signal.")

    def vertex_indices(self, incl_vertexing=False):
        """Retrieve cleaned vertex indices for the tagger.

        Parameters
        ----------
        incl_vertexing : bool, optional
            Whether to merge all vertex indices, by default False.

        Returns
        -------
        np.ndarray
            Vertex indices for the tagger
        """
        if "vertexing" not in self.aux_tasks:
            raise ValueError("Vertexing aux task not available for this tagger.")
        else:
            truth_indices = self.aux_labels["vertexing"]
            reco_indices = self.aux_scores["vertexing"]

        # clean truth vertex indices - remove indices from true PV, PU, fake
        truth_removal_cond = np.logical_or(
            self.aux_labels["vertexing"] == 0,
            np.isin(self.aux_labels["track_origin"], [0, 1, 2]),
        )
        truth_indices = clean_indices(
            truth_indices,
            truth_removal_cond,
            mode="remove",
        )

        # merge truth vertices from HF for inclusive performance
        if incl_vertexing:
            truth_merge_cond = np.logical_and(
                self.aux_labels["vertexing"] > 0,
                np.isin(self.aux_labels["track_origin"], [3, 4, 5, 6]),
            )
            truth_indices = clean_indices(
                truth_indices,
                truth_merge_cond,
                mode="merge",
            )

        # clean reco vertex indices - remove indices from reco PV, PU, fake
        if "track_origin" in self.aux_tasks:
            reco_indices = clean_indices(
                reco_indices,
                np.isin(self.aux_scores["track_origin"], [0, 1, 2]),
                mode="remove",
            )

            # merge reco vertices - vertices with > 0 from HF if trk origin is available
            if incl_vertexing:
                hf_vertex_indices = np.unique(
                    self.aux_scores["vertexing"][
                        np.isin(self.aux_scores["track_origin"], [3, 4, 5, 6])
                    ]
                )
                # remove remaining vertices without HF tracks
                reco_indices = clean_indices(
                    reco_indices,
                    np.isin(
                        self.aux_scores["vertexing"], hf_vertex_indices, invert=True
                    ),
                    mode="remove",
                )
                # merge remaining vertices with HF tracks
                reco_indices = clean_indices(
                    reco_indices,
                    np.isin(self.aux_scores["vertexing"], hf_vertex_indices),
                    mode="merge",
                )
        else:
            # merge reco vertices - all if track origin isn't available
            if incl_vertexing:
                reco_indices = clean_indices(
                    reco_indices,
                    reco_indices >= 0,
                    mode="merge",
                )

        return truth_indices, reco_indices
