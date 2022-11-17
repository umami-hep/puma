"""Tagger module for high level API."""
# import h5py
import numpy as np

from puma.metrics import calc_rej


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

        Raises
        ------
        KeyError
            If template contains keys which are not a class attribute.
        """
        self.model_name = model_name
        self.label = None
        self.reference = False

        self.colour = None

        self.discs = None

        self.disc_cut = None
        self.working_point = None

        self.is_b = None
        self.is_light = None
        self.is_c = None

        self.ujets_rej = None
        self.cjets_rej = None

        if template is not None:
            for key, val in template.items():
                if hasattr(self, key):
                    setattr(self, key, val)
                else:
                    raise KeyError(f"`{key}` is not an attribute of the Tagger class.")

    def calc_rej(self, sig_eff):
        """Calculate c and light rejection.

        Parameters
        ----------
        sig_eff : array
            signal efficiency
        """
        if self.discs is None:
            raise ValueError(
                "You need to first specify discs to calculate the rejection."
            )
        self.ujets_rej = calc_rej(
            self.discs[self.is_b],  # pylint: disable=unsubscriptable-object
            self.discs[self.is_light],  # pylint: disable=unsubscriptable-object
            sig_eff,
        )
        self.cjets_rej = calc_rej(
            self.discs[self.is_b],  # pylint: disable=unsubscriptable-object
            self.discs[self.is_c],  # pylint: disable=unsubscriptable-object
            sig_eff,
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
