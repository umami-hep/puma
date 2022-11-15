# import h5py
import numpy as np

from puma.metrics import calc_rej
from puma.utils.histogram import save_divide

# from umami.metrics import get_score

# from puma.utils import global_config, logger

# TODO: for now lots of things are hard coded, e.g. the 70% WP


class Tagger:
    """Class storing tagger results."""

    def __init__(self, model_name: str, template: dict = None) -> None:
        """Init Tagger class.

        Parameters
        ----------
        model_name : str
            Name of the model, also correspondinng to the pre-fix of the tagger
            variables.
        """
        self.model_name = model_name
        self.label = None
        self.c_tagging = False
        self.f_c = None

        self.discs = None

        self.ujets_rej = None
        self.cjets_rej = None

        self.reference = False

        self.disc_cut = None
        self.working_point = None
        self.colour = None

        self.is_b = None
        self.is_light = None
        self.is_c = None

    def calc_discs(self, df):
        """Calculate b-tagging discriminant."""
        # fixing here only 1 fc value
        # frac_dict = {"cjets": self.f_c, "ujets": 1 - self.f_c}
        arr = df[[self.model_name + fl for fl in self.flvs]].values
        self.discs = np.log(
            save_divide(
                arr[:, 2],
                self.f_c * arr[:, 1] + (1 - self.f_c) * arr[:, 0],
                default=np.infty,
            )
        )

    def calc_rej(self, sig_eff):
        """Calculate c and ligth rejection.

        Parameters
        ----------
        sig_eff : array
            signal efficiency
        """
        self.ujets_rej = calc_rej(
            self.discs[self.is_b],
            self.discs[self.is_light],
            sig_eff,
        )
        self.cjets_rej = calc_rej(
            self.discs[self.is_b],
            self.discs[self.is_c],
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
