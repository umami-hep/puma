import h5py
import numpy as np

from puma.metrics import calc_rej
from puma.utils.histogram import save_divide

# from umami.metrics import get_score

# from puma.utils import global_config, logger

# TODO: for now lots of things are hard coded, e.g. the 70% WP


def decorate_df(df, result_file, model_name):
    with h5py.File(result_file, "r") as file:
        net_probs = file["probs"][: len(df)]
        df[f"{model_name}_pu"] = net_probs[:, 0]
        df[f"{model_name}_pc"] = net_probs[:, 1]
        df[f"{model_name}_pb"] = net_probs[:, 2]


class Tagger:
    """Class storing tagger results."""

    def __init__(self, model_name: str) -> None:
        """Init Tagger class.

        Parameters
        ----------
        model_name : str
            Name of the model, also correspondinng to the pre-fix of the tagger
            variables.
        """
        self.model_name = model_name
        self.label = None
        self.f_c = None
        self.discs = None
        self.ujets_rej = None
        self.cjets_rej = None
        self.reference = False
        self.flvs = ["_pu", "_pc", "_pb"]
        self.class_labels = ["ujets", "cjets", "bjets"]
        self.disc_cut = None
        self.working_point = None
        self.colour = None

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

    def calc_rej(self, sig_eff, is_b, is_light, is_c):
        """Calculate c and ligth rejection.

        Parameters
        ----------
        sig_eff : array
            signal efficiency
        is_b : bool array
            array of bool values indicating if b jet
        is_light : bool array
            array of bool values indicating if light jet
        is_c : bool array
            array of bool values indicating if c jet
        """
        self.ujets_rej = calc_rej(self.discs[is_b], self.discs[is_light], sig_eff)
        self.cjets_rej = calc_rej(self.discs[is_b], self.discs[is_c], sig_eff)
