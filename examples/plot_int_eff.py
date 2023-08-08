"""Produce Integrated Efficiency curves from tagger output and labels."""
import numpy as np

from puma import IntegratedEfficiency, IntegratedEfficiencyPlot
from puma.utils import get_dummy_2_taggers, logger

# The line below generates dummy data which is similar to a NN output
df = get_dummy_2_taggers()

logger.info("caclulate tagger discriminants")


# define a small function to calculate discriminant
def disc_fct(arr: np.ndarray, f_c: float = 0.018) -> np.ndarray:
    """Tagger discriminant.

    Parameters
    ----------
    arr : numpy.ndarray
        array with with shape (, 3)
    f_c : float, optional
        f_c value in the discriminant (weight for c-jets rejection)

    Returns
    -------
    np.ndarray
        Array with the discriminant values inside.
    """
    # you can adapt this for your needs
    return np.log(arr[2] / (f_c * arr[1] + (1 - f_c) * arr[0]))


# you can also use a lambda function
# discs_rnnip = np.apply_along_axis(
#     lambda a: np.log(a[2] / (0.018 * a[1] + (1 - 0.018) * a[0])),
#     1,
#     df[["rnnip_pu", "rnnip_pc", "rnnip_pb"]].values,
# )

# calculate discriminant
discs_rnnip = np.apply_along_axis(
    disc_fct, 1, df[["rnnip_pu", "rnnip_pc", "rnnip_pb"]].values
)
discs_dips = np.apply_along_axis(
    disc_fct, 1, df[["dips_pu", "dips_pc", "dips_pb"]].values
)
# defining boolean arrays to select the different flavour classes
is_light = df["HadronConeExclTruthLabelID"] == 0
is_c = df["HadronConeExclTruthLabelID"] == 4
is_b = df["HadronConeExclTruthLabelID"] == 5

n_jets_light = sum(is_light)
n_jets_c = sum(is_c)

logger.info("Calculate signal and background discriminant values.")
rnnip = {
    "sig_disc": discs_rnnip[is_b],
    "bkg_disc_b": discs_rnnip[is_b],
    "bkg_disc_c": discs_rnnip[is_c],
    "bkg_disc_l": discs_rnnip[is_light],
}
dips = {
    "sig_disc": discs_dips[is_b],
    "bkg_disc_b": discs_dips[is_b],
    "bkg_disc_c": discs_dips[is_c],
    "bkg_disc_l": discs_dips[is_light],
}

# here the plotting of the Integrated Efficiency curves starts
logger.info("Plotting IntegratedEfficiency curves.")
plot = IntegratedEfficiencyPlot(
    ylabel="Integrated efficiency",
    xlabel="Discriminant",
    atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets \ndummy sample, $f_{c}=0.018$",
    figsize=(6.5, 6),
    y_scale=1.4,
)
plot.add(
    IntegratedEfficiency(
        rnnip["sig_disc"], rnnip["bkg_disc_b"], flavour="bjets", tagger="RRNIP"
    )
)
plot.add(
    IntegratedEfficiency(
        rnnip["sig_disc"], rnnip["bkg_disc_c"], flavour="cjets", tagger="RRNIP"
    )
)
plot.add(
    IntegratedEfficiency(
        rnnip["sig_disc"], rnnip["bkg_disc_l"], flavour="ujets", tagger="RRNIP"
    )
)
plot.add(
    IntegratedEfficiency(
        dips["sig_disc"], dips["bkg_disc_b"], flavour="bjets", tagger="DIPS"
    )
)
plot.add(
    IntegratedEfficiency(
        dips["sig_disc"], dips["bkg_disc_c"], flavour="cjets", tagger="DIPS"
    )
)
plot.add(
    IntegratedEfficiency(
        dips["sig_disc"], dips["bkg_disc_l"], flavour="ujets", tagger="DIPS"
    )
)

plot.draw()
plot.savefig("integrated_efficiency.png", transparent=False)
