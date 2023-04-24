"""Produce roc curves from tagger output and labels."""
import numpy as np

from puma import Roc, RocPlot
from puma.metrics import calc_rej
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
# defining target efficiency
sig_eff = np.linspace(0.49, 1, 20)
# defining boolean arrays to select the different flavour classes
is_light = df["HadronConeExclTruthLabelID"] == 0
is_c = df["HadronConeExclTruthLabelID"] == 4
is_b = df["HadronConeExclTruthLabelID"] == 5

n_jets_light = sum(is_light)
n_jets_c = sum(is_c)

logger.info("Calculate rejection")
rnnip_ujets_rej = calc_rej(discs_rnnip[is_b], discs_rnnip[is_light], sig_eff)
rnnip_cjets_rej = calc_rej(discs_rnnip[is_b], discs_rnnip[is_c], sig_eff)
dips_ujets_rej = calc_rej(discs_dips[is_b], discs_dips[is_light], sig_eff)
dips_cjets_rej = calc_rej(discs_dips[is_b], discs_dips[is_c], sig_eff)

# Alternatively you can simply use a results file with the rejection values
# logger.info("read h5")
# df = pd.read_hdf("results-rej_per_eff-1_new.h5", "ttbar")
# print(df.columns.values)
# sig_eff = df["effs"]
# rnnip_ujets_rej = df["rnnip_ujets_rej"]
# rnnip_cjets_rej = df["rnnip_cjets_rej"]
# dips_ujets_rej = df["dips_ujets_rej"]
# dips_cjets_rej = df["dips_cjets_rej"]
# n_test = 10_000

# here the plotting of the roc starts
logger.info("Plotting ROC curves.")
plot_roc = RocPlot(
    n_ratio_panels=2,
    ylabel="Background rejection",
    xlabel="$b$-jet efficiency",
    atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets \ndummy sample, $f_{c}=0.018$",
    figsize=(6.5, 6),
    y_scale=1.4,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        rnnip_ujets_rej,
        n_test=n_jets_light,
        rej_class="ujets",
        signal_class="bjets",
        label="RNNIP",
    ),
    reference=True,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        dips_ujets_rej,
        n_test=n_jets_light,
        rej_class="ujets",
        signal_class="bjets",
        label="DIPS r22",
    ),
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        rnnip_cjets_rej,
        n_test=n_jets_c,
        rej_class="cjets",
        signal_class="bjets",
        label="RNNIP",
    ),
    reference=True,
)
plot_roc.add_roc(
    Roc(
        sig_eff,
        dips_cjets_rej,
        n_test=n_jets_c,
        rej_class="cjets",
        signal_class="bjets",
        label="DIPS r22",
    ),
)
# setting which flavour rejection ratio is drawn in which ratio panel
plot_roc.set_ratio_class(1, "ujets")
plot_roc.set_ratio_class(2, "cjets")
# if you want to swap the ratios just uncomment the following 2 lines
# plot_roc.set_ratio_class(2, "ujets")
# plot_roc.set_ratio_class(1, "cjets")

plot_roc.draw()
plot_roc.savefig("roc.png", transparent=False)
