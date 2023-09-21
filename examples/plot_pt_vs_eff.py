"""Produce pT vs efficiency plot from tagger output and labels."""
from __future__ import annotations

import numpy as np

from puma import VarVsEff, VarVsEffPlot
from puma.utils import get_dummy_2_taggers, logger


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

# The line below generates dummy data which is similar to a NN output
df = get_dummy_2_taggers(add_pt=True)

# calculate discriminant
discs_rnnip = np.apply_along_axis(
    disc_fct, 1, df[["rnnip_pu", "rnnip_pc", "rnnip_pb"]].values
)
discs_dips = np.apply_along_axis(
    disc_fct,
    1,
    df[["dips_pu", "dips_pc", "dips_pb"]].values,
)

# you can also use a results file directly, you can comment everything above and
# uncomment below
# ttbar_file = "<resultsfile.h5"
# df = pd.read_hdf(ttbar_file, key="ttbar")

# discs_rnnip = df["disc_rnnip"]
# discs_dips = df["disc_dips"]
# is_light = df["labels"] == 0
# is_c = df["labels"] == 1
# is_b = df["labels"] == 2

# Getting jet pt in GeV
pt = df["pt"].values / 1e3
# defining target efficiency
sig_eff = np.linspace(0.49, 1, 20)
# defining boolean arrays to select the different flavour classes
is_light = df["HadronConeExclTruthLabelID"] == 0
is_c = df["HadronConeExclTruthLabelID"] == 4
is_b = df["HadronConeExclTruthLabelID"] == 5

# here the plotting starts

# define the curves
rnnip_light = VarVsEff(
    x_var_sig=pt[is_b],
    disc_sig=discs_rnnip[is_b],
    x_var_bkg=pt[is_light],
    disc_bkg=discs_rnnip[is_light],
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    working_point=0.7,
    disc_cut=None,
    fixed_eff_bin=False,
    label="RNNIP",
)
dips_light = VarVsEff(
    x_var_sig=pt[is_b],
    disc_sig=discs_dips[is_b],
    x_var_bkg=pt[is_light],
    disc_bkg=discs_dips[is_light],
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    working_point=0.7,
    disc_cut=None,
    fixed_eff_bin=False,
    label="DIPS",
)


logger.info("Plotting bkg rejection for inclusive efficiency as a function of pt.")
# You can choose between different modes: "sig_eff", "bkg_eff", "sig_rej", "bkg_rej"
plot_bkg_rej = VarVsEffPlot(
    mode="bkg_rej",
    ylabel="Light-flavour jets rejection",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets \ndummy sample, $f_{c}=0.018$",
    n_ratio_panels=1,
)
plot_bkg_rej.add(rnnip_light, reference=True)
plot_bkg_rej.add(dips_light)

plot_bkg_rej.draw()
plot_bkg_rej.savefig("pt_light_rej.png")

plot_sig_eff = VarVsEffPlot(
    mode="sig_eff",
    ylabel="$b$-jets efficiency",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets, \ndummy sample, $f_{c}=0.018$",
    n_ratio_panels=1,
)
plot_sig_eff.add(rnnip_light, reference=True)
plot_sig_eff.add(dips_light)

plot_sig_eff.atlas_second_tag += "\nInclusive $\\epsilon_b=70%%$"

# If you want to inverse the discriminant cut you can enable it via
# plot_sig_eff.set_inverse_cut()
plot_sig_eff.draw()
# Drawing a hline indicating inclusive efficiency
plot_sig_eff.draw_hline(0.7)
plot_sig_eff.savefig("pt_b_eff.png", transparent=False)
