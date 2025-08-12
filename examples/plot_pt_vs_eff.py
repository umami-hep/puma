"""Produce pT vs efficiency plot from tagger output and labels."""

from __future__ import annotations

import numpy as np
from ftag import Flavours
from ftag.utils import get_discriminant

from puma import VarVsEff, VarVsEffPlot
from puma.utils import get_dummy_2_taggers, logger

# The line below generates dummy data which is similar to a NN output
df = get_dummy_2_taggers(add_pt=True)

logger.info("caclulate tagger discriminants")
discs_dips = get_discriminant(
    jets=df,
    tagger="dips",
    signal=Flavours["bjets"],
    flavours=Flavours.by_category("single-btag"),
    fraction_values={
        "fc": 0.018,
        "fu": 0.982,
        "ftau": 0,
    },
)
discs_rnnip = get_discriminant(
    jets=df,
    tagger="rnnip",
    signal=Flavours["bjets"],
    flavours=Flavours.by_category("single-btag"),
    fraction_values={
        "fc": 0.018,
        "fu": 0.982,
        "ftau": 0,
    },
)

# Getting jet pt in GeV
pt = df["pt"] / 1e3

# defining target efficiency
sig_eff = np.linspace(0.49, 1, 20)

# defining boolean arrays to select the different flavour classes
is_b = Flavours["bjets"].cuts(df).idx
is_c = Flavours["cjets"].cuts(df).idx
is_light = Flavours["ujets"].cuts(df).idx
it_tau = Flavours["taujets"].cuts(df).idx

# here the plotting starts

# Define the curves. The signal is here bjets (so b-tagging) and we want
# to look a the light-jet rejection (so background is light jets)
rnnip_light = VarVsEff(
    x_var_sig=pt[is_b],
    disc_sig=discs_rnnip[is_b],
    x_var_bkg=pt[is_light],
    disc_bkg=discs_rnnip[is_light],
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    working_point=0.7,
    fixed_bkg_rej=None,
    disc_cut=None,
    flat_per_bin=False,
    label="RNNIP",
)
dips_light = VarVsEff(
    x_var_sig=pt[is_b],
    disc_sig=discs_dips[is_b],
    x_var_bkg=pt[is_light],
    disc_bkg=discs_dips[is_light],
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    working_point=0.7,
    fixed_bkg_rej=None,
    disc_cut=None,
    flat_per_bin=False,
    label="DIPS",
)

# You can also store and load now the different VarVsEff curves
# Supported formats are .yaml and .json, which are human-readable
# and can be changed before loading them back in.
dips_light.save("dips_light.yaml")
rnnip_light.save("rnnip_light.yaml")

# To load them again, you simply call the class with load().
# Once loaded, you can use them as before. All needed info for plotting are still there
dips_light_loaded = VarVsEff.load("dips_light.yaml")
rnnip_light_loaded = VarVsEff.load("rnnip_light.yaml")

# Now to the actual plotting
logger.info("Plotting bkg rejection for inclusive efficiency as a function of pt.")

# You can choose between different modes:
# "sig_eff": Plots the signal efficiency as a function of your variable
# "bkg_eff": Plots the background mis-tag efficiency as a function of your variable
# "sig_rej": Plots the signal rejection (how much you miss) as a function of your variable
# "bkg_rej": Plots the background rejection as a function of your variable

# We go for light-flavour rejection here and define the actual plot object
plot_bkg_rej = VarVsEffPlot(
    mode="bkg_rej",
    ylabel="Light-flavour jets rejection",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets \ndummy sample, $f_{c}=0.018$",
    n_ratio_panels=1,
)

# Adding now the two already-defined curves and set RNNIP to be the reference
plot_bkg_rej.add(rnnip_light, reference=True)
plot_bkg_rej.add(dips_light)

# Draw the actual curves in the plot object
plot_bkg_rej.draw()

# Save the plot as png
plot_bkg_rej.savefig("pt_light_rej.png")

# Now we also want to plot the signal efficiency. Init a new plot object with "sig_eff"
plot_sig_eff = VarVsEffPlot(
    mode="sig_eff",
    ylabel="$b$-jets efficiency",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets, \ndummy sample, $f_{c}=0.018$",
    n_ratio_panels=1,
)

# Adding now the two already-defined curves and set RNNIP to be the reference
plot_sig_eff.add(rnnip_light, reference=True)
plot_sig_eff.add(dips_light)

# Adapt the text under the ATLAS logo in the plots to say that we used the 70% working point
plot_sig_eff.atlas_second_tag += "\nInclusive $\\epsilon_b=70\\%$"

# If you want to inverse the discriminant cut you can enable it via
# plot_sig_eff.set_inverse_cut()

# Draw the actual curves in the plot object
plot_sig_eff.draw()

# Drawing a hline indicating the inclusive efficiency working point
plot_sig_eff.draw_hline(0.7)

# Save the plot as png. Transparent false will colour the background of the plot white
plot_sig_eff.savefig("pt_b_eff.png", transparent=False)

# We can also plot the performance non-inclusively, so instead of having a global
# signal efficiency of 70% for the signal, we can enforce a 70% signal efficiency in each bin
# For that, we need to redefine our curves by setting the option "flat_per_bin" to True
rnnip_light = VarVsEff(
    x_var_sig=pt[is_b],
    disc_sig=discs_rnnip[is_b],
    x_var_bkg=pt[is_light],
    disc_bkg=discs_rnnip[is_light],
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    working_point=0.7,
    fixed_bkg_rej=None,
    disc_cut=None,
    flat_per_bin=True,
    label="RNNIP",
)
dips_light = VarVsEff(
    x_var_sig=pt[is_b],
    disc_sig=discs_dips[is_b],
    x_var_bkg=pt[is_light],
    disc_bkg=discs_dips[is_light],
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    working_point=0.7,
    fixed_bkg_rej=None,
    disc_cut=None,
    flat_per_bin=True,
    label="DIPS",
)

# Now we can plot the signal efficiency and background rejection
plot_sig_eff = VarVsEffPlot(
    mode="sig_eff",
    ylabel="$b$-jets efficiency",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets, \ndummy sample, $f_{c}=0.018$",
    n_ratio_panels=1,
)

# Adding now the two already-defined curves and set RNNIP to be the reference
plot_sig_eff.add(rnnip_light, reference=True)
plot_sig_eff.add(dips_light)

# Adapt the text under the ATLAS logo in the plots to say that we used the 70% working point
plot_sig_eff.atlas_second_tag += "\nFixed $\\epsilon_b=70\\% per bin$"

# Draw the actual curves in the plot object
plot_sig_eff.draw()

# Save the plot as png
plot_sig_eff.savefig("pt_b_eff_fixed_per_bin.png")

# You will see the plot show now the same signal efficiency for each bin
# Now to the background rejection
plot_bkg_rej = VarVsEffPlot(
    mode="bkg_rej",
    ylabel="Light-flavour jets rejection",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets \ndummy sample, $f_{c}=0.018$",
    n_ratio_panels=1,
)

# Adding now the two already-defined curves and set RNNIP to be the reference
plot_bkg_rej.add(rnnip_light, reference=True)
plot_bkg_rej.add(dips_light)

# Draw the actual curves in the plot object
plot_bkg_rej.draw()

# Save the plot as png
plot_bkg_rej.savefig("pt_light_rej_fixed_per_bin.png")

# Instead of fixing the signal efficiency to a working point, like 70%, you can also
# fix the background rejection. Redefine the curves to enable this
rnnip_light = VarVsEff(
    x_var_sig=pt[is_b],
    disc_sig=discs_rnnip[is_b],
    x_var_bkg=pt[is_light],
    disc_bkg=discs_rnnip[is_light],
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    working_point=None,
    fixed_bkg_rej=100,
    disc_cut=None,
    flat_per_bin=False,
    label="RNNIP",
)
dips_light = VarVsEff(
    x_var_sig=pt[is_b],
    disc_sig=discs_dips[is_b],
    x_var_bkg=pt[is_light],
    disc_bkg=discs_dips[is_light],
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    working_point=None,
    fixed_bkg_rej=100,
    disc_cut=None,
    flat_per_bin=False,
    label="DIPS",
)

# Now we can plot the signal efficiency and background rejection
plot_sig_eff = VarVsEffPlot(
    mode="sig_eff",
    ylabel="$b$-jets efficiency",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets, \ndummy sample, $f_{c}=0.018$",
    n_ratio_panels=1,
)

# Adding now the two already-defined curves and set RNNIP to be the reference
plot_sig_eff.add(rnnip_light, reference=True)
plot_sig_eff.add(dips_light)

# Adapt the text under the ATLAS logo in the plots to say that we used the 70% working point
plot_sig_eff.atlas_second_tag += "\nFixed Rejection 100$"

# Draw the actual curves in the plot object
plot_sig_eff.draw()

# Save the plot as png
plot_sig_eff.savefig("pt_b_eff_rej_100.png")

# You will see the plot show now the same signal efficiency for each bin
# Now to the background rejection
plot_bkg_rej = VarVsEffPlot(
    mode="bkg_rej",
    ylabel="Light-flavour jets rejection",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets \ndummy sample, $f_{c}=0.018$",
    n_ratio_panels=1,
)

# Adding now the two already-defined curves and set RNNIP to be the reference
plot_bkg_rej.add(rnnip_light, reference=True)
plot_bkg_rej.add(dips_light)

# Draw the actual curves in the plot object
plot_bkg_rej.draw()

# Save the plot as png
plot_bkg_rej.savefig("pt_light_rej_rej_100.png")

# Similar to the working point, also a fixed rejection per bin is possible
rnnip_light = VarVsEff(
    x_var_sig=pt[is_b],
    disc_sig=discs_rnnip[is_b],
    x_var_bkg=pt[is_light],
    disc_bkg=discs_rnnip[is_light],
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    working_point=None,
    fixed_bkg_rej=100,
    disc_cut=None,
    flat_per_bin=True,
    label="RNNIP",
)
dips_light = VarVsEff(
    x_var_sig=pt[is_b],
    disc_sig=discs_dips[is_b],
    x_var_bkg=pt[is_light],
    disc_bkg=discs_dips[is_light],
    bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
    working_point=None,
    fixed_bkg_rej=100,
    disc_cut=None,
    flat_per_bin=True,
    label="DIPS",
)

# Now we can plot the signal efficiency and background rejection
plot_sig_eff = VarVsEffPlot(
    mode="sig_eff",
    ylabel="$b$-jets efficiency",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets, \ndummy sample, $f_{c}=0.018$",
    n_ratio_panels=1,
)

# Adding now the two already-defined curves and set RNNIP to be the reference
plot_sig_eff.add(rnnip_light, reference=True)
plot_sig_eff.add(dips_light)

# Adapt the text under the ATLAS logo in the plots to say that we used the 70% working point
plot_sig_eff.atlas_second_tag += "\nFixed Rejection 100 per bin$"

# Draw the actual curves in the plot object
plot_sig_eff.draw()

# Save the plot as png
plot_sig_eff.savefig("pt_b_eff_rej_100_per_bin.png")

# You will see the plot show now the same signal efficiency for each bin
# Now to the background rejection
plot_bkg_rej = VarVsEffPlot(
    mode="bkg_rej",
    ylabel="Light-flavour jets rejection",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets \ndummy sample, $f_{c}=0.018$",
    n_ratio_panels=1,
)

# Adding now the two already-defined curves and set RNNIP to be the reference
plot_bkg_rej.add(rnnip_light, reference=True)
plot_bkg_rej.add(dips_light)

# Draw the actual curves in the plot object
plot_bkg_rej.draw()

# Save the plot as png
plot_bkg_rej.savefig("pt_light_rej_rej_100_per_bin.png")

# Another nice feature for efficiencies where the rejection is fixed, is to show the absolute
# difference. This can be done with the ratio_method "subtract"
plot_sig_eff = VarVsEffPlot(
    mode="sig_eff",
    ylabel="$b$-jets efficiency",
    xlabel=r"$p_{T}$ [GeV]",
    logy=False,
    atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets, \ndummy sample, $f_{c}=0.018$",
    n_ratio_panels=1,
    ratio_method="subtract",
    ylabel_ratio="Difference",
)

# Adding now the two already-defined curves and set RNNIP to be the reference
plot_sig_eff.add(rnnip_light, reference=True)
plot_sig_eff.add(dips_light)

# Draw the actual curves in the plot object
plot_sig_eff.draw()

# Save the plot as png
plot_sig_eff.savefig("pt_light_rej_rej_100_per_bin_subtracted.png")
