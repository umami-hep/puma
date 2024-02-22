"""Produce Integrated Efficiency curves from tagger output and labels."""

from __future__ import annotations

from ftag import get_discriminant

from puma import IntegratedEfficiency, IntegratedEfficiencyPlot
from puma.utils import get_dummy_2_taggers, logger

# The line below generates dummy data which is similar to a NN output
df = get_dummy_2_taggers()

logger.info("caclulate tagger discriminants")
discs_rnnip = get_discriminant(df, "rnnip", signal="bjets", fc=0.018)
discs_dips = get_discriminant(df, "dips", signal="bjets", fc=0.018)

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
