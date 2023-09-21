"""Example plot script for flavour probability comparison."""
from __future__ import annotations

from puma import Histogram, HistogramPlot
from puma.utils import get_dummy_2_taggers

# The line below generates dummy data which is similar to a NN output
df = get_dummy_2_taggers()

# Initialise histogram plot
plot_histo = HistogramPlot(
    n_ratio_panels=0,
    ylabel="Normalised number of jets",
    xlabel="$b$-jets probability",
    logy=True,
    leg_ncol=1,
    atlas_first_tag="Simulation, $\\sqrt{s}=13$ TeV",
    atlas_second_tag="dummy sample, dummy jets",
    atlas_brand=None,  # You can deactivate the ATLAS branding (e.g. for a thesis)
    draw_errors=False,
    # bins=np.linspace(0, 1, 30),  # you can also force a binning for the plot here
)

# Add the ttbar histograms
u_jets = df.query("HadronConeExclTruthLabelID==0")
c_jets = df.query("HadronConeExclTruthLabelID==4")
b_jets = df.query("HadronConeExclTruthLabelID==5")

# the "flavour" argument will add a "light-flavour jets" (or other) prefix to the label
# + set the colour to the one that is defined in puma.utils.global_config
plot_histo.add(Histogram(u_jets["dips_pb"], flavour="ujets", linestyle="dashed"))
plot_histo.add(Histogram(c_jets["dips_pb"], flavour="cjets", linestyle="dashdot"))
plot_histo.add(Histogram(b_jets["dips_pb"], flavour="bjets"))

plot_histo.draw()
plot_histo.savefig("histogram_bjets_probability.png", transparent=False)
