"""Example plot script for flavour probability comparison."""

from pathlib import Path

import h5py
import pandas as pd

from puma import Histogram, HistogramPlot

# from puma.utils import get_dummy_2_taggers

# The line below generates dummy data which is similar to a NN output
# df = get_dummy_2_taggers()


pathi = Path(
    "/srv/beegfs/scratch/groups/dpnc/atlas/FTag/samples/r22/p5169/user.pgadow.410470."
    "e6337_s3681_r13144_p5169.tdd.EMPFlow.22_2_82.22-07-28_ftagv00_output.h5"
)
sample = pathi / "user.pgadow.29879654._000028.output.h5"

with h5py.File(sample, "r") as f_h5:
    df = pd.DataFrame(f_h5["jets"][:100_000])
print("-" * 8)
# print(df.columns.values)
# Initialise histogram plot
plot_histo = HistogramPlot(
    n_ratio_panels=0,
    ylabel="Normalised number of jets",
    xlabel="absEta_btagJes",
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
plot_histo.add(Histogram(u_jets["absEta_btagJes"], flavour="ujets", linestyle="dashed"))
plot_histo.add(
    Histogram(c_jets["absEta_btagJes"], flavour="cjets", linestyle="dashdot")
)
plot_histo.add(Histogram(b_jets["absEta_btagJes"], flavour="bjets"))

plot_histo.draw()
plot_histo.savefig("absEta_btagJes.png", transparent=False)
