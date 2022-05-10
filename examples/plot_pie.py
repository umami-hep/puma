from puma import histogram, histogram_plot
from puma.utils import get_dummy_2_taggers

# The line below generates dummy data which is similar to a NN output
df = get_dummy_2_taggers(size=12)

HadrTruthLabel_vals = [0, 4, 5, 15]
HadrTruthLabel_labels = ["light-flavour jets", "c-jets", "b-jets", "tau-jets"]
title = "HadronConeExclTruthLabelID"

# the number of bins should be the number of bins needed to have a separat bin
# for every discrete value.
bins = 16
bins_range = (0, 16)

plot_pie = histogram_plot(
    n_ratio_panels=0,
    title=title,
    discrete_vals=HadrTruthLabel_vals,
    atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow Jets,\n$t\\bar{t}$ Test Sample",
    bins=bins,
    bins_range=bins_range,
    draw_errors=False,
    vertical_split=True,
    plot_pie=True,
    pie_colours=None,
    pie_labels=HadrTruthLabel_labels,
)

plot_pie.add(histogram(df["HadronConeExclTruthLabelID"]))
plot_pie.draw()
plot_pie.savefig("pie_chart_HadronConeExclTruthLabelID.png", transparent=False)
