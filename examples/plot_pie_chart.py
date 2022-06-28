from puma.pie import PiePlot

HadrTruthLabel_fracs = [70, 10, 15, 5]
HadrTruthLabel_labels = ["Light-flavour jets", "$c$-jets", "$b$-jets", "$\\tau$-jets"]

example_plot_1 = PiePlot(
    fracs=HadrTruthLabel_fracs,
    labels=HadrTruthLabel_labels,
    draw_legend=False,
    # have a look at the possible kwargs for matplotlib.pyplot.pie here:
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html
    mpl_pie_kwargs={
        "explode": [0, 0.2, 0, 0.1],
        "shadow": False,
        "startangle": 90,
        "textprops": {"fontsize": 10},
        "radius": 1,
        "wedgeprops": dict(width=0.4, edgecolor="w"),
        "pctdistance": 0.4,
    },
    # kwargs passed to puma.PlotObject
    atlas_second_tag="Dummy flavour fractions",
    figsize=(7, 5),
    ymax=1.5,
)
example_plot_1.savefig("pie_example_1.png")


# Another example with other styling
example_plot_2 = PiePlot(
    fracs=HadrTruthLabel_fracs,
    labels=HadrTruthLabel_labels,
    draw_legend=True,
    colour_scheme="blue",
    # have a look at the possible kwargs for matplotlib.pyplot.pie here:
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html
    mpl_pie_kwargs={
        "textprops": {"fontsize": 10},
        "radius": 1,
        "wedgeprops": dict(width=1, edgecolor="w"),
        "pctdistance": 1.2,
    },
    # kwargs passed to puma.PlotObject
    atlas_second_tag="Dummy flavour fractions",
    figsize=(7, 5),
    ymax=1.5,
)
example_plot_2.savefig("pie_example_2.png")
