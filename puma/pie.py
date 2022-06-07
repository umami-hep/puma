"""Pie plot functions."""
import matplotlib as mpl

from puma.plot_base import PlotBase
from puma.utils import get_good_pie_colours


class PiePlot(
    PlotBase
):  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """
    Histogram class storing info about histogram and allows to calculate ratio w.r.t
    other histograms.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        fracs,
        colours: list = None,
        colour_scheme: str = None,
        labels: list = None,
        vertical_split: bool = True,
        **kwargs,
    ):
        super().__init__(vertical_split=vertical_split, **kwargs)
        self.fracs = fracs
        self.colours = (
            colours
            if colours is not None
            else get_good_pie_colours(colour_scheme)[: len(fracs)]
        )
        self.labels = labels if labels is not None else ["" for i in range(len(fracs))]

        self.initialise_figure()
        self.plot()

    def plot(
        self,
    ):
        """
        Plot the pie chart
        """

        self.axis_top.pie(
            x=self.fracs,
            labels=None,
            colors=self.colours,
            autopct="%1.1f%%",
        )

        self.axis_leg.axis("off")

        plt_handles = []

        for pie_label, pie_colour in zip(self.labels, self.colours):
            plt_handles.append(
                mpl.patches.Patch(
                    label=pie_label,
                    color=pie_colour,
                )
            )

        self.plotting_done = True
        self.make_legend(plt_handles, ax_mpl=self.axis_leg)
        self.set_title()
        self.fig.tight_layout()

        if self.apply_atlas_style:
            self.atlasify()
