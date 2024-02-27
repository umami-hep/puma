"""IntegratedEfficiency functions."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from ftag import Flavour, Flavours

from puma.metrics import calc_eff
from puma.plot_base import PlotBase, PlotLineObject
from puma.utils import get_good_colours, get_good_linestyles, logger


class IntegratedEfficiency(PlotLineObject):
    """Represent a single IntegratedEfficiency curve."""

    def __init__(
        self,
        disc_sig: np.ndarray,
        disc_bkg: np.ndarray,
        key: str | None = None,
        n_vals: int = 500,
        tagger: str | None = None,
        flavour: str | Flavour = None,
        **kwargs,
    ) -> None:
        """Initialise properties of IntegratedEfficiency object.

        Parameters
        ----------
        disc_sig : np.array
            Discriminant values for signal
        disc_bkg : np.array
            Discriminant values for background
        key : str
            Identifier for IntegratedEfficiency e.g. tagger, by default None
        n_vals : int, optional
            Number of values to calculate the efficiency at, by default 500
        tagger : str, optional
            Tagger name, by default None
        flavour : str or Flavour, optional
            Flavour of the jets, by default None
        **kwargs : kwargs
            Keyword arguments passed to `puma.PlotLineObject`

        Raises
        ------
        ValueError
            If `sig_eff` and `bkg_rej` have a different shape
        """
        super().__init__(**kwargs)
        self.disc_sig = np.asarray(disc_sig)
        self.disc_bkg = np.asarray(disc_bkg)
        self.n_vals = n_vals
        self.tagger = tagger
        self.key = key
        self.flavour = Flavours[flavour] if isinstance(flavour, str) else flavour
        if self.label is None and self.flavour is not None:
            self.label = self.flavour.label
        self._calc_profile()

    def _calc_profile(self):
        """Calculate the profile of the integrated efficiency curve."""
        self.eff, self.x = calc_eff(
            self.disc_sig,
            self.disc_bkg,
            np.linspace(0, 1, self.n_vals),
            return_cuts=True,
        )


class IntegratedEfficiencyPlot(PlotBase):
    """IntegratedEfficiencyPlot class."""

    def __init__(self, grid: bool = True, **kwargs) -> None:
        """IntegratedEfficiency plot properties.

        Parameters
        ----------
        grid : bool, optional
            Set the grid for the plots.
        **kwargs : kwargs
            Keyword arguments from `puma.PlotObject`
        """
        super().__init__(grid=grid, **kwargs)
        self.int_effs = {}
        self.tagger_ls = {}
        self.label_colours = {}
        self.leg_tagger_labels = {}
        self.initialise_figure()
        self.disc_min, self.disc_max = (1e3, -1e3)
        self.default_linestyles = get_good_linestyles()
        self.legend_flavs = None
        self.leg_tagger_loc = "lower left"

        self.ymin = 0
        self.ymax = 1.2

    def add(self, int_eff: object, key: str | None = None):
        """Adding puma.Roc object to figure.

        Parameters
        ----------
        int_effs : puma.IntegratedEfficiency
            IntegratedEfficiency curve
        key : str, optional
            Unique identifier for IntegratedEfficiency curve, by default None

        Raises
        ------
        KeyError
            If unique identifier key is used twice
        """
        if key is None:
            key = len(self.int_effs) + 1
        if key in self.int_effs:
            raise KeyError(f"Duplicated key {key} already used for roc unique identifier.")

        self.int_effs[key] = int_eff
        # set linestyle
        if int_eff.tagger not in self.tagger_ls:
            self.tagger_ls[int_eff.tagger] = (
                self.default_linestyles[len(self.tagger_ls)]
                if int_eff.linestyle is None
                else int_eff.linestyle
            )
        elif int_eff.linestyle != self.tagger_ls[int_eff.tagger] and int_eff.linestyle is not None:
            logger.warning(
                "You specified a different linestyle for the same tagger"
                " %s. This will lead to a mismatch in the line colours"
                " and the legend.",
                int_eff.tagger,
            )
        if int_eff.linestyle is None:
            int_eff.linestyle = self.tagger_ls[int_eff.tagger]

        # set colours
        if int_eff.label not in self.label_colours:
            if int_eff.flavour is not None:
                self.label_colours[int_eff.label] = int_eff.flavour.colour
            else:
                curr_colours = set(self.label_colours.values())
                possible_colours = set(get_good_colours()) - curr_colours
                self.label_colours[int_eff.label] = (
                    possible_colours.pop() if int_eff.colour is None else int_eff.colour
                )
        elif int_eff.colour != self.label_colours[int_eff.label] and int_eff.colour is not None:
            logger.warning(
                "You specified a different colour for the same label"
                " %s. This will lead to a mismatch in the line colours"
                " and the legend.",
                int_eff.label,
            )
        if int_eff.colour is None:
            int_eff.colour = self.label_colours[int_eff.label]

    def get_xlim_auto(self):
        """Returns min and max efficiency values.

        Returns
        -------
        float
            Min and max efficiency values
        """
        for elem in self.int_effs.values():
            self.disc_min = min(np.min(elem.x), self.disc_min)
            self.disc_max = max(np.max(elem.x), self.disc_max)

        return self.disc_min, self.disc_max

    def make_legend(self, handles: list):
        """Make legend.

        Parameters
        ----------
        handles : list
            List of handles
        """
        line_list_tagger = [
            mpl.lines.Line2D(
                [],
                [],
                color="k",
                linestyle=self.tagger_ls[tagger],
                label=tagger,
            )
            for tagger in self.tagger_ls
        ]
        self.legend_flavs = self.axis_top.legend(
            handles=line_list_tagger,
            labels=[handle.get_label() for handle in line_list_tagger],
            loc=self.leg_tagger_loc,
            fontsize=self.leg_fontsize,
            ncol=self.leg_ncol,
        )
        self.axis_top.add_artist(self.legend_flavs)
        # Get the labels for the legends
        labels_list = []
        lines_list = []

        for line in handles:
            if line.get_label() not in labels_list:
                labels_list.append(line.get_label())
                lines_list.append(line)

        # Define the legend
        self.axis_top.legend(
            handles=lines_list,
            labels=labels_list,
            loc=self.leg_loc,
            fontsize=self.leg_fontsize,
            ncol=self.leg_ncol,
        )

    def draw(
        self,
        x_label: str = "Discriminant",
    ):
        """Draw plotting.

        Parameters
        ----------
        x_label : str, optional
            x-axis label, by default Discriminant
        """
        plt_handles = self.plot()
        xmin, xmax = self.get_xlim_auto()

        self.set_xlim(
            xmin if self.xmin is None else self.disc_min,
            xmax if self.xmax is None else self.disc_max,
        )
        self.set_title()
        self.set_y_lim()
        # self.set_log()
        self.set_y_lim()
        self.set_xlabel(label=x_label)
        self.set_ylabel(self.axis_top, label="Integrated efficiency")

        self.make_legend(plt_handles)

        self.plotting_done = True
        if self.apply_atlas_style is True:
            self.atlasify()
            # atlasify can only handle one legend. Therefore, we remove the frame of
            # the second legend by hand
            if self.legend_flavs is not None:
                self.legend_flavs.set_frame_on(False)

    def plot(self, **kwargs) -> mpl.lines.Line2D:
        """Plotting integrated efficiency curves.

        Parameters
        ----------
        **kwargs: kwargs
            Keyword arguments passed to plt.axis.plot

        Returns
        -------
        Line2D
            matplotlib Line2D object
        """
        plt_handles = []
        for key, elem in self.int_effs.items():
            plt_handles += self.axis_top.plot(
                elem.x,
                elem.eff,
                linestyle=elem.linestyle,
                color=elem.colour,
                label=elem.label if elem is not None else key,
                zorder=2,
                **kwargs,
            )
        return plt_handles
