"""puma contour plot"""
import numpy as np
from atlasify import atlasify
from matplotlib import lines
from matplotlib.figure import Figure

from puma.plot_base import PlotBase
from puma.utils import get_single_colour_cmap, logger


class ContourPlot(PlotBase):  # pylint: disable=too-many-instance-attributes
    """2D contour plot that projects the variables onto the axes with histograms"""

    def __init__(
        self,
        x_range: tuple,
        y_range: tuple,
        left: float = 0.2,
        width: float = 0.55,
        height: float = 0.55,
        bottom: float = 0.2,
        spacing: float = 0.005,
        width_hist: float = 0.2,
        bins: int = 20,
        bins2d: int = 20,
        **kwargs,
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """Sets up the figure and axes

        Parameters
        ----------
        x_range : tuple
            x-range
        y_range : tuple
            y-range
        left : float, optional
            Whitespace to the left of the figure, by default 0.2
        width : float, optional
            Width of the main figure (relative fraction of total width), by default 0.55
        height : float, optional
            Height of the main figure, by default 0.55
        bottom : float, optional
                Whitespace to the bottom of the figure, by default 0.2
        spacing : float, optional
            Spacing between main figure and the histogram panels, by default 0.005
        width_hist : float, optional
            Width of the histogram panels, by default 0.2
        bins : int, optional
            Number of bins along the x/y projections, by default 20
        bins2d : int, optional
            Number of bins in the 2D plane. Choosing 10 will result in 10x10=100 bins.
            By default 20
        **kwargs : kwargs
            Keyword arguments from `puma.PlotObject`
        """

        super().__init__(**kwargs)

        self.x_range = x_range
        self.y_range = y_range
        self.bins = bins
        self.bins2d = bins2d

        # initialise figure and add axes (main plot + x/y projection axes)
        self.fig = Figure(figsize=self.figsize)
        # rect: [left, bottom, width, height] of the "box"
        rect_scatter = [left, bottom, width, height]
        rect_histy = [left + width + spacing, bottom, width_hist, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        self.ax_contour = self.fig.add_axes(rect_scatter)
        self.ax_histx = self.fig.add_axes(rect_histx, sharex=self.ax_contour)
        self.ax_histy = self.fig.add_axes(rect_histy, sharey=self.ax_contour)
        self.ax_histx.axis("off")
        self.ax_histy.axis("off")
        self.handles = []

    def add(
        self,
        x_values,
        y_values,
        label=None,
        colour="b",
        cmap=None,
    ):  # pylint: disable=too-many-arguments,too-many-locals
        """Add another dataset to the contour plot

        Parameters
        ----------
        x_values : array
            x values
        y_values : array
            y values
        label : str, optional
            Label of this dataset, by default None
        colour : str, optional
            Colour for the projection histograms, by default "b"
        cmap : colourmap, optional
            Matplotlib colourmap, by default None
        """
        # https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html
        self.ax_histy.tick_params(axis="y", labelleft=False)
        # ax.scatter(x, y, s=10, label=label, color=colour)
        hist2d, binsx2d, binsy2d = np.histogram2d(
            x_values,
            y_values,
            bins=self.bins2d,
            range=(self.x_range, self.y_range),
            density=True,
        )
        bin_width_x = binsx2d[1] - binsx2d[0]
        bin_width_y = binsy2d[1] - binsy2d[0]
        bin_area = (binsx2d[1] - binsx2d[0]) * (binsy2d[1] - binsy2d[0])
        # multiply with bin area such that bin entry represents fraction of entries
        hist2d = hist2d * bin_area
        x_values_mesh, y_values_mesh = np.meshgrid(
            binsx2d[:-1] + bin_width_x / 2, binsy2d[:-1] + bin_width_y / 2
        )
        self.ax_contour.contour(
            x_values_mesh,
            y_values_mesh,
            hist2d.T,
            cmap=cmap if cmap is not None else get_single_colour_cmap(colour),
            levels=np.linspace(
                hist2d.min() + 1e-5 * (hist2d.max() - hist2d.min()),
                hist2d.max(),
                10,
            ),
            linewidths=1.4,
        )
        # this could be easily extended to have an option "style" which also allows
        # scatter plots (but the tend to get messy with many data points)
        # elif style == "scatter":
        #     self.ax_contour.scatter(x_values, y_values, color=colour)
        histx, bin_edges_x = np.histogram(
            x_values, bins=self.bins, range=self.x_range, density=True
        )
        histy, bin_edges_y = np.histogram(
            y_values, bins=self.bins, range=self.y_range, density=True
        )
        self.ax_histx.hist(
            bin_edges_x[:-1],
            bins=bin_edges_x,
            weights=histx,
            orientation="vertical",
            histtype="step",
            color=colour,
        )
        self.ax_histy.hist(
            bin_edges_y[:-1],
            bins=bin_edges_y,
            weights=histy,
            orientation="horizontal",
            histtype="step",
            color=colour,
        )
        self.handles.append(
            lines.Line2D(
                [],
                [],
                color=colour,
                label=label,
            )
        )

    def draw(self):
        """Draw the lines"""
        self.ax_contour.set_xlabel(self.xlabel)
        self.ax_contour.set_ylabel(self.ylabel)
        self.ax_contour.grid(which="major")
        self.ax_contour.add_artist(
            self.ax_contour.legend(
                handles=self.handles,
                labels=[handle.get_label() for handle in self.handles],
                loc="upper center",
                ncol=2,
                fontsize=9,
            )
        )
        atlasify(axes=self.ax_contour, enlarge=1, brand="")

    def draw_bins(self):
        """Debug function to draw the bins"""
        # TODO: remove this
        for x_bin in np.linspace(*self.x_range, self.bins):
            self.ax_contour.axvline(x_bin, color="#000000", linestyle="--")

    def savefig(
        self,
        plot_name: str,
        transparent: bool = None,
        dpi: int = None,
        **kwargs,
    ):
        """Save plot to disk.

        Parameters
        ----------
        plot_name : str
            File name of the plot
        transparent : bool, optional
            Specify if plot background is transparent, by default False
        dpi : int, optional
            DPI for plotting, by default 400
        **kwargs : kwargs
            Keyword arguments passed to `matplotlib.figure.Figure.savefig()`
        """
        logger.debug("Saving plot to %s", plot_name)
        self.fig.savefig(
            plot_name,
            transparent=self.transparent if transparent is None else transparent,
            dpi=self.dpi if dpi is None else dpi,
            **kwargs,
        )
