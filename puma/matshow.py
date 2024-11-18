from __future__ import annotations

import math as m

import numpy as np
from matplotlib import pyplot as plt

from puma.plot_base import PlotBase
from puma.utils import logger


class MatshowPlot(PlotBase):
    """Plot Matrix class."""

    def __init__(
        self,
        x_ticklabels: list | None = None,
        x_ticks_rotation: int = 90,
        y_ticklabels: list | None = None,
        show_entries: bool = True,
        show_percentage: bool = False,
        text_color_threshold: float = 0.408,
        colormap: plt.cm = plt.cm.Oranges,
        show_cbar: bool = True,
        cbar_label: str | None = None,
        **kwargs,
    ) -> None:
        """Plot a matrix with matplotlib matshow.

        Parameters
        ----------
        x_ticklabels : list | None, optional
            Names of the matrix's columns; if None, indices are shown. by default None
        x_ticks_rotation : int, optional
            Rotation of the columns' names, by default 90
        y_ticklabels : list | None, optional
            Names of the matrix's rows; if None, indices are shown. by default None
        show_entries : bool, optional
            If True, show matrix entries as numbers in the matrix pixels. by default True
        show_percentage : bool, optional
            If True, if matrix entries are percentages (i.e. numbers in [0,1]), format them as
            percentages. by default False
        text_color_threshold : float, optional
            threshold on the relative luminance of the colormap bkg color after which the text color
            switches to black, to allow better readability on lighter cmap bkg colors.
            If 1, all text is white; if 0, all text is black. by default 0.408
        colormap : plt.cm, optional
            Colormap for the plot, by default `plt.cm.Oranges`
        show_cbar : bool, optional
            Whether to plot the colorbar or not, by default True
        cbar_label : str | None, optional
            Label of the colorbar, by default None
        **kwargs : kwargs
            Keyword arguments for `puma.PlotObject`

        Example
        -------
        >>> matrix_plotter = MatshowPlot()
        >>> mat = np.random.rand(4, 3)
        >>> matrix_plotter.draw(mat)
        """
        super().__init__(**kwargs)

        self.x_ticklabels = x_ticklabels
        self.x_ticks_rotation = x_ticks_rotation
        self.y_ticklabels = y_ticklabels
        self.show_entries = show_entries
        self.show_percentage = show_percentage
        self.text_color_threshold = text_color_threshold
        self.colormap = colormap
        self.show_cbar = show_cbar
        self.cbar_label = cbar_label

        # Specifying figsize if not specified by user
        if self.figsize is None:
            self.figsize = (10, 10.5)
        self.initialise_figure()

    def __get_luminance(self, rgbaColor):
        """Calculate the relative luminance of a color according to W3C standards.
        For the details of the conversion see: https://www.w3.org/WAI/GL/wiki/Relative_luminance .

        Parameters
        ----------
        rgbColor : tuple
            (r,g,b,a) color (returned from `plt.cm` colormap)

        Returns
        -------
        float
            Relative luminance of the color.
        """
        # Converting to np.ndarray, ignoring alpha channel
        rgbaColor = np.array(rgbaColor[:-1])
        rgbaColor = np.where(
            rgbaColor <= 0.03928, rgbaColor / 12.92, ((rgbaColor + 0.055) / 1.055) ** 2.4
        )
        weights = np.array([0.2126, 0.7152, 0.0722])
        return np.dot(rgbaColor, weights)

    def __plot(self, matrix):
        """Plot the Matrix."""
        n_cols = matrix.shape[1]
        n_rows = matrix.shape[0]

        im = self.axis_top.matshow(matrix, cmap=self.colormap)

        # If mat entries have to be plotted
        if self.show_entries:
            # Mapping mat values in [0,1], as it's done by matplotlib
            # to associate them to the colors of the colormap
            normMat = matrix - np.min(matrix)
            normMat /= np.max(matrix) - np.min(matrix)

            # Adding text values in the matrix pixels
            for i in range(n_rows):
                for j in range(n_cols):
                    # Choosing the text color: black if color is light, white if color is dark
                    # Getting the bkg color from the cmap
                    color = self.colormap(normMat[i, j])
                    # Calculating the bkg relative luminance
                    luminance = self.__get_luminance(color)
                    # Choosing the appropriate color
                    color = "white" if luminance <= self.text_color_threshold else "black"

                    # If matrix entry is an int, do not show decimals
                    if m.modf(matrix[i, j])[0] == 0:
                        text = f"{matrix[i, j]:.0f}"
                    # Else, round it or show it as percentage
                    else:
                        text = (
                            f"{matrix[i, j]:.3f}"
                            if not self.show_percentage
                            else f"{matrix[i, j] * 100:.0f}%"
                        )
                    # Plotting text
                    self.axis_top.text(
                        x=j,
                        y=i,
                        s=text,
                        va="center",
                        ha="center",
                        c=color,
                        fontsize=self.fontsize,
                    )
        if self.show_cbar:
            # Plotting colorbar
            cbar = self.fig.colorbar(im)
            # If using percentages, converting cbar labels to percentages
            if self.show_entries and self.show_percentage:
                minMat = np.min(matrix)
                maxMat = np.max(matrix)
                cbar.set_ticks(
                    ticks=np.linspace(minMat, maxMat, 5),
                    labels=[f"{i}%" for i in np.round(np.linspace(minMat, maxMat, 5) * 100, 2)],
                    fontsize=self.fontsize,
                )
            if self.cbar_label is not None:
                cbar.ax.set_ylabel(self.cbar_label, fontsize=self.fontsize)

        # Setting tick labels
        if self.x_ticklabels is None:
            self.x_ticklabels = [str(i) for i in range(n_cols)]
            logger.info("MatshowPlot: no x_ticklabels given, using indices instead.")
        if self.y_ticklabels is None:
            self.y_ticklabels = [str(i) for i in range(n_rows)]
            logger.info("MatshowPlot: no y_ticklabels given, using indices instead.")

        # Writing class names on the axes
        positions = list(range(n_cols))
        self.axis_top.set_xticks(
            np.array(positions) + 0.25,
            labels=self.x_ticklabels,
            rotation=self.x_ticks_rotation,
            fontsize=self.fontsize,
            ha="right",
        )
        positions = list(range(n_rows))
        self.axis_top.set_yticks(positions, labels=self.y_ticklabels, fontsize=self.fontsize)
        # Put xticks to the bottom
        self.axis_top.xaxis.tick_bottom()

        # Finished plotting, can apply atlas_style
        self.plotting_done = True
        # Applying atlas style
        if self.apply_atlas_style:
            # Apply ATLAS style
            self.atlasify()
            # Mirror y axis to have the diagonal in the common orientation
            self.axis_top.invert_yaxis()

        # Disable x and y ticks for better appearance
        self.axis_top.tick_params("x", which="both", top=False, bottom=False)
        self.axis_top.tick_params("y", which="both", right=False, left=False)
        # Disable grid for better appearance
        self.axis_top.grid(False)

        # Setting title and label
        self.set_xlabel()
        self.set_ylabel(self.axis_top)
        self.set_title()

    def draw(self, matrix):
        """Draw a matrix with the class customized appearance.

        Parameters
        ----------
        matrix : np.ndarray
            The matrix to be plotted.
        """
        # Checking size consistency of ticklabels
        if self.x_ticklabels is not None:
            assert (
                len(self.x_ticklabels) == matrix.shape[1]
            ), "MatshowPlot: mismatch between x_tickslabels and number of columns in the matrix."

        if self.y_ticklabels is not None:
            assert (
                len(self.y_ticklabels) == matrix.shape[0]
            ), "MatshowPlot: mismatch between y_tickslabels and number of rows in the matrix."

        self.__plot(matrix)
