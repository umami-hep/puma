from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from puma.plot_base import PlotBase
from puma.utils import logger


class MatshowPlot(PlotBase):
    """Plot Matrix class."""

    def __init__(
        self,
        matrix: np.ndarray,
        x_ticklabels: list | None = None,
        x_ticks_rotation: int = 90,
        y_ticklabels: list | None = None,
        show_entries: bool = True,
        show_percentage: bool = False,
        text_color_threshold: float = 0.408,
        colormap: plt.cm = plt.cm.Oranges,
        cbar_label: str | None = None,
        atlas_offset: float = 1,
        **kwargs,
    ) -> None:
        """Plot a matrix with matplotlib matshow.

        Parameters
        ----------
        matrix : np.ndarray
            The matrix to be plotted.
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
        cbar_label : str | None, optional
            Label of the colorbar, by default None
        atlas_offset : float, optional
            Space at the top of the plot reserved to the Atlasify text. by default 1
        **kwargs : kwargs
            Keyword arguments for `puma.PlotObject`

        Example
        -------
        >>> mat = np.random.rand(4, 3)
        >>> plot_mat = MatshowPlot(mat)
        """
        super().__init__(**kwargs)

        # Checking size consistency of ticklabels
        if x_ticklabels is not None:
            assert (
                len(x_ticklabels) == matrix.shape[1]
            ), "MatshowPlot: mismatch between x_tickslabels and number of columns in the matrix."

        if y_ticklabels is not None:
            assert (
                len(y_ticklabels) == matrix.shape[0]
            ), "MatshowPlot: mismatch between y_tickslabels and number of rows in the matrix."

        self.mat = matrix
        self.x_ticklabels = x_ticklabels
        self.x_ticks_rotation = x_ticks_rotation
        self.y_ticklabels = y_ticklabels
        self.show_entries = show_entries
        self.show_percentage = show_percentage
        self.text_color_threshold = text_color_threshold
        self.colormap = colormap
        self.cbar_label = cbar_label
        self.atlas_offset = atlas_offset

        # Specifying figsize if not specified by user
        if self.figsize is None:
            self.figsize = (10, 10.5)
        self.initialise_figure()
        self.__plot()

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

    def __plot(self):
        """Plot the Matrix."""
        im = self.axis_top.matshow(self.mat, cmap=self.colormap)

        # If mat entries have to be plotted
        if self.show_entries:
            # Mapping mat values in [0,1], as it's done by matplotlib
            # to associate them to the colors of the colormap
            normMat = self.mat - np.min(self.mat)
            normMat = normMat / (np.max(self.mat) - np.min(self.mat))

            # Adding text values in the matrix pixels
            for i in range(self.mat.shape[0]):
                for j in range(self.mat.shape[1]):
                    # Choosing the text color: black if color is light, white if color is dark
                    # Getting the bkg color from the cmap
                    color = self.colormap(normMat[i, j])
                    # Calculating the bkg relative luminance
                    luminance = self.__get_luminance(color)
                    # Choosing the appropriate color
                    color = "white" if luminance <= self.text_color_threshold else "black"

                    # Value of the matrix, eventually converted to percentage
                    text = (
                        str(round(self.mat[i, j], 3))
                        if not self.show_percentage
                        else str(round(self.mat[i, j] * 100, 2)) + "%"
                    )
                    # Plotting text
                    self.axis_top.text(
                        x=j, y=i, s=text, va="center", ha="center", size="x-large", c=color
                    )

        # Plotting colorbar
        cbar = self.fig.colorbar(im)
        # If using percentages, converting cbar labels to percentages
        if self.show_entries and self.show_percentage:
            minMat = np.min(self.mat)
            maxMat = np.max(self.mat)
            cbar.set_ticks(
                ticks=np.linspace(minMat, maxMat, 5),
                labels=[str(i) + "%" for i in np.round(np.linspace(minMat, maxMat, 5) * 100, 2)],
            )
        if self.cbar_label is not None:
            cbar.ax.set_ylabel(self.cbar_label)

        # Setting tick labels
        if self.x_ticklabels is None:
            self.x_ticklabels = [str(i) for i in range(self.mat.shape[1])]
            logger.info("MatshowPlot: no x_ticklabels given, using indices instead.")
        if self.y_ticklabels is None:
            self.y_ticklabels = [str(i) for i in range(self.mat.shape[0])]
            logger.info("MatshowPlot: no y_ticklabels given, using indices instead.")

        # Writing class names on the axes
        positions = list(range(self.mat.shape[1]))
        self.axis_top.set_xticks(
            positions, labels=self.x_ticklabels, rotation=self.x_ticks_rotation
        )
        positions = list(range(self.mat.shape[0]))
        self.axis_top.set_yticks(positions, labels=self.y_ticklabels)
        # Put xticks to the bottom
        self.axis_top.xaxis.tick_bottom()

        # Finished plotting, can apply atlas_style
        self.plotting_done = True
        # Applying atlas style
        if self.apply_atlas_style:
            # Allow some space for the ATLAS legend
            self.axis_top.set_ylim(-self.atlas_offset, self.mat.shape[0] - 0.5)
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
