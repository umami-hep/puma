"""Matrix plotting with matplotlib matshow."""

from __future__ import annotations

import math as m

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
            rgbaColor <= 0.03928,
            rgbaColor / 12.92,
            ((rgbaColor + 0.055) / 1.055) ** 2.4,
        )
        weights = np.array([0.2126, 0.7152, 0.0722])
        return np.dot(rgbaColor, weights)

    def __plot(self, matrix):
        """Plot the Matrix."""
        n_cols = matrix.shape[1]
        n_rows = matrix.shape[0]

        # If using percentage values, make the colormap in range [0,1]
        if self.show_percentage:
            im = self.axis_top.matshow(matrix * 100, vmin=0, vmax=100, cmap=self.colormap)
        else:
            im = self.axis_top.matshow(matrix, cmap=self.colormap)

        # If mat entries have to be plotted
        if self.show_entries:
            # Mapping mat values in [0,1], as it's done by matplotlib
            # to associate them to the colors of the colormap
            normMat = matrix - np.min(matrix)
            # Casting to float64 in case the matrix is of integer type
            normMat = normMat.astype(np.float64)
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
                    if not self.show_percentage and m.modf(matrix[i, j])[0] == 0:
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

        # inverting y axis to have the diagonal in the common orientation
        self.axis_top.invert_yaxis()

        if self.show_cbar:
            # Plotting colorbar
            divider = make_axes_locatable(self.axis_top)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = self.fig.colorbar(im, cax=cax)
            # If using percentages, converting cbar labels to percentages
            if self.show_entries and self.show_percentage:
                cbar.set_ticks(
                    ticks=np.linspace(0, 100, 5),
                    labels=[f"{i}%" for i in np.round(np.linspace(0, 100, 5), 2)],
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
        positions = np.array(range(n_cols))
        self.axis_top.set_xticks(
            positions + 0.25,
            labels=self.x_ticklabels,
            rotation=self.x_ticks_rotation,
            fontsize=self.fontsize,
            ha="right",
        )
        positions = np.array(range(n_rows))
        self.axis_top.set_yticks(positions, labels=self.y_ticklabels, fontsize=self.fontsize)
        # Put xticks to the bottom
        self.axis_top.xaxis.tick_bottom()

        # Finished plotting, can apply atlas_style
        self.plotting_done = True
        # Applying atlas style
        if self.apply_atlas_style:
            # Apply ATLAS style
            self.atlasify()

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


class MatrixComparison(MatshowPlot):
    """Plot a comparison between two matrices. The resulting plot is a matrix with each bin split in
    two triangles, each containing the value of the corresponding matrix.
    """

    def __init__(self, show_legend: bool = True, matrix_names: list[str] | None = None, **kwargs):
        """Initialize the MatrixComparison plotter."""
        self.show_legend = show_legend
        self.matrix_names = matrix_names
        if matrix_names is None:
            self.matrix_names = ["m1", "m2"]
        super().__init__(**kwargs)

    def __plot(self, m1, m2):
        """Plot the two matrices."""
        n_rows, n_cols = m1.shape
        all_values = np.concatenate([m1.flatten(), m2.flatten()])

        # If using percentage values, make the colormap in range [0,1]
        if self.show_percentage:
            norm = plt.Normalize(vmin=0, vmax=1)
        else:
            norm = plt.Normalize(vmin=np.min(all_values), vmax=np.max(all_values))
        cmap = self.colormap

        # Set up divider for colorbar and legend
        divider = make_axes_locatable(self.axis_top)

        # Prepare optional axes: colorbar and legend axes
        cax = None
        legend_ax = None
        if self.show_cbar:
            cax = divider.append_axes("right", size="5%", pad=0.1)
        if self.show_legend:
            padding = 1.2 if self.cbar_label is not None else 0.6
            legend_ax = divider.append_axes("right", size="20%", pad=padding)
            legend_ax.set_box_aspect(1)

        # Matrix plotting: Plotting triangles for m1 and m2
        ax = self.axis_top
        for y in range(n_rows):
            for x in range(n_cols):
                top_left = (x, y + 1)
                top_right = (x + 1, y + 1)
                bottom_left = (x, y)
                bottom_right = (x + 1, y)

                # Upper triangle (for m1)
                tri1 = [top_right, top_left, bottom_right]
                color1 = cmap(norm(m1[y, x]))
                patch1 = patches.Polygon(tri1, facecolor=color1, edgecolor="gray")
                ax.add_patch(patch1)

                # Lower triangle (for m2)
                tri2 = [bottom_left, bottom_right, top_left]
                color2 = cmap(norm(m2[y, x]))
                patch2 = patches.Polygon(tri2, facecolor=color2, edgecolor="gray")
                ax.add_patch(patch2)

                if self.show_entries:
                    # Text for m1 (upper triangle)
                    luminance1 = self._MatshowPlot__get_luminance(color1)
                    text_color1 = "white" if luminance1 <= self.text_color_threshold else "black"
                    text1 = (
                        f"{m1[y, x]:.0f}"
                        if not self.show_percentage and m.modf(m1[y, x])[0] == 0
                        else (
                            f"{m1[y, x]:.3f}"
                            if not self.show_percentage
                            else f"{m1[y, x] * 100:.0f}%"
                        )
                    )
                    ax.text(
                        x + 0.75,
                        y + 0.75,
                        text1,
                        ha="center",
                        va="center",
                        fontsize=self.fontsize,
                        color=text_color1,
                    )

                    # Text for m2 (lower triangle)
                    luminance2 = self._MatshowPlot__get_luminance(color2)
                    text_color2 = "white" if luminance2 <= self.text_color_threshold else "black"
                    text2 = (
                        f"{m2[y, x]:.0f}"
                        if not self.show_percentage and m.modf(m2[y, x])[0] == 0
                        else (
                            f"{m2[y, x]:.3f}"
                            if not self.show_percentage
                            else f"{m2[y, x] * 100:.0f}%"
                        )
                    )
                    ax.text(
                        x + 0.25,
                        y + 0.25,
                        text2,
                        ha="center",
                        va="center",
                        fontsize=self.fontsize,
                        color=text_color2,
                    )

        # Configure axis settings (no ticks)
        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        ax.set_aspect("equal")

        # Add colorbar if enabled
        if self.show_cbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=cax)
            # If using percentages, convert cbar labels to percentages
            if self.show_entries and self.show_percentage:
                ticks = np.linspace(0, 1, 5)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{t * 100:.0f}%" for t in ticks])
                cbar.ax.tick_params(labelsize=self.fontsize)

            if self.cbar_label:
                cbar.ax.set_ylabel(self.cbar_label, fontsize=self.fontsize)

        # Add triangle legend if enabled
        if self.show_legend:
            legend_ax.axis("off")

            triangle1 = patches.Polygon(
                [[0, 0], [1, 0], [0, 1]],
                facecolor=cmap(norm(np.min(all_values))),
                edgecolor="gray",
            )
            triangle2 = patches.Polygon(
                [[1, 1], [1, 0], [0, 1]],
                facecolor=cmap(norm(np.min(all_values))),
                edgecolor="gray",
            )
            square = patches.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="black")

            legend_ax.add_patch(square)
            legend_ax.add_patch(triangle1)
            legend_ax.add_patch(triangle2)

            legend_ax.text(
                0.75,
                0.75,
                self.matrix_names[0],
                ha="center",
                va="center",
                fontsize=self.fontsize,
            )
            legend_ax.text(
                0.25,
                0.25,
                self.matrix_names[1],
                ha="center",
                va="center",
                fontsize=self.fontsize,
            )

        # Set tick labels
        if self.x_ticklabels is None:
            self.x_ticklabels = [str(i) for i in range(n_cols)]
        if self.y_ticklabels is None:
            self.y_ticklabels = [str(i) for i in range(n_rows)]

        x_pos = np.arange(n_cols) + 0.75
        y_pos = np.arange(n_rows) + 0.5
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            self.x_ticklabels,
            rotation=self.x_ticks_rotation,
            fontsize=self.fontsize,
            ha="right",
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.y_ticklabels, fontsize=self.fontsize)
        ax.xaxis.tick_bottom()

        self.set_xlabel()
        self.set_ylabel(ax)
        self.set_title()

        self.plotting_done = True
        if self.apply_atlas_style:
            self.atlasify()

        # disable ticks and grid which are enabled by atlasify
        self.axis_top.tick_params(axis="both", which="both", length=0)
        self.axis_top.grid(False)

    def draw(self, matrix1, matrix2):
        """Draw a comparison between two matrices with the class customized appearance.

        Parameters
        ----------
        matrix1 : np.ndarray
            The first matrix to be plotted.
        matrix2 : np.ndarray
            The second matrix to be plotted.
        """
        # Checking size consistency of the two matrices
        assert matrix1.shape == matrix2.shape, "MatrixComparison: matrices have different shapes."
        # Checking size consistency of ticklabels
        if self.x_ticklabels is not None:
            assert (
                len(self.x_ticklabels) == matrix1.shape[1]
            ), "MatshowPlot: mismatch between x_tickslabels and number of columns in the matrix."

        if self.y_ticklabels is not None:
            assert (
                len(self.y_ticklabels) == matrix1.shape[0]
            ), "MatshowPlot: mismatch between y_tickslabels and number of rows in the matrix."

        self.__plot(matrix1, matrix2)
