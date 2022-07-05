"""Plotting bases for specialised plotting."""

from dataclasses import dataclass

import atlasify
from matplotlib import axis, gridspec
from matplotlib.figure import Figure

from puma.utils import logger, set_xaxis_ticklabels_invisible

atlasify.LINE_SPACING = 1.3  # overwrite the default, which is 1.2


# TODO: enable `kw_only` when switching to Python 3.10
# @dataclass(kw_only=True)
@dataclass
class PlotLineObject:  # pylint: disable=too-many-instance-attributes
    """Base data class defining properties of a plot object.

    Parameters
    ----------
    xmin : float, optional
        Minimum value of the x-axis, by default None
    xmax : float, optional
        Maximum value of the x-axis, by default None
    colour : str, optional
        Colour of the object, by default None
    label : str, optional
        Label of object, by default None
    linestyle : str, optional
        Linestyle following numpy style, by default None
    alpha : float, optional
        Value for visibility of the plot lines, by default None
    marker : str, optional
        Marker that is used in the plot. For example an x.
        By default None
    markersize : int, optional
        Size of the marker. By default None
    markeredgewidth : int, optional
        Edge width of the marker. By default None
    is_marker : bool, optional
        Bool, to give info about if this is a marker or a line.
        By default None
    """

    xmin: float = None
    xmax: float = None
    colour: str = None
    label: str = None
    linestyle: str = None
    linewidth: str = None
    alpha: float = None
    marker: str = None
    markersize: int = None
    markeredgewidth: int = None
    is_marker: bool = None


# TODO: enable `kw_only` when switching to Python 3.10
# @dataclass(kw_only=True)
@dataclass
class PlotObject:  # pylint: disable=too-many-instance-attributes
    """Data base class defining properties of a plot object.

    Parameters
    ----------
    title : str, optional
        Title of the plot, by default ""
    draw_errors : bool, optional
        Draw statistical uncertainty on the lines, by default True
    xmin : float, optional
        Minimum value of the x-axis, by default None
    xmax : float, optional
        Maximum value of the x-axis, by default None
    ymin : float, optional
        Minimum value of the y-axis, by default None
    ymax : float, optional
        Maximum value of the y-axis, by default None
    ymin_ratio_1 : float, optional
        Set the lower y limit of the first ratio subplot, by default None.
    ymax_ratio_1 : float, optional
        Set the upper y limit of the first ratio subplot, by default None.
    ymin_ratio_2 : float, optional
        Set the lower y limit of the second ratio subplot, by default None.
    ymax_ratio_2 : float, optional
        Set the upper y limit of the second ratio subplot, by default None.
    y_scale : float, optional
        Scaling up the y axis, e.g. to fit the ATLAS Tag. Applied if ymax not defined,
        by default 1.3
    logx: bool, optional
        Set the log of x-axis, by default False
    logy : bool, optional
        Set log of y-axis of main panel, by default True
    xlabel : str, optional
        Label of the x-axis, by default None
    ylabel : str, optional
        Label of the y-axis, by default None
    ylabel_ratio_1 : str, optional
        Label of the y-axis in the first ratio plot, by default "Ratio"
    ylabel_ratio_2 : str, optional
        Label of the y-axis in the second ratio plot, by default "Ratio"
    label_fontsize : int, optional
        Used fontsize in label, by default 12
    fontsize : int, optional
        Used fontsize, by default 10
    n_ratio_panels : int, optional
        Amount of ratio panels between 0 and 2, by default 0
    vertical_split: bool
        Set to False if you would like to split the figure horizonally. If set to
        True the figure is split vertically (e.g for pie chart). By default False.
    figsize : (float, float), optional
        Tuple of figure size `(width, height)` in inches, by default (8, 6)
    dpi : int, optional
        dpi used for plotting, by default 400
    grid : bool, optional
        Set the grid for the plots.
    leg_fontsize : int, optional
        Fontsize of the legend, by default 10
    leg_loc : str, optional
        Position of the legend in the plot, by default "upper right"
    leg_ncol : int, optional
        Number of legend columns, by default 1
    apply_atlas_style : bool, optional
        Apply ATLAS style for matplotlib, by default True
    use_atlas_tag : bool, optional
        Use the ATLAS Tag in the plots, by default True
    atlas_first_tag : str, optional
        First row of the ATLAS tag (i.e. the first row is "ATLAS <atlas_first_tag>"),
        by default "Simulation Internal"
    atlas_second_tag : str, optional
        Second row of the ATLAS tag, by default ""
    atlas_fontsize : float, optional
        Fontsize of ATLAS label, by default 10
    atlas_vertical_offset : float, optional
        Vertical offset of the ATLAS tag, by default 7
    atlas_horizontal_offset : float, optional
        Horizontal offset of the ATLAS tag, by default 8
    atlas_brand : str, optional
        `brand` argument handed to atlasify. If you want to remove it just use an empty
        string or None, by default "ATLAS"
    atlas_tag_outside : bool, optional
        `outside` argument handed to atlasify. Decides if the ATLAS logo is plotted
        outside of the plot (on top), by default False
    atlas_second_tag_distance : float, optional
        Distance between the `atlas_first_tag` and `atlas_second_tag` text in units
        of line spacing, by default 0
    plotting_done : bool
        Bool that indicates if plotting is done. Only then `atlasify()` can be called,
        by default False
    """

    title: str = ""
    draw_errors: bool = True

    xmin: float = None
    xmax: float = None
    ymin: float = None
    ymax: float = None
    ymin_ratio_1: float = None
    ymax_ratio_1: float = None
    ymin_ratio_2: float = None
    ymax_ratio_2: float = None
    y_scale: float = 1.3
    logx: bool = False
    logy: bool = True
    xlabel: str = None
    ylabel: str = None
    ylabel_ratio_1: str = "Ratio"
    ylabel_ratio_2: str = "Ratio"
    label_fontsize: int = 12
    fontsize: int = 10

    n_ratio_panels: int = 0
    vertical_split: bool = False

    figsize: tuple = None
    dpi: int = 400

    grid: bool = True

    # legend settings
    leg_fontsize: int = None
    leg_loc: str = "upper right"
    leg_ncol: int = 1

    # defining ATLAS style and tags
    apply_atlas_style: bool = True
    use_atlas_tag: bool = True
    atlas_first_tag: str = "Simulation Internal"
    atlas_second_tag: str = None
    atlas_fontsize: int = None
    atlas_vertical_offset: float = 7
    atlas_horizontal_offset: float = 8
    atlas_brand: str = "ATLAS"
    atlas_tag_outside: bool = False
    atlas_second_tag_distance: float = 0

    plotting_done: bool = False

    def __post_init__(self):
        """Check for allowed values.

        Raises
        ------
        ValueError
            If n_ratio_panels not in [0, 1, 2]
        """
        self.__check_figsize()
        allowed_n_ratio_panels = [0, 1, 2]
        if self.n_ratio_panels not in allowed_n_ratio_panels:
            raise ValueError(
                f"{self.n_ratio_panels} not allwed value for `n_ratio_panels`. "
                f"Allowed are {allowed_n_ratio_panels}"
            )
        if self.leg_fontsize is None:
            self.leg_fontsize = self.fontsize
        if self.atlas_fontsize is None:
            self.atlas_fontsize = self.fontsize
        if self.apply_atlas_style is False and (
            self.atlas_first_tag is not None or self.atlas_second_tag is not None
        ):
            logger.warning(
                "You specified an ATLAS tag, but `apply_atlas_style` is set to false. "
                "Tag will therefore not be shown on plot."
            )

    def __check_figsize(self):
        """Check `figsize`

        Raises
        ------
        ValueError
            If shape of `figsize` is not a tuple or list with length 2
        """
        if self.figsize is None:
            return
        if isinstance(self.figsize, list) and len(self.figsize) == 2:
            self.figsize = tuple(self.figsize)
        elif not isinstance(self.figsize, tuple) or len(self.figsize) != 2:
            raise ValueError(
                f"You passed `figsize` as {self.figsize} which is not allowed. "
                "Either a tuple or a list of size 2 is allowed"
            )


class PlotBase(PlotObject):  # pylint: disable=too-many-instance-attributes
    """Base class for plotting"""

    def __init__(self, **kwargs) -> None:
        """Initialise class

        Parameters
        ----------
        **kwargs : kwargs
            Keyword arguments from `puma.PlotObject`
        """
        super().__init__(**kwargs)
        self.axis_top = None
        self.axis_ratio_1 = None
        self.axis_ratio_2 = None
        self.axis_leg = None
        self.fig = None

    def initialise_figure(self, sub_plot_index: int = 5):
        """
        Initialising matplotlib.figure.Figure for different scenarios depending on how
        many ratio panels are requested.

        Parameters
        ----------
        sub_plot_index : int, optional
            Indicates for the scenario with one ratio how large the upper and lower
            panels are, by default 5
        """
        # TODO: switch to cases syntax in python 3.10

        if self.vertical_split:
            # split figure vertically instead of horizonally
            if self.n_ratio_panels >= 1:
                logger.warning(
                    "You set the number of ratio panels to %i "
                    "but also set the vertical splitting to True. Therefore no ratio"
                    "panels are created.",
                    self.n_ratio_panels,
                )
            self.fig = Figure(
                figsize=(6, 4.5) if self.figsize is None else self.figsize
            )
            g_spec = gridspec.GridSpec(1, 11, figure=self.fig)
            self.axis_top = self.fig.add_subplot(g_spec[0, :9])
            self.axis_leg = self.fig.add_subplot(g_spec[0, 9:])

        else:
            if self.n_ratio_panels == 0:
                # no ratio panel
                self.fig = Figure(
                    figsize=(5, 3.5) if self.figsize is None else self.figsize
                )
                self.axis_top = self.fig.gca()

            elif self.n_ratio_panels == 1:
                # 1 ratio panel
                self.fig = Figure(
                    figsize=(5, 4) if self.figsize is None else self.figsize
                )

                g_spec = gridspec.GridSpec(8, 1, figure=self.fig)
                self.axis_top = self.fig.add_subplot(g_spec[:sub_plot_index, 0])
                self.axis_ratio_1 = self.fig.add_subplot(
                    g_spec[sub_plot_index:, 0], sharex=self.axis_top
                )

            elif self.n_ratio_panels == 2:
                # 2 ratio panels
                self.fig = Figure(
                    figsize=(8, 8) if self.figsize is None else self.figsize
                )

                # Define the grid of the subplots
                g_spec = gridspec.GridSpec(11, 1, figure=self.fig)
                self.axis_top = self.fig.add_subplot(g_spec[:5, 0])
                self.axis_ratio_1 = self.fig.add_subplot(
                    g_spec[5:8, 0], sharex=self.axis_top
                )
                self.axis_ratio_2 = self.fig.add_subplot(
                    g_spec[8:, 0], sharex=self.axis_top
                )

            if self.n_ratio_panels >= 1:
                set_xaxis_ticklabels_invisible(self.axis_top)
            if self.n_ratio_panels >= 2:
                set_xaxis_ticklabels_invisible(self.axis_ratio_1)

    def draw_vlines(
        self,
        vlines_xvalues: list,
        vlines_label_list: list = None,
        vlines_line_height_list: list = None,
        same_height: bool = False,
        colour: str = "#000000",
        fontsize: int = 10,
    ):  # pylint: disable=too-many-arguments
        """Drawing working points in plot

        Parameters
        ----------
        vlines_xvalues : list
            List of working points x values to draw
        vlines_label_list : list, optional
            List with labels for the vertical lines. Must be the same
            order as the vlines_xvalues. If None, the xvalues * 100 will be
            used as labels. By default None
        vlines_line_height_list : list, optional
            List with the y height of the vertical lines in percent of the
            upper plot (0 is bottom, 1 is top). Must be the same
            order as the vlines_xvalues and the labels. By default None
        same_height : bool, optional
            Working point lines on same height, by default False
        colour : str, optional
            Colour of the vertical line, by default "#000000" (black)
        fontsize : int, optional
            Fontsize of the vertical line text. By default 10.
        """
        for vline_counter, vline in enumerate(vlines_xvalues):
            # Set y-point of the WP lines/text
            if vlines_line_height_list is None:
                ytext = 0.65 if same_height else 0.65 - vline_counter * 0.07

            else:
                ytext = vlines_line_height_list[vline_counter]

            self.axis_top.axvline(
                x=vline,
                ymax=ytext,
                color=colour,
                linestyle="dashed",
                linewidth=1.0,
            )

            # Set the number above the line
            self.axis_top.text(
                x=vline - 0.005,
                y=ytext + 0.005,
                s=f"{int(vline * 100)}%"
                if vlines_label_list is None
                else f"{vlines_label_list[vline_counter]}",
                transform=self.axis_top.get_xaxis_text1_transform(0)[0],
                fontsize=fontsize,
            )

            if self.n_ratio_panels > 0:
                self.axis_ratio_1.axvline(
                    x=vline, color=colour, linestyle="dashed", linewidth=1.0
                )
            if self.n_ratio_panels == 2:
                self.axis_ratio_2.axvline(
                    x=vline, color=colour, linestyle="dashed", linewidth=1.0
                )

    def set_title(self, title: str = None, **kwargs):
        """Set title of top panel.

        Parameters
        ----------
        title : str, optional
            Title of top panel, if None using the value form the class variables,
            by default None
        **kwargs : kwargs
            Keyword arguments passed to `matplotlib.axes.Axes.set_title()`
        """
        self.axis_top.set_title(self.title if title is None else title, **kwargs)

    def set_log(self, force_x: bool = False, force_y: bool = False):
        """Set log scale of the axes. For the y-axis, only the main panel is
        set. For the x-axes (also from the ratio subpanels), all are changed.

        Parameters
        ----------
        force_x : bool, optional
            Forcing log on x-axis even if `logx` attribute is False, by default False
        force_y : bool, optional
            Forcing log on y-axis even if `logy` attribute is False, by default False
        """

        if self.logx or force_x:
            if not self.logx:
                logger.warning(
                    "Setting log of x-axis but `logx` flag was set to False."
                )

            # Set log scale for all plots
            self.axis_top.set_xscale("log")
            if self.axis_ratio_1:
                self.axis_ratio_1.set_xscale("log")
            if self.axis_ratio_2:
                self.axis_ratio_2.set_xscale("log")

        if self.logy or force_y:
            if not self.logy:
                logger.warning(
                    "Setting log of y-axis but `logy` flag was set to False."
                )

            self.axis_top.set_yscale("log")
            ymin, ymax = self.axis_top.get_ylim()
            self.y_scale = ymin * ((ymax / ymin) ** self.y_scale) / ymax

    def set_y_lim(self):
        """Set limits of y-axis."""
        ymin, ymax = self.axis_top.get_ylim()
        self.axis_top.set_ylim(
            ymin if self.ymin is None else self.ymin,
            ymax * self.y_scale if self.ymax is None else self.ymax,
        )

        if self.axis_ratio_1:
            if self.ymin_ratio_1 or self.ymax_ratio_1:
                ymin, ymax = self.axis_ratio_1.get_ylim()

                if self.ymin_ratio_1:
                    ymin = self.ymin_ratio_1

                if self.ymax_ratio_1:
                    ymax = self.ymax_ratio_1

                self.axis_ratio_1.set_ylim(bottom=ymin, top=ymax)

        if self.axis_ratio_2:
            if self.ymin_ratio_2 or self.ymax_ratio_2:
                ymin, ymax = self.axis_ratio_2.get_ylim()

                if self.ymin_ratio_2:
                    ymin = self.ymin_ratio_2

                if self.ymax_ratio_2:
                    ymax = self.ymax_ratio_2

                self.axis_ratio_2.set_ylim(bottom=ymin, top=ymax)

    def set_ylabel(self, ax_mpl, label: str = None, align_right: bool = True, **kwargs):
        """Set y-axis label.

        Parameters
        ----------
        ax_mpl : matplotlib.axes.Axes
            matplotlib axis object
        label : str, optional
            x-axis label, by default None
        align_right : bool, optional
            Alignment of y-axis label, by default True
        **kwargs, kwargs
            Keyword arguments passed to `matplotlib.axes.Axes.set_ylabel()`
        """
        label_options = {}
        if align_right:
            label_options = {
                "fontsize": self.label_fontsize,
                "horizontalalignment": "right",
                "y": 1.0,
            }
        else:
            label_options = {
                "fontsize": self.label_fontsize,
            }

        ax_mpl.set_ylabel(
            self.ylabel if label is None else label,
            **label_options,
            **kwargs,
        )

    def set_xlabel(self, label: str = None, **kwargs):
        """Set x-axis label.

        Parameters
        ----------
        label : str, optional
            x-axis label, by default None
        **kwargs : kwargs
            Keyword arguments passed to `matplotlib.axes.Axes.set_xlabel()`
        """
        xlabel_args = {
            "xlabel": self.xlabel if label is None else label,
            "horizontalalignment": "right",
            "x": 1.0,
            "fontsize": self.label_fontsize,
        }
        # TODO: switch to cases syntax in python 3.10
        if self.n_ratio_panels == 0:
            self.axis_top.set_xlabel(**xlabel_args, **kwargs)
        elif self.n_ratio_panels == 1:
            self.axis_ratio_1.set_xlabel(**xlabel_args, **kwargs)
        elif self.n_ratio_panels == 2:
            self.axis_ratio_2.set_xlabel(**xlabel_args, **kwargs)

    def set_tick_params(self, labelsize: int = None, **kwargs):
        """Set x-axis label.

        Parameters
        ----------
        labelsize : int, optional
            Label size of x- and y- axis ticks, by default None.
            If None then using global fontsize
        **kwargs : kwargs
            Keyword arguments passed to `matplotlib.axes.Axes.set_xlabel()`
        """
        labelsize = self.fontsize if labelsize is None else labelsize
        self.axis_top.tick_params(axis="y", labelsize=labelsize, **kwargs)
        # TODO: switch to cases syntax in python 3.10
        if self.n_ratio_panels == 0:
            self.axis_top.tick_params(axis="x", labelsize=labelsize, **kwargs)
        elif self.n_ratio_panels == 1:
            self.axis_ratio_1.tick_params(axis="y", labelsize=labelsize, **kwargs)
            self.axis_ratio_1.tick_params(axis="x", labelsize=labelsize, **kwargs)
        elif self.n_ratio_panels == 2:
            self.axis_ratio_1.tick_params(axis="y", labelsize=labelsize, **kwargs)
            self.axis_ratio_2.tick_params(axis="y", labelsize=labelsize, **kwargs)
            self.axis_ratio_2.tick_params(axis="x", labelsize=labelsize, **kwargs)

    def set_xlim(self, xmin: float = None, xmax: float = None, **kwargs):
        """Set limits of x-axis

        Parameters
        ----------
        xmin : float, optional
            Min of x-axis, by default None
        xmax : float, optional
            Max of x-axis, by default None
        **kwargs : kwargs
            Keyword arguments passed to `matplotlib.axes.Axes.set_xlim()`
        """
        self.axis_top.set_xlim(
            self.xmin if xmin is None else xmin,
            self.xmax if xmax is None else xmax,
            **kwargs,
        )

    def savefig(
        self,
        plot_name: str,
        transparent: bool = False,
        dpi: int = None,
        **kwargs,
    ):
        """Save plot to disk.

        Parameters
        ----------
        plot_name : str
            File name of the plot
        transparent : bool, optional
            If plot transparent, by default False
        dpi : int, optional
            DPI for plotting, by default 400
        **kwargs : kwargs
            Keyword arguments passed to `matplotlib.figure.Figure.savefig()`
        """
        logger.debug("Saving plot to %s", plot_name)
        self.fig.savefig(
            plot_name,
            transparent=transparent,
            dpi=self.dpi if dpi is None else dpi,
            **kwargs,
        )

    def tight_layout(self, **kwargs):
        """abstract function of matplotlib.figure.Figure.tight_layout

        Parameters
        ----------
        **kwargs: kwargs
            Keyword arguments from `matplotlib.figure.Figure.tight_layout()`
        """
        self.fig.tight_layout(**kwargs)

    def atlasify(self, use_tag: bool = True, force: bool = False):
        """Apply ATLAS style to all axes using the atlasify package

        Parameters
        ----------
        use_tag : bool, optional
            If False, ATLAS style will be applied but no tag will be put on the plot.
            If True, the tag will be put on as well, by default True
        force : bool, optional
            Force ATLAS style also if class variable is False, by default False
        """

        if self.plotting_done is False and force is False:
            logger.warning(
                "`atlasify()` has to be called after plotting --> "
                "ATLAS style will not be adapted. If you want to do it anyway, "
                "you can use `force`."
            )
            return

        if self.apply_atlas_style or force:
            logger.debug("Initialise ATLAS style using atlasify.")
            if use_tag is True:
                # TODO: for some reason, pylint complains about the used arguments
                # when calling atlasify ("unexpected-keyword-arg") error
                # --> fix this
                atlasify.atlasify(  # pylint: disable=E1123
                    atlas=self.atlas_first_tag,
                    subtext=self.atlas_second_tag,
                    axes=self.axis_top,
                    font_size=self.atlas_fontsize,
                    label_font_size=self.atlas_fontsize,
                    sub_font_size=self.atlas_fontsize,
                    offset=self.atlas_vertical_offset,
                    indent=self.atlas_horizontal_offset,
                    enlarge=1,
                    brand="" if self.atlas_brand is None else self.atlas_brand,
                    outside=self.atlas_tag_outside,
                    subtext_distance=self.atlas_second_tag_distance,
                )
            else:
                atlasify.atlasify(atlas=False, axes=self.axis_top, enlarge=1)
            if self.n_ratio_panels >= 1:
                atlasify.atlasify(atlas=False, axes=self.axis_ratio_1, enlarge=1)
            if self.n_ratio_panels == 2:
                atlasify.atlasify(atlas=False, axes=self.axis_ratio_2, enlarge=1)
            if self.vertical_split:
                atlasify.atlasify(atlas=False, axes=self.axis_leg, enlarge=1)
            if force:
                if self.apply_atlas_style is False:
                    logger.warning(
                        "Initialising ATLAS style even though `apply_atlas_style` is  "
                        "set to False."
                    )
                if self.plotting_done is False:
                    logger.warning(
                        "Initialising ATLAS style even though `plotting_done` is set to"
                        " False."
                    )

    def make_legend(self, handles: list, ax_mpl: axis, labels: list = None, **kwargs):
        """Drawing legend on axis.

        Parameters
        ----------
        handles :  list
            List of matplotlib.lines.Line2D object returned when plotting
        ax_mpl : matplotlib.axis.Axes
            `matplotlib.axis.Axes` object where the legend should be plotted
        labels : list, optional
            Plot labels. If None, the labels are extracted from the `handles`.
            By default None
        **kwargs : kwargs
            Keyword arguments which can be passed to matplotlib axis
        """
        ax_mpl.legend(
            handles=handles,
            labels=[handle.get_label() for handle in handles]
            if labels is None
            else labels,
            loc=self.leg_loc,
            fontsize=self.leg_fontsize,
            ncol=self.leg_ncol,
            **kwargs,
        )

    def set_ratio_label(self, ratio_panel: int, label: str):
        """Associate the rejection class to a ratio panel

        Parameters
        ----------
        ratio_panel : int
            Indicates which ratio panel to modify (either 1 or 2).
        label : str
            y-axis label of the ratio panel

        Raises
        ------
        ValueError
            If requested ratio panels and given ratio_panel do not match.
        """
        # TODO: could add possibility to specify ratio label as function of rej_class
        if self.n_ratio_panels < ratio_panel and ratio_panel not in [1, 2]:
            raise ValueError(
                "Requested ratio panels and given ratio_panel do not match."
            )
        if ratio_panel == 1:
            self.ylabel_ratio_1 = label
        if ratio_panel == 2:
            self.ylabel_ratio_2 = label

    def initialise_plot(self):
        """Calls other methods which are usually used when plotting"""
        self.set_title()
        self.set_log()
        self.set_y_lim()
        self.set_xlabel()
        self.set_ylabel(self.axis_top)
        self.set_tick_params()
        if not self.atlas_tag_outside:
            self.fig.tight_layout()
        self.plotting_done = True
        if self.apply_atlas_style:
            self.atlasify()
