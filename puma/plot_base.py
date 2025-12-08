"""Plotting bases for specialised plotting."""

from __future__ import annotations

import json
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import atlasify
import numpy as np
import yaml
from ftag.cuts import Cut, Cuts
from ftag.labels import Label
from IPython import get_ipython
from IPython.display import display
from matplotlib import gridspec, lines
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from typing_extensions import Self

from puma.utils import logger, set_xaxis_ticklabels_invisible

atlasify.LINE_SPACING = 1.3  # overwrite the default, which is 1.2

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes


@dataclass
class PlotLineObject:
    """Base data class defining properties of a plot object.

    Attributes
    ----------
    xmin : float | None, optional
        Minimum value of the x-axis, by default None
    xmax : float | None, optional
        Maximum value of the x-axis, by default None
    colour : str | None, optional
        Colour of the object, by default None
    label : str | None, optional
        Label of object, by default None
    linestyle : str | None, optional
        Linestyle following numpy style, by default None
    linewidth : float | None, optional
        Linewidth that will be used, by default None
    alpha : float | None, optional
        Value for visibility of the plot lines, by default None
    marker : str | None, optional
        Marker that is used in the plot. For example an x.
        By default None
    markersize : int | None, optional
        Size of the marker. By default None
    markeredgewidth : int | None, optional
        Edge width of the marker. By default None
    is_marker : bool | None, optional
        Bool, to give info about if this is a marker or a line.
        By default None
    """

    xmin: float | None = None
    xmax: float | None = None
    colour: str | None = None
    label: str | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    alpha: float | None = None
    marker: str | None = None
    markersize: int | None = None
    markeredgewidth: int | None = None
    is_marker: bool | None = None

    @property
    def args_to_store(self) -> dict[str, Any]:
        """Returns the arguments that need to be stored/loaded.

        Returns
        -------
        dict[str, Any]
            Dict with the arguments
        """
        # Create the dict with the args to store/load
        return {
            "xmin": self.xmin,
            "xmax": self.xmax,
            "colour": self.colour,
            "label": self.label,
            "linestyle": self.linestyle,
            "linewidth": self.linewidth,
            "alpha": self.alpha,
            "marker": self.marker,
            "markersize": self.markersize,
            "markeredgewidth": self.markeredgewidth,
            "is_marker": self.is_marker,
        }

    @staticmethod
    def encode(obj: Any) -> Any:
        """Return a JSON/YAML-safe version of obj, tagging special types.

        Parameters
        ----------
        obj : Any
            Object that is to be encoded

        Returns
        -------
        Any
            The encoded object
        """
        # Setup the object so that it can be returned if no encoding is needed
        encoded: Any = obj

        # Encode special cases which can't be easily stored in json and yaml
        if isinstance(obj, np.ndarray):
            encoded = {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype)}
        elif isinstance(obj, Label):
            encoded = {"__label__": {k: PlotLineObject.encode(v) for k, v in obj.__dict__.items()}}
        elif isinstance(obj, tuple):
            encoded = {"__tuple__": [PlotLineObject.encode(v) for v in obj]}
        elif isinstance(obj, Cuts):
            encoded = {"__cuts__": {k: PlotLineObject.encode(v) for k, v in obj.__dict__.items()}}
        elif isinstance(obj, Cut):
            encoded = {"__cut__": {k: PlotLineObject.encode(v) for k, v in obj.__dict__.items()}}

        # For lists and dicts, walk through them and ensure correct encoding for sub-objects
        elif isinstance(obj, list):
            encoded = [PlotLineObject.encode(v) for v in obj]
        elif isinstance(obj, dict):
            encoded = {k: PlotLineObject.encode(v) for k, v in obj.items()}

        # If no encoding is needed, return the object
        return encoded

    @staticmethod
    def decode(obj: Any) -> Any:
        """Inverse of encode, turning tags back into real objects.

        Parameters
        ----------
        obj : Any
            Object that is to be decoded

        Returns
        -------
        Any
            The decoded object
        """
        # Setup the object so that it can be returned if no decoding is needed
        decoded: Any = obj

        # If a dict was used, go through and check for types
        if isinstance(obj, dict):
            if "__ndarray__" in obj:
                decoded = np.asarray(obj["__ndarray__"], dtype=obj["dtype"])
            elif "__label__" in obj:
                decoded = Label(**{
                    k: PlotLineObject.decode(v) for k, v in obj["__label__"].items()
                })
            elif "__tuple__" in obj:
                decoded = tuple(PlotLineObject.decode(v) for v in obj["__tuple__"])
            elif "__cuts__" in obj:
                decoded = Cuts(**{k: PlotLineObject.decode(v) for k, v in obj["__cuts__"].items()})
            elif "__cut__" in obj:
                decoded = Cut(**{k: PlotLineObject.decode(v) for k, v in obj["__cut__"].items()})
            else:
                # If it's a regular dict, walk down the keys
                decoded = {k: PlotLineObject.decode(v) for k, v in obj.items()}

        # If a list was used, check that all sub-objects are correctly loaded
        elif isinstance(obj, list):
            decoded = [PlotLineObject.decode(v) for v in obj]

        # If no decoding is needed, return the object
        return decoded

    def save(self, path: str | Path) -> None:
        """Store class attributes in a file (json or yaml).

        Saving can be performed to a yaml and a json file.

        Parameters
        ----------
        path : str | Path
            Path to which the class object attributes are written.

        Raises
        ------
        ValueError
            If an unknown file extension was given
        """
        # Ensure path is a path object
        path = Path(path)

        # Get the attributes as a dict
        data = self.encode(self.args_to_store)

        # Check for json and store it as such
        if path.suffix == ".json":
            with path.open("w") as f:
                json.dump(data, f, indent=2)

        # Check for yaml and store it as such
        elif path.suffix in {".yaml", ".yml"}:
            with path.open("w") as f:
                yaml.safe_dump(data, f)

        # Else ValueError
        else:
            raise ValueError("Unknown file extension. Use '.json', '.yaml' or '.yml'!")

    @classmethod
    def load(cls, path: str | Path, **extra_kwargs: Any) -> Self:
        """Load attributes from file and construct the object without __init__.

        Parameters
        ----------
        path : str | Path
            Path in which the attributes are stored.
        **extra_kwargs : Any
            Extra kwargs to overwrite certain stored options.

        Returns
        -------
        Class Instance
            Instance of class with the given attributes.

        Raises
        ------
        ValueError
            If the given file is neither json nor a yaml file.
        """
        # Ensure path is a path object
        path = Path(path)

        # Check if json and load it as such
        if path.suffix == ".json":
            with path.open() as f:
                data = json.load(f)

        # Check if yaml and load it as such
        elif path.suffix in {".yaml", ".yml"}:
            with path.open() as f:
                data = yaml.safe_load(f)

        # Else ValueError
        else:
            raise ValueError("Unknown file extension. Use '.json', '.yaml' or '.yml'.")

        # Convert back to numpy where appropriate
        data = cls.decode(data)

        # allow caller to override
        data.update(extra_kwargs)

        # Init the class without running __init__
        obj: Self = cls.__new__(cls)

        # Set attributes verbatim
        for key, val in data.items():
            setattr(obj, key, val)
        return obj


@dataclass
class PlotObject:
    """Data base class defining properties of a plot object.

    Attributes
    ----------
    title : str, optional
        Title of the plot, by default ""
    draw_errors : bool, optional
        Draw statistical uncertainty on the lines, by default True
    xmin : float | None, optional
        Minimum value of the x-axis, by default None
    xmax : float | None, optional
        Maximum value of the x-axis, by default None
    ymin : float | None, optional
        Minimum value of the y-axis, by default None
    ymax : float | None, optional
        Maximum value of the y-axis, by default None
    ymin_ratio : list[float | None] | None, optional
        Set the lower y limit of each of the ratio subplots, by default None.
    ymax_ratio : list[float | None] | None, optional
        Set the upper y limit of each of the ratio subplots, by default None.
    y_scale : float, optional
        Scaling up the y axis, e.g. to fit the ATLAS Tag. Applied if ymax not
        defined, by default 1.3
    logx : bool, optional
        Set the log of x-axis, by default False
    logy : bool, optional
        Set log of y-axis of main panel, by default True
    xlabel : str | None, optional
        Label of the x-axis, by default None
    ylabel : str | None, optional
        Label of the y-axis, by default None
    ylabel_ratio : list[str] | None, optional
        List of labels for the y-axis in the ratio plots, by default "Ratio"
    label_fontsize : int, optional
        Used fontsize in label, by default 12
    fontsize : int, optional
        Used fontsize, by default 10
    n_ratio_panels : int, optional
        Amount of ratio panels between 0 and 2, by default 0
    vertical_split : bool, optional
        Set to False if you would like to split the figure horizontally. If set
        to True the figure is split vertically (e.g. for pie chart), by
        default False.
    figsize : tuple[float, float] | None, optional
        Tuple of figure size (width, height) in inches, by default None
    dpi : int, optional
        DPI used for plotting, by default 400
    transparent : bool, optional
        Specify if the background of the plot should be transparent, by
        default False
    grid : bool, optional
        Set the grid for the plots, by default True.
    figure_layout : str, optional
        Set the layout that is used for the plot, by default "constrained"
    leg_fontsize : int | None, optional
        Fontsize of the legend, by default None (falls back to fontsize)
    leg_loc : str, optional
        Position of the legend in the plot, by default "upper right"
    leg_linestyle_loc : str, optional
        Position of the linestyle legend in the plot, by default "upper center"
    leg_ncol : int, optional
        Number of legend columns, by default 1
    apply_atlas_style : bool, optional
        Apply ATLAS style for matplotlib, by default True
    use_atlas_tag : bool, optional
        Use the ATLAS Tag in the plots, by default True
    atlas_first_tag : str, optional
        First row of the ATLAS tag (i.e. "ATLAS <atlas_first_tag>"),
        by default "Simulation Internal"
    atlas_second_tag : str | None, optional
        Second row of the ATLAS tag, by default None
    atlas_fontsize : int | None, optional
        Fontsize of ATLAS label, by default None (falls back to fontsize)
    atlas_vertical_offset : float, optional
        Vertical offset of the ATLAS tag, by default 7
    atlas_horizontal_offset : float, optional
        Horizontal offset of the ATLAS tag, by default 8
    atlas_brand : str | None, optional
        brand argument handed to atlasify. Use an empty string or
        None to remove it, by default "ATLAS"
    atlas_tag_outside : bool, optional
        outside argument handed to atlasify. Decides if the ATLAS logo
        is plotted outside of the plot (on top), by default False
    atlas_second_tag_distance : float, optional
        Distance between atlas_first_tag and atlas_second_tag in units
        of line spacing, by default 0
    plotting_done : bool, optional
        Indicates if plotting is done. Only then atlasify() can be called,
        by default False
    """

    title: str = ""
    draw_errors: bool = True

    xmin: float | None = None
    xmax: float | None = None
    ymin: float | None = None
    ymax: float | None = None
    ymin_ratio: list[float | None] | None = None
    ymax_ratio: list[float | None] | None = None
    y_scale: float = 1.3
    logx: bool = False
    logy: bool = True
    xlabel: str | None = None
    ylabel: str | None = None
    ylabel_ratio: list[str] | None = None
    label_fontsize: int = 12
    fontsize: int = 10

    n_ratio_panels: int = 0
    vertical_split: bool = False

    figsize: tuple[float, float] | None = None
    dpi: int = 400
    transparent: bool = False

    grid: bool = True
    figure_layout: str = "constrained"

    # legend settings
    leg_fontsize: int | None = None
    leg_loc: str = "upper right"
    leg_linestyle_loc: str = "upper center"
    leg_ncol: int = 1

    # defining ATLAS style and tags
    apply_atlas_style: bool = True
    use_atlas_tag: bool = True
    atlas_first_tag: str = "Simulation Internal"
    atlas_second_tag: str | None = None
    atlas_fontsize: int | None = None
    atlas_vertical_offset: float = 7
    atlas_horizontal_offset: float = 8
    atlas_brand: str | None = "ATLAS"
    atlas_tag_outside: bool = False
    atlas_second_tag_distance: float = 0

    plotting_done: bool = False

    def __post_init__(self) -> None:
        """Check for allowed values.

        Raises
        ------
        ValueError
            If n_ratio_panels not in [0, 1, 2, 3]
        """
        self.__check_figsize()
        allowed_n_ratio_panels = [0, 1, 2, 3]
        if self.n_ratio_panels not in allowed_n_ratio_panels:
            raise ValueError(
                f"{self.n_ratio_panels} not allwed value for `n_ratio_panels`. "
                f"Allowed are {allowed_n_ratio_panels}"
            )
        self.__check_yratio(self.ymin_ratio)
        self.ymin_ratio = (
            [None] * self.n_ratio_panels if self.ymin_ratio is None else self.ymin_ratio
        )
        self.__check_yratio(self.ymax_ratio)
        self.ymax_ratio = (
            [None] * self.n_ratio_panels if self.ymax_ratio is None else self.ymax_ratio
        )

        if self.ylabel_ratio is None:
            self.ylabel_ratio = ["Ratio"] * self.n_ratio_panels
        elif isinstance(self.ylabel_ratio, str):
            self.ylabel_ratio = [self.ylabel_ratio]
        if len(self.ylabel_ratio) != self.n_ratio_panels:
            raise ValueError(
                f"You passed `ylabel_ratio` of length {len(self.ylabel_ratio)}, "
                f"but `n_ratio_panels` of {self.n_ratio_panels}. "
                f"These should be equal."
            )
        if self.leg_fontsize is None:
            self.leg_fontsize = self.fontsize
        if self.atlas_fontsize is None:
            self.atlas_fontsize = self.fontsize
        if not self.apply_atlas_style and (
            self.atlas_first_tag is not None or self.atlas_second_tag is not None
        ):
            logger.warning(
                "You specified an ATLAS tag, but `apply_atlas_style` is set to false. "
                "Tag will therefore not be shown on plot."
            )

    def __check_figsize(self) -> None:
        """Check `figsize` is a tuple/list of length 2.

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

    def __check_yratio(self, yratio: Sequence[float | None] | None) -> None:
        """Check `yratio` is a sequence of length n_ratio_panels.

        Parameters
        ----------
        yratio : Sequence[float | None] | None
            List of min or max limits of ratio plots

        Raises
        ------
        ValueError
            If `yratio` is not a list and it's length
            is not equal to number of ratio panels
        """
        if yratio is None:
            return
        if not isinstance(yratio, (list, tuple)) or len(yratio) != self.n_ratio_panels:
            raise ValueError(
                f"You passed `min/max_yratio` as {yratio} which is not allowed. "
                f"Either a tuple or a list of size {self.n_ratio_panels} is allowed"
            )


class PlotBase(PlotObject):
    """Base class for plotting.

    Parameters
    ----------
    **kwargs : Any
        Keyword arguments from `puma.PlotObject`
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.axis_top: Axes | None = None
        self.ratio_axes: list[Axes] = []
        self.axis_leg: Axes | None = None
        self.fig: Figure | None = None

    def initialise_figure(self) -> None:
        """Create matplotlib Figure and subplots based on layout options."""
        if self.vertical_split:  # split figure vertically instead of horizonally
            if self.n_ratio_panels >= 1:
                logger.warning(
                    "You set the number of ratio panels to %i but also set the"
                    " vertical splitting to True. Therefore no ratiopanels are"
                    " created.",
                    self.n_ratio_panels,
                )
            self.fig = Figure(figsize=(6, 4.5) if self.figsize is None else self.figsize)
            g_spec = gridspec.GridSpec(1, 11, figure=self.fig)
            self.axis_top = self.fig.add_subplot(g_spec[0, :9])
            self.axis_leg = self.fig.add_subplot(g_spec[0, 9:])

        else:
            # you must use increments of 0.1 for the dimensions
            width = 4.7
            top_height = 2.7 if self.n_ratio_panels else 3.5
            ratio_height = 1.0
            height = top_height + self.n_ratio_panels * ratio_height
            figsize = (width, height) if self.figsize is None else self.figsize
            self.fig = Figure(figsize=figsize, layout=self.figure_layout)

            if self.n_ratio_panels == 0:
                self.axis_top = self.fig.gca()
            elif self.n_ratio_panels > 0:
                g_spec_height = (top_height + ratio_height * self.n_ratio_panels) * 10
                g_spec = gridspec.GridSpec(int(g_spec_height), 1, figure=self.fig)
                self.axis_top = self.fig.add_subplot(g_spec[: int(top_height * 10), 0])
                set_xaxis_ticklabels_invisible(self.axis_top)
                for i in range(1, self.n_ratio_panels + 1):
                    start = int((top_height + ratio_height * (i - 1)) * 10)
                    stop = int(start + ratio_height * 10)
                    sub_axis = self.fig.add_subplot(g_spec[start:stop, 0], sharex=self.axis_top)
                    if i < self.n_ratio_panels:
                        set_xaxis_ticklabels_invisible(sub_axis)
                    self.ratio_axes.append(sub_axis)

        # type-narrowing: required before any use
        assert self.axis_top is not None
        assert self.fig is not None

        # Add the locator to all axes
        self.axis_top.yaxis.set_major_locator(
            locator=MaxNLocator(
                nbins="auto",
                prune="both",
                steps=[1, 2, 5, 10],
            )
        )
        for ratio_axis in self.ratio_axes:
            ratio_axis.yaxis.set_major_locator(
                locator=MaxNLocator(
                    nbins="auto",
                    prune="both",
                    steps=[1, 2, 5, 10],
                )
            )

        if self.grid:
            self.axis_top.grid(lw=0.3)
            for ratio_axis in self.ratio_axes:
                ratio_axis.grid(lw=0.3)

    def draw_vlines(
        self,
        xs: Sequence[float],
        labels: Sequence[str | None] | None = None,
        ys: Sequence[float] | None = None,
        same_height: bool = False,
        colour: str = "#000000",
        linestyle: str = "dashed",
        fontsize: int = 10,
    ) -> None:
        """Draw vertical lines and optional labels on the axes.

        Parameters
        ----------
        xs : Sequence[float]
            List of working points x values to draw
        labels : Sequence[str | None] | None, optional
            List with labels for the vertical lines. Must be the same
            order as the xs. If None, the xvalues * 100 will be
            used as labels. By default None
        ys : Sequence[float] | None, optional
            List with the y height of the vertical lines in percent of the
            upper plot (0 is bottom, 1 is top). Must be the same
            order as the xs and the labels. By default None
        same_height : bool, optional
            Working point lines on same height, by default False
        colour : str, optional
            Colour of the vertical line, by default "#000000" (black)
        linestyle : str, optional
            Linestyle of the vertical line, by default "dashed"
        fontsize : int, optional
            Fontsize of the vertical line text. By default 10.
        """
        assert self.axis_top is not None
        for i, vline_x in enumerate(xs):
            # Set y-point of the WP lines/text
            ytext = (0.65 if same_height else 0.65 - i * 0.07) if ys is None else ys[i]

            self.axis_top.axvline(
                x=vline_x,
                ymax=ytext,
                color=colour,
                linestyle=linestyle,
                linewidth=1.0,
            )

            # Set the number above the line
            self.axis_top.text(
                x=vline_x - 0.005,
                y=ytext + 0.005,
                s=(labels[i] if labels else None),
                transform=self.axis_top.get_xaxis_text1_transform(0)[0],
                fontsize=fontsize,
            )

            for ratio_axis in self.ratio_axes:
                ratio_axis.axvline(x=vline_x, color=colour, linestyle=linestyle, linewidth=1.0)

    def set_title(self, title: str | None = None, **kwargs: Any) -> None:
        """Set title of top panel.

        Parameters
        ----------
        title : str | None, optional
            Title of top panel, if None using the value form the class variables,
            by default None
        **kwargs : Any
            Keyword arguments passed to `matplotlib.axes.Axes.set_title()`
        """
        assert self.axis_top is not None
        self.axis_top.set_title(self.title if title is None else title, **kwargs)

    def set_log(self) -> None:
        """Set log scale of axes as configured."""
        assert self.axis_top is not None
        if self.logx:
            self.axis_top.set_xscale("log")
            for ratio_axis in self.ratio_axes:
                ratio_axis.set_xscale("log")

        if self.logy:
            self.axis_top.set_yscale("log")
            ymin, ymax = self.axis_top.get_ylim()
            self.y_scale = ymin * ((ymax / ymin) ** self.y_scale) / ymax

    def set_y_lim(self) -> None:
        """Set limits of y-axis (main and ratios)."""
        assert self.axis_top is not None
        ymin, ymax = self.axis_top.get_ylim()
        self.axis_top.set_ylim(
            self.ymin if self.ymin is not None else ymin,
            (ymin + (ymax - ymin) * self.y_scale) if self.ymax is None else self.ymax,
        )

        if self.ymin_ratio is None or self.ymax_ratio is None:
            return

        for i, ratio_axis in enumerate(self.ratio_axes):
            if self.ymin_ratio[i] is not None or self.ymax_ratio[i] is not None:
                ymin_i, ymax_i = ratio_axis.get_ylim()
                ymin_i = self.ymin_ratio[i] if self.ymin_ratio[i] is not None else ymin_i
                ymax_i = self.ymax_ratio[i] if self.ymax_ratio[i] is not None else ymax_i
                ratio_axis.set_ylim(bottom=ymin_i, top=ymax_i)

    def set_ylabel(
        self,
        ax_mpl: Axes,
        label: str | None = None,
        align: str | None = "right",
        **kwargs: Any,
    ) -> None:
        """Set y-axis label.

        Parameters
        ----------
        ax_mpl : Axes
            matplotlib axis object
        label : str | None, optional
            x-axis label, by default None
        align : str | None, optional
            Alignment of y-axis label, by default "right"
        **kwargs : Any
            Keyword arguments passed to `matplotlib.axes.Axes.set_ylabel()`
        """
        assert self.fig is not None
        label_options: dict[str, Any] = {"fontsize": self.label_fontsize}
        if align:
            label_options["horizontalalignment"] = align
            if align == "right":
                label_options["y"] = 1
            elif align == "left":
                label_options["y"] = 0

        ax_mpl.set_ylabel(
            self.ylabel if label is None else label,
            **label_options,
            **kwargs,
        )
        self.fig.align_labels()

    def set_xlabel(self, label: str | None = None, **kwargs: Any) -> None:
        """Set x-axis label on the bottom-most axis.

        Parameters
        ----------
        label : str | None, optional
            x-axis label, by default None
        **kwargs : Any
            Keyword arguments passed to `matplotlib.axes.Axes.set_xlabel()`
        """
        assert self.axis_top is not None
        xlabel_args = {
            "xlabel": self.xlabel if label is None else label,
            "horizontalalignment": "right",
            "x": 1.0,
            "fontsize": self.label_fontsize,
        }
        if self.n_ratio_panels == 0:
            self.axis_top.set_xlabel(**xlabel_args, **kwargs)
        else:
            self.ratio_axes[-1].set_xlabel(**xlabel_args, **kwargs)

    def set_tick_params(self, labelsize: int | None = None, **kwargs: Any) -> None:
        """Set tick params on all relevant axes.

        Parameters
        ----------
        labelsize : int | None, optional
            Label size of x- and y- axis ticks, by default None.
            If None then using global fontsize
        **kwargs : Any
            Keyword arguments passed to `matplotlib.axes.Axes.set_xlabel()`
        """
        assert self.axis_top is not None
        labelsize_eff = self.fontsize if labelsize is None else labelsize
        self.axis_top.tick_params(axis="y", labelsize=labelsize_eff, **kwargs)
        if self.n_ratio_panels == 0:
            self.axis_top.tick_params(axis="x", labelsize=labelsize_eff, **kwargs)
        for i, ratio_axis in enumerate(self.ratio_axes):
            ratio_axis.tick_params(axis="y", labelsize=labelsize_eff, **kwargs)
            if i == self.n_ratio_panels - 1:
                ratio_axis.tick_params(axis="x", labelsize=labelsize_eff, **kwargs)

    def set_xlim(self, xmin: float | None = None, xmax: float | None = None, **kwargs: Any) -> None:
        """Set limits of x-axis.

        Parameters
        ----------
        xmin : float | None, optional
            Min of x-axis, by default None
        xmax : float | None, optional
            Max of x-axis, by default None
        **kwargs : Any
            Keyword arguments passed to `matplotlib.axes.Axes.set_xlim()`
        """
        assert self.axis_top is not None
        self.axis_top.set_xlim(
            self.xmin if xmin is None else xmin,
            self.xmax if xmax is None else xmax,
            **kwargs,
        )

    def savefig(
        self,
        plot_name: str,
        transparent: bool | None = None,
        dpi: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Save plot to disk.

        Parameters
        ----------
        plot_name : str
            File name of the plot
        transparent : bool | None, optional
            Specify if plot background is transparent, by default False
        dpi : int | None, optional
            DPI for plotting, by default 400
        **kwargs : Any
            Keyword arguments passed to `matplotlib.figure.Figure.savefig()`
        """
        assert self.fig is not None
        logger.debug("Saving plot to %s", plot_name)
        self.fig.savefig(
            plot_name,
            transparent=self.transparent if transparent is None else transparent,
            dpi=self.dpi if dpi is None else dpi,
            bbox_inches="tight",
            pad_inches=0.04,
            **kwargs,
        )

    def is_running_in_jupyter(self) -> bool:
        """Detect if running inside a Jupyter notebook.

        Returns
        -------
        bool
            If the code is run inside a jupyter notebook
        """
        try:
            shell = get_ipython()

            # Running in standard Python interpreter
            if shell is None:
                return False

            shell_name = shell.__class__.__name__

            # Jupyter notebook or qtconsole
            if shell_name == "ZMQInteractiveShell":
                return True

            # Terminal running IPython
            if shell_name == "TerminalInteractiveShell":
                return False

        # Probably standard Python interpreter
        except (NameError, ImportError):
            return False

        else:
            # Other type (?)
            return False

    def close_window(self, root: tk.Tk | None) -> None:
        """Properly close the Tkinter window and exit the main loop.

        Parameters
        ----------
        root : tk.Tk | None
            The Tkinter root window instance to be closed.
        """
        if root is not None:
            logger.debug("Closing plot window.")

            # Stop the Tkinter main loop and destroy the window
            root.quit()
            root.destroy()

            # Explicitly delete the root object (optional but helps with garbage collection)
            del root

    def show(self, auto_close_after: int | None = None) -> None:
        """Show the plot using tkinter in CLI and detect Jupyter to avoid issues.

        Parameters
        ----------
        auto_close_after : int | None, optional
            After how many milliseconds, the window is automatically closed, by default None

        Raises
        ------
        ValueError
            If the figure is not initalized yet
        """
        if self.is_running_in_jupyter():
            logger.debug("Detected Jupyter Notebook, displaying inline.")
            assert self.fig is not None
            display(self.fig)
            return

        logger.debug("Showing plot using tkinter")

        # Ensure figure is initialized
        if self.fig is None:
            raise ValueError("You need to initalize the figure before using show().")

        # Create the Tkinter root window
        root = tk.Tk()
        root.title("Plot Display")

        # Embed the figure into a Tkinter canvas
        canvas = FigureCanvasTkAgg(self.fig, master=root)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)

        # Render the figure
        canvas.draw()

        # If auto_close_after is set, close the window automatically
        if auto_close_after:
            logger.debug(f"Auto-closing window after {auto_close_after} ms")
            root.after(auto_close_after, lambda: self.close_window(root))

        # Handle window close event manually
        root.protocol("WM_DELETE_WINDOW", lambda: self.close_window(root))

        # Start Tkinter event loop
        root.mainloop()

    def atlasify(self, force: bool = False) -> None:
        """Apply ATLAS style to all axes using the atlasify package.

        Parameters
        ----------
        force : bool, optional
            Force ATLAS style also if class variable is False, by default False
        """
        if not self.plotting_done and not force:
            logger.warning(
                "`atlasify()` has to be called after plotting --> "
                "ATLAS style will not be adapted. If you want to do it anyway, "
                "you can use `force`."
            )
            return

        if self.apply_atlas_style or force:
            assert self.axis_top is not None
            logger.debug("Initialise ATLAS style using atlasify.")
            if self.use_atlas_tag:
                atlasify.atlasify(
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

            for ratio_axis in self.ratio_axes:
                atlasify.atlasify(atlas=False, axes=ratio_axis, enlarge=1)

            if self.vertical_split and self.axis_leg is not None:
                atlasify.atlasify(atlas=False, axes=self.axis_leg, enlarge=1)

            if force:
                if not self.apply_atlas_style:
                    logger.warning(
                        "Initialising ATLAS style even though `apply_atlas_style` is set to False."
                    )
                if not self.plotting_done:
                    logger.warning(
                        "Initialising ATLAS style even though `plotting_done` is set to False."
                    )

    def make_legend(
        self,
        handles: list[lines.Line2D],
        ax_mpl: Axes,
        labels: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Draw legend on a given axis.

        Parameters
        ----------
        handles : list[lines.Line2D]
            List of matplotlib.lines.Line2D object returned when plotting
        ax_mpl : Axes
            `matplotlib.axis.Axes` object where the legend should be plotted
        labels : list[str] | None, optional
            Plot labels. If None, the labels are extracted from the `handles`.
            By default None
        **kwargs : Any
            Keyword arguments which can be passed to matplotlib axis
        """
        if labels is None:
            # remove the handles which have label=None
            handles = [handle for handle in handles if handle.get_label() is not None]
        ax_mpl.add_artist(
            ax_mpl.legend(
                handles=handles,
                labels=([handle.get_label() for handle in handles] if labels is None else labels),
                loc=self.leg_loc,
                fontsize=self.leg_fontsize,
                ncol=self.leg_ncol,
                **kwargs,
            )
        )

    def make_linestyle_legend(
        self,
        linestyles: Sequence[str],
        labels: Sequence[str],
        loc: str | None = None,
        bbox_to_anchor: tuple[float, float] | tuple[float, float, float, float] | None = None,
        axis_for_legend: Axes | None = None,
    ) -> None:
        """Create a legend to indicate what different linestyles correspond to.

        Parameters
        ----------
        linestyles : Sequence[str]
            List of the linestyles to draw in the legend
        labels : Sequence[str]
            List of the corresponding labels. Has to be in the same order as the
            linestyles
        loc : str | None, optional
            Location of the legend (matplotlib supported locations), by default None
        bbox_to_anchor : tuple[float, float] | tuple[float, float, float, float] | None, optional
            Allows to specify the precise position of this legend. Either a 2-tuple
            (x, y) or a 4-tuple (x, y, width, height), by default None
        axis_for_legend : Axes | None, optional
            Axis on which to draw the legend, by default None
        """
        axis_for_legend = self.axis_top if axis_for_legend is None else axis_for_legend
        assert axis_for_legend is not None

        lines_list: list[lines.Line2D] = []
        for linestyle, label in zip(linestyles, labels):
            lines_list.append(
                lines.Line2D(
                    [],
                    [],
                    color="k",
                    label=label,
                    linestyle=linestyle,
                )
            )

        linestyle_legend = axis_for_legend.legend(
            handles=lines_list,
            labels=[handle.get_label() for handle in lines_list],
            loc=(loc if loc is not None else self.leg_linestyle_loc),
            fontsize=self.leg_fontsize,
            bbox_to_anchor=bbox_to_anchor,
            frameon=False,
        )
        axis_for_legend.add_artist(linestyle_legend)

    def set_ratio_label(self, ratio_panel: int, label: str) -> None:
        """Associate the rejection class to a ratio panel.

        Parameters
        ----------
        ratio_panel : int
            Index of the ratio panel to label.
        label : str
            y-axis label of the ratio panel.

        Raises
        ------
        ValueError
            If the requested ratio panel does not exist.
        """
        if ratio_panel > self.n_ratio_panels:
            raise ValueError(f"Plot has {self.n_ratio_panels} ratio panels, not {ratio_panel}")
        assert self.ylabel_ratio is not None
        self.ylabel_ratio[ratio_panel - 1] = label
