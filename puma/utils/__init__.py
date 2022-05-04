"""Module for usefule tools in puma."""

from puma.utils.logging import logger, set_log_level  # noqa: F401


def set_xaxis_ticklabels_invisible(axis):
    """Helper function to set the ticklabels of the xaxis invisible

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis you want to modify
    """

    for label in axis.get_xticklabels():
        label.set_visible(False)
