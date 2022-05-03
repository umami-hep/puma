"""Module for usefule tools in puma."""
from puma.utils.logging import logger, set_log_level


def set_xaxis_ticklabels_invisible(ax):
    """Helper function to set the ticklabels of the xaxis invisible

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis you want to modify
    """

    for label in ax.get_xticklabels():
        label.set_visible(False)
