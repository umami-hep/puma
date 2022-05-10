"""puma framework - Plotting UMami Api."""
__version__ = "0.0.0rc5"

from histogram import histogram, histogram_plot
from plot_base import plot_base, plot_line_object, plot_object
from roc import roc, roc_plot
from var_vs_eff import var_vs_eff, var_vs_eff_plot


def get_good_colours():
    """List of colours adequate for plotting

    Returns
    -------
    list
        list with colours
    """
    return ["#AA3377", "#228833", "#4477AA", "#CCBB44", "#EE6677", "#BBBBBB"]
