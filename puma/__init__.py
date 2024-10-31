"""puma framework - Plotting UMami Api."""

# flake8: noqa

__version__ = "0.4.0"

from puma.histogram import Histogram, HistogramPlot
from puma.integrated_eff import IntegratedEfficiency, IntegratedEfficiencyPlot
from puma.line_plot_2d import Line2D, Line2DPlot
from puma.pie import PiePlot
from puma.plot_base import PlotBase, PlotLineObject, PlotObject
from puma.roc import Roc, RocPlot
from puma.var_vs_eff import VarVsEff, VarVsEffPlot
from puma.var_vs_var import VarVsVar, VarVsVarPlot
from puma.var_vs_vtx import VarVsVtx, VarVsVtxPlot

__all__ = [
    "Histogram",
    "HistogramPlot",
    "Line2D",
    "Line2DPlot",
    "PiePlot",
    "PlotBase",
    "PlotLineObject",
    "PlotObject",
    "Roc",
    "RocPlot",
    "VarVsEff",
    "VarVsEffPlot",
    "VarVsVar",
    "VarVsVarPlot",
    "VarVsVtx",
    "VarVsVtxPlot",
    "IntegratedEfficiency",
    "IntegratedEfficiencyPlot",
]
