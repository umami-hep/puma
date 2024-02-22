"""High level plotting API within puma, to avoid code duplication."""

from __future__ import annotations

from puma.hlplots.aux_results import AuxResults
from puma.hlplots.configs import PlotConfig
from puma.hlplots.results import Results
from puma.hlplots.tagger import Tagger
from puma.hlplots.yutils import (
    get_included_taggers,
    get_plot_kwargs,
    select_configs,
)

__all__ = [
    "AuxResults",
    "PlotConfig",
    "Results",
    "Tagger",
    "get_included_taggers",
    "get_plot_kwargs",
    "select_configs",
]
