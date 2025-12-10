"""Module for usefule tools in puma."""

from __future__ import annotations

from puma.utils.auxiliary import get_aux_labels
from puma.utils.colours import (
    get_good_colours,
    get_good_linestyles,
    get_good_markers,
    get_good_pie_colours,
)
from puma.utils.generate import (
    get_dummy_2_taggers,
    get_dummy_multiclass_scores,
    get_dummy_tagger_aux,
)
from puma.utils.logger import logger, set_log_level

__all__ = [
    "get_aux_labels",
    "get_dummy_2_taggers",
    "get_dummy_multiclass_scores",
    "get_dummy_tagger_aux",
    "get_good_colours",
    "get_good_linestyles",
    "get_good_markers",
    "get_good_pie_colours",
    "logger",
    "set_log_level",
]
