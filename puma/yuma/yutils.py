from __future__ import annotations

import copy
from pathlib import Path

from ftag import Flavour
from ftag.hdf5 import H5Reader

from puma.utils import logger


def get_signals(plt_cfg):
    """Iterates all plots in the config and returns a list of all signals."""
    all_plots = [
        *plt_cfg.roc_plots,
        *plt_cfg.fracscan_plots,
        *plt_cfg.disc_plots,
        *plt_cfg.prob_plots,
        *plt_cfg.eff_vs_var_plots,
    ]
    all_sig = list(set([p["args"]["signal"] for p in all_plots]))
    return all_sig


def select_configs(configs, plt_cfg):
    """Selects only configs that match the current sample and signal"""
    return [c for c in configs if c["args"]["signal"] == plt_cfg.signal]


def get_plot_kwargs(config, suffix=""):
    plot_kwargs = config["args"].get("plot_kwargs", {})
    chosen_suffix = config["args"].get("suffix", "")
    plot_kwargs["suffix"] = "_".join(
        [s for s in [plot_kwargs.get("suffix", ""), suffix, chosen_suffix] if s != ""]
    )
    return plot_kwargs


def get_include_exclude_str(include_taggers, all_taggers):
    """Generates the name of the plot, based on the included taggers"""
    if len(include_taggers) == len(all_taggers):
        return ""

    return "taggers_" + "_".join([t.yaml_name for t in include_taggers.values()])


def get_included_taggers(results, plot_config):
    """Converts 'include_taggers' or 'exclude_taggers' into a list of the taggers
    to include
    """
    all_taggers = results.taggers
    all_tagger_names = [t.yaml_name for t in all_taggers.values()]
    if (
        "args" not in plot_config
        or "include_taggers" not in plot_config["args"]
        and "exclude_taggers" not in plot_config["args"]
    ):
        include_taggers = results.taggers
    elif include_taggers := plot_config["args"].get("include_taggers", None):
        assert all(
            [t in all_tagger_names for t in include_taggers]
        ), f"Not all taggers in include_taggers are in the results: {include_taggers}"

        include_taggers = {
            t: v for t, v in results.taggers.items() if v.yaml_name in include_taggers
        }

    elif exclude_taggers := plot_config["args"].get("exclude_taggers", None):
        assert all([t in all_tagger_names for t in exclude_taggers])
        include_taggers = {
            t: v
            for t, v in results.taggers.items()
            if v.yaml_name not in exclude_taggers
        }
    else:
        raise ValueError("Should not be here...")
    logger.debug("Include taggers: %s", include_taggers)

    # Set which tagger to use as a reference, if no reference is set, use the first
    #  tagger.This is only needed for plots with a ratio, but still...
    if not any([t.reference for t in include_taggers.values()]):
        if reference := plot_config["args"].get("reference", None):
            if reference not in [t.yaml_name for t in include_taggers.values()]:
                raise ValueError(
                    f"Reference {reference} not in included taggers"
                    f" {include_taggers.keys()}"
                )
            reference = str(
                next(t for t in include_taggers.values() if t.yaml_name == reference)
            )
            # Create a copy, and set it as reference, this is the easiest way of doing
            #  this but might be a bit slow
        else:
            reference = next(iter(include_taggers.keys()))
            logger.warning(
                "No reference set for plot, using " + reference + " as reference"
            )
        # We ensure that the model which is used for reference as default, is
        # not used as a reference here if don't want it to be.
        default_ref = [k for k in include_taggers if include_taggers[k].reference]
        if len(default_ref) > 0:
            assert len(default_ref) == 1, "More than 1 tagger set as a reference..."
            include_taggers[default_ref[0]] = copy.deepcopy(
                include_taggers[default_ref[0]]
            )
            include_taggers[default_ref[0]].reference = False

        include_taggers[reference] = copy.deepcopy(include_taggers[reference])
        include_taggers[reference].reference = True

    return (
        include_taggers,
        all_taggers,
        get_include_exclude_str(include_taggers, all_taggers),
    )


def get_tagger_name(name: str, sample_path: Path, flavours: list[Flavour]):
    """Attempts to return the name of the tagger if it is not specified in the
    config file by looking at available variable names.
    """
    if name:
        return name

    reader = H5Reader(sample_path)
    jet_vars = reader.dtypes()["jets"].names
    req_keys = [f"_p{flav.name[:-4]}" for flav in flavours]

    potential_taggers = {}

    # Identify potential taggers
    for var in jet_vars:
        for suffix in req_keys:
            if var.endswith(suffix):
                base_name = var.rsplit(suffix, 1)[0]
                if base_name in potential_taggers:
                    potential_taggers[base_name].append(suffix)
                else:
                    potential_taggers[base_name] = [suffix]

    # Check if any base name has all three suffixes
    valid_taggers = [
        base
        for base, suffixes in potential_taggers.items()
        if set(suffixes) == set(req_keys)
    ]

    if len(valid_taggers) == 0:
        raise ValueError("No valid tagger found.")
    elif len(valid_taggers) > 1:
        raise ValueError(f"Multiple valid taggers found: {', '.join(valid_taggers)}")
    else:
        return valid_taggers[0]
