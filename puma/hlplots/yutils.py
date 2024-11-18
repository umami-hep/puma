from __future__ import annotations

import copy
from pathlib import Path

from ftag import Label
from ftag.hdf5 import H5Reader

from puma.utils import logger


def combine_suffixes(suffixes):
    """Combines a list of suffixes into a single suffix."""
    clean_suffixes = [s for s in suffixes if s not in {"", None}]
    if len(clean_suffixes) == 0:
        return None
    return "_".join([s for s in suffixes if s not in {"", None}])


def get_include_exclude_str(include_taggers, all_taggers):
    """Generates the name of the plot, based on the included taggers."""
    if len(include_taggers) == len(all_taggers):
        return ""

    return "taggers_" + "_".join([t.yaml_name for t in include_taggers.values()])


def get_included_taggers(results, plot_config):
    """Converts 'include_taggers' or 'exclude_taggers' into a list of the taggers
    to include.
    """
    all_taggers = results.taggers
    all_tagger_names = [t.yaml_name for t in all_taggers.values()]
    if "include_taggers" not in plot_config and "exclude_taggers" not in plot_config:
        include_taggers = results.taggers
    elif include_taggers := plot_config.get("include_taggers", None):
        assert all(
            t in all_tagger_names for t in include_taggers
        ), f"Not all taggers are in the results: {include_taggers}"

        include_taggers = {
            t: v for t, v in results.taggers.items() if v.yaml_name in include_taggers
        }

    elif exclude_taggers := plot_config.get("exclude_taggers", None):
        assert all(t in all_tagger_names for t in exclude_taggers)
        include_taggers = {
            t: v for t, v in results.taggers.items() if v.yaml_name not in exclude_taggers
        }

    if len(include_taggers) == 0:
        raise ValueError(
            "No taggers included in plot, check that 'exclude_taggers' doesn't exclude "
            "all taggers, or that atleast 1 tagger is defined in 'include_taggers'"
        )
    logger.debug("Include taggers: %s", include_taggers)

    # Set which tagger to use as a reference, if no reference is set, use the first
    #  tagger.This is only needed for plots with a ratio, but still...
    if not any(t.reference for t in include_taggers.values()):
        if reference := plot_config.get("reference", None):
            if reference not in [t.yaml_name for t in include_taggers.values()]:
                raise ValueError(
                    f"Reference {reference} not in included taggers" f" {include_taggers.keys()}"
                )
            reference = str(next(t for t in include_taggers.values() if t.yaml_name == reference))
            # Create a copy, and set it as reference, this is the easiest way of doing
            #  this but might be a bit slow
        else:
            reference = next(iter(include_taggers.keys()))
            logger.info("No reference set for plot, using " + reference + " as reference")

        include_taggers[reference] = copy.deepcopy(include_taggers[reference])
        include_taggers[reference].reference = True

    return (
        include_taggers,
        all_taggers,
        get_include_exclude_str(include_taggers, all_taggers),
    )


def get_tagger_name(name: str, sample_path: Path, key: str, flavours: list[Label]):
    """Attempts to return the name of the tagger if it is not specified in the config
    file by looking at available variable names, and the key of the tagger in the config.

    Parameters
    ----------
    name : str
        The name of the tagger, if specified in the config file
    sample_path : Path
        The path to the sample file
    key : str
        The key of the tagger in the config file, if multiple taggers are found in the
        sample path, this is used to select the correct tagger
    flavours : list[Label]
        The flavours of the tagger, used to identify the correct tagger. A 'valid'
        tagger in the file is one where tagger_p{flav} exists for all flavours defined
        in this list

    Returns
    -------
    str
        The name of the tagger to use
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
        base for base, suffixes in potential_taggers.items() if set(suffixes) == set(req_keys)
    ]

    if len(valid_taggers) == 0:
        raise ValueError("No valid tagger found.")
    if len(valid_taggers) > 1:
        if key in valid_taggers:
            return key
        raise ValueError(
            f"Multiple valid taggers found: {', '.join(valid_taggers)} in file "
            f"{sample_path}, please specify the tagger name in the config file"
        )
    return valid_taggers[0]
