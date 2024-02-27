from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from puma.hlplots.results import Results
from puma.hlplots.tagger import Tagger
from puma.hlplots.yutils import get_tagger_name
from puma.utils import get_good_colours


@dataclass
class PlotConfig:
    config_path: Path
    plot_dir: Path

    results_config: dict[str, dict[str, str]]
    taggers_config: dict
    taggers: list[str] | list[Tagger] | None = None

    timestamp: bool = False
    base_path: Path = None

    roc_plots: dict[str, dict] = field(default_factory=dict)
    fracscan_plots: dict[str, dict] = field(default_factory=dict)
    disc_plots: dict[str, dict] = field(default_factory=dict)
    prob_plots: dict[str, dict] = field(default_factory=dict)
    eff_vs_var_plots: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self):
        # Define a plot directory based on the plot config file name, and a date time
        plot_dir_name = self.config_path.stem
        if self.timestamp:
            date_time_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_dir_name += "_" + date_time_file
        self.plot_dir_final = Path(self.plot_dir) / plot_dir_name

        for k, kwargs in self.taggers_config.items():
            kwargs["yaml_name"] = k

    @classmethod
    def load_config(cls, path: Path, **kwargs) -> PlotConfig:
        if not path.exists():
            raise FileNotFoundError(f"Config at {path} does not exist")
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(config_path=path, **config, **kwargs)

    def get_results(self):
        """
        Create the high-level 'Results' object from the config file, using the
        previously set signal and sample. Iterates and loads all models in the config
        file, and adds them.
        """
        kwargs = self.results_config
        kwargs["signal"] = self.signal
        kwargs["perf_vars"] = list({
            plot["args"].get("perf_var", "pt") for plot in self.eff_vs_var_plots
        })

        sample_path = kwargs.pop("sample_path", None)
        if self.base_path and sample_path:
            sample_path = self.base_path / sample_path

        # Instantiate the results object
        results = Results(**kwargs)

        # Store default tag incase other plots need to temporarily modify it
        results.default_atlas_second_tag = results.atlas_second_tag

        good_colours = get_good_colours()
        col_idx = 0
        # Add taggers to results, then bulk load
        for key, t in self.taggers_config.items():
            # if the a sample is not defined for the tagger, use the default sample
            if not sample_path and not t.get("sample_path", None):
                raise ValueError(f"No sample path defined for tagger {key}")
            if sample_path and not t.get("sample_path", None):
                t["sample_path"] = sample_path
            if self.base_path and t.get("sample_path", None):
                t["sample_path"] = self.base_path / t["sample_path"]
            # Allows automatic selection of tagger name in eval files
            t["name"] = get_tagger_name(
                t.get("name", None), t["sample_path"], key, results.flavours
            )
            # Enforces a tagger to have same colour across multiple plots
            if "colour" not in t:
                t["colour"] = good_colours[col_idx]
                col_idx += 1
            if key in self.taggers:
                results.add(Tagger(**t))

        results.load()
        self.results = results

    @property
    def signals(self):
        """Iterates all plots in the config and returns a list of all signals."""
        all_plots = [
            *self.roc_plots,
            *self.fracscan_plots,
            *self.disc_plots,
            *self.prob_plots,
            *self.eff_vs_var_plots,
        ]
        return list({p["args"]["signal"] for p in all_plots})
