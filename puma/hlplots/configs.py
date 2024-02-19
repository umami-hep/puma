from __future__ import annotations

from dataclasses import dataclass
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

    taggers_config: dict
    sample: dict

    results_config: dict[str, dict[str, str]]
    taggers: list[str] | list[Tagger] | None = None
    timestamp: bool = True

    roc_plots: dict[str, dict] = None
    fracscan_plots: dict[str, dict] = None
    disc_plots: dict[str, dict] = None
    prob_plots: dict[str, dict] = None
    eff_vs_var_plots: dict[str, dict] = None

    signal: str = None

    results: Results = None
    default_second_atlas_tag: str = None

    def __post_init__(self):
        # Define a plot directory based on the plot config file name, and a date time
        plot_dir_name = self.config_path.stem
        if self.timestamp:
            date_time_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_dir_name += "_" + date_time_file
        self.plot_dir_final = Path(self.plot_dir) / plot_dir_name

        for k, kwargs in self.taggers_config.items():
            kwargs["yaml_name"] = k

        self.roc_plots = self.roc_plots or {}
        self.fracscan_plots = self.fracscan_plots or {}
        self.disc_plots = self.disc_plots or {}
        self.prob_plots = self.prob_plots or {}
        self.eff_vs_var_plots = self.eff_vs_var_plots or {}

    @classmethod
    def load_config(cls, path: Path) -> PlotConfig:
        if not path.exists():
            raise FileNotFoundError(f"Config at {path} does not exist")
        with open(path) as f:
            config = yaml.safe_load(f)

        return cls(config_path=path, **config)

    def get_results(self):
        """
        Create the high-level 'Results' object from the config file, using the
        previously set signal and sample. Iterates and loads all models in the config
        file, and adds them.
        """
        kwargs = self.results_config
        tag = kwargs.get("atlas_second_tag", "")
        kwargs["atlas_second_tag"] = tag + "\n" + self.sample.pop("tag", "")
        kwargs.update(self.sample)
        kwargs["signal"] = self.signal
        kwargs["perf_vars"] = list(
            {plot["args"].get("perf_var", "pt") for plot in self.eff_vs_var_plots}
        )
        default_sample_path = kwargs.pop("sample_path", None)

        # Store default tag incase other plots need to temporarily modify it
        self.default_second_atlas_tag = kwargs["atlas_second_tag"]

        # Instantiate the results object
        results = Results(**kwargs)

        good_colours = get_good_colours()
        col_idx = 0
        # Add taggers to results, then bulk load
        for key, t in self.taggers_config.items():
            if default_sample_path and not t.get("sample_path", None):
                t["sample_path"] = default_sample_path
            # Allows automatic selection of tagger name in eval files
            t["name"] = get_tagger_name(
                t.get("name", None), t["sample_path"], key, results.flavours
            )
            # Enforces a tagger to have same colour across multiple plots
            if "colour" not in t:
                t["colour"] = good_colours[col_idx]
                col_idx += 1
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
