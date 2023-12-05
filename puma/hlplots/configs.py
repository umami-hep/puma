from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from ftag import Cuts

from puma.hlplots import Results, Tagger
from puma.hlplots.yutils import get_tagger_name
from puma.utils import get_good_colours


@dataclass
class PlotConfig:
    config_path: Path
    plot_dir: Path

    taggers_config: Path
    taggers: list[str] | list[Tagger]
    reference_tagger: str

    sample: dict[str, str]

    # global_plot_kwargs : dict[str, dict[str, str]] = None
    results_default: dict[str, dict[str, str]] = None
    timestamp: bool = True

    roc_plots: dict[str, dict] = None
    fracscan_plots: dict[str, dict] = None
    disc_plots: dict[str, dict] = None
    prob_plots: dict[str, dict] = None
    eff_vs_var_plots: dict[str, dict] = None

    signal: str = None

    # num_jets: int = None

    results: Results = None
    default_second_atlas_tag: str = None
    plot_dir_final: Path = None

    def __post_init__(self):
        # Define a plot directory based on the plot config file name, and a date time
        plot_dir_name = self.config_path.stem
        if self.timestamp:
            date_time_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_dir_name += "_" + date_time_file
        self.plot_dir_final = Path(self.plot_dir) / plot_dir_name
        if not self.results_default:
            self.results_default = {}

        with open(self.taggers_config) as f:
            self.taggers_config = yaml.safe_load(f)
        tagger_defaults = self.taggers_config.get("tagger_defaults", {})
        # print(self.taggers_config.get("taggers", {}))
        taggers = self.taggers_config.get("taggers", {})
        assert (
            self.reference_tagger in taggers
        ), f"Reference tagger {self.reference_tagger} not in taggers config"
        self.taggers = {
            k: {
                **tagger_defaults,
                **t,
                "reference": k == self.reference_tagger,
                "yaml_name": k,
            }
            for k, t in taggers.items()
            if k in self.taggers
        }

        self.roc_plots = self.roc_plots or {}
        self.fracscan_plots = self.fracscan_plots or {}
        self.disc_plots = self.disc_plots or {}
        self.prob_plots = self.prob_plots or {}
        self.eff_vs_var_plots = self.eff_vs_var_plots or {}

    @classmethod
    def load_config(cls, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Config at {path} does not exist")
        with open(path) as f:
            config = yaml.safe_load(f)

        # If the model config is not an absolute path, assume it is relative
        if config["taggers_config"].startswith("/"):
            config["taggers_config"] = Path(config["taggers_config"])
        else:
            config["taggers_config"] = path.parent / config["taggers_config"]

        return cls(config_path=path, **config)

    def get_results(self, perf_var="pt"):
        """Creates the high-level 'Results' object from the config file, using the
        previously set signal and sample. Iterates and loads all models in the config
        file, and adds them
        """
        results_default = {
            "atlas_first_tag": "Simulation Internal",
            "atlas_second_tag": r"$\sqrt{s} = 13.0 $ TeV",
            "global_cuts": Cuts.empty(),
            'sample' : self.sample['name'],
            'perf_var' : perf_var,
            'signal' : self.signal,
            
        }
        results_default.update(self.results_default)
        results_default["atlas_second_tag"] += "\n" + self.sample.get("str", "")

        # Store default tag incase other plots need to temporarily modify it
        self.default_second_atlas_tag = results_default["atlas_second_tag"]
        sample_cuts = Cuts.from_list(self.sample.get("cuts", []))
        results_default["global_cuts"] = results_default["global_cuts"] + sample_cuts

        results = Results(
            **results_default
        )

        good_colours = get_good_colours()
        col_idx = 0
        # Add taggers to results, then bulk load
        for t in self.taggers.values():
            # Allows automatic selection of tagger name in eval files
            t["name"] = get_tagger_name(
                t.get("name", None), t["sample_path"], results.flavours
            )
            # Enforces a tagger to have same colour across multiple plots
            if "colour" not in t:
                t["colour"] = good_colours[col_idx]
                col_idx += 1
            results.add(Tagger(**t))

        results.load()

        final_plot_dir = self.plot_dir_final / f"{self.signal}_tagging"
        final_plot_dir.mkdir(parents=True, exist_ok=True)

        results.output_dir = final_plot_dir
        self.results = results
