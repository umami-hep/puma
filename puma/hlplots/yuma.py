from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml
from yamlinclude import YamlIncludeConstructor

from puma.hlplots import Results, Tagger, combine_suffixes, get_included_taggers
from puma.hlplots.yutils import get_tagger_name
from puma.utils import logger

ALL_PLOTS = ["roc", "scan", "disc", "probs", "peff"]


def get_args(args):
    parser = argparse.ArgumentParser(description="YUMA: Plotting from Yaml in pUMA")
    parser.add_argument("-c", "--config", required=True, type=str, help="Path to config")

    parser.add_argument(
        "-p",
        "--plots",
        nargs="+",
        choices=ALL_PLOTS,
        help=f"Plot types to make. Allowed are: {ALL_PLOTS} ",
    )
    parser.add_argument(
        "-s",
        "--signals",
        nargs="+",
        choices=["bjets", "cjets"],
        help="Signals to plot",
    )
    parser.add_argument("-d", "--dir", type=Path, help="Base sample directory")

    return parser.parse_args(args)


@dataclass
class YumaConfig:
    config_path: Path
    plot_dir: Path

    results_config: dict[str, dict[str, str]]
    taggers_config: dict
    taggers: list[str] | list[Tagger] | None = None

    timestamp: bool = False
    base_path: Path = None

    # dict like {roc : [list of roc plots], scan: [list of scan plots], ...}
    plots: dict[list[dict[str, dict[str, str]]]] = field(default_factory=list)

    def __post_init__(self):
        # Define a plot directory based on the plot config file name, and a date time
        plot_dir_name = self.config_path.stem
        if self.timestamp:
            date_time_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_dir_name += "_" + date_time_file
        self.plot_dir_final = Path(self.plot_dir) / plot_dir_name

        for k, kwargs in self.taggers_config.items():
            kwargs["yaml_name"] = k
        if not self.taggers:
            logger.info("No taggers specified in config, using all")
            self.taggers = list(self.taggers_config.keys())

    @classmethod
    def load_config(cls, path: Path, **kwargs) -> YumaConfig:
        if not path.exists():
            raise FileNotFoundError(f"Config at {path} does not exist")
        with open(path) as f:
            config = yaml.safe_load(f)

        config = cls(config_path=path, **config, **kwargs)
        config.check_config()
        return config

    def check_config(self):
        """Checks the config for any issues, raises an error if any are found."""
        allowed_keys = ["signal", "plot_kwargs", "include_taggers", "exclude_taggers", "reference"]
        for plots in self.plots.values():
            for p in plots:
                for k in p:
                    if k not in allowed_keys:
                        raise ValueError(
                            f"Unknown key {k} in plot config. Maybe '{k}' belongs"
                            "under 'plot_kwargs'?"
                        )

        return True

    def get_results(self):
        """
        Create the high-level 'Results' object from the config file, using the
        previously set signal and sample. Iterates and loads all models in the config
        file, and adds them.
        """
        kwargs = self.results_config
        kwargs["signal"] = self.signal
        kwargs["perf_vars"] = self.peff_vars

        sample_path = kwargs.pop("sample_path", None)
        if self.base_path and sample_path:
            sample_path = self.base_path / sample_path

        # Instantiate the results object
        results = Results(**kwargs, output_dir=self.plot_dir_final)

        # Add taggers to results, then bulk load
        for key, t in self.taggers_config.items():
            if key not in self.taggers:
                continue
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

            results.add(Tagger(**t))

        results.load()
        self.results = results

    @property
    def signals(self):
        """Iterates all plots in the config and returns a list of all signals."""
        return sorted({p["signal"] for pt in self.plots.values() for p in pt})

    @property
    def peff_vars(self):
        """Iterates plots and returns a list of all performance variables."""
        return list({p["plot_kwargs"].get("perf_var", "pt") for p in self.plots.get("peff", [])})

    def make_plots(self, plot_types):
        """Makes all desired plots.

        Parameters
        ----------
        plot_types : list[str]
            List of plot types to make.
        """
        for plot_type, plots in self.plots.items():
            if plot_type not in plot_types:
                continue
            for plot in plots:
                if plot["signal"] != self.signal:
                    continue
                self.results.taggers, all_taggers, inc_str = get_included_taggers(
                    self.results, plot
                )
                plot_kwargs = plot.get("plot_kwargs", {})
                plot_kwargs["suffix"] = combine_suffixes([plot_kwargs.get("suffix", ""), inc_str])
                self.results.make_plot(plot_type, plot_kwargs)
                self.results.taggers = all_taggers


def main(args=None):
    args = get_args(args)

    config_path = Path(args.config)
    YamlIncludeConstructor.add_to_loader_class(
        loader_class=yaml.SafeLoader, base_dir=config_path.parent
    )
    yuma = YumaConfig.load_config(config_path, base_path=args.dir)

    # select and check plots
    plots = args.plots if args.plots else ALL_PLOTS
    if missing := [p for p in plots if p not in ALL_PLOTS]:
        raise ValueError(f"Unknown plot types {missing}, choose from {ALL_PLOTS}")

    # select and check signals
    signals = args.signals if args.signals else yuma.signals
    if missing := [s for s in signals if s not in yuma.signals]:
        raise ValueError(f"Unknown signals {missing}, choose from {yuma.signals}")

    logger.info(f"Plotting in {yuma.plot_dir_final}")

    logger.info("Instantiating Results")
    yuma.signal = signals[0]
    yuma.get_results()  # only run once
    for signal in signals:
        logger.info(f"Plotting signal {signal}")
        yuma.signal = signal
        yuma.results.set_signal(signal)
        yuma.make_plots(plots)


if __name__ == "__main__":
    main()
