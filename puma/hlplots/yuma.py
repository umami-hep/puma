from __future__ import annotations

from collections import defaultdict
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import copy
import yaml
import re
from yamlinclude import YamlIncludeConstructor
import h5py
from puma.hlplots import Results, Tagger, combine_suffixes, get_included_taggers
from puma.hlplots.yutils import get_tagger_name
from puma.utils import logger
from puma.hlplots.input_distributions import InputDistribution
ALL_PLOTS = ["roc", "scan", "disc", "probs", "peff", "regression_median_sigma", "regression_binned"]


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
    parser.add_argument(
        '--inputs',
        action='store_true',
    )
    parser.add_argument("-d", "--dir", type=Path, help="Base sample directory")

    return parser.parse_args(args)

def numeric_sort_key(path):
    """
    Sort key that extracts numbers from a file path and returns a tuple of integers,
    falling back to the full string for non-numeric parts.
    """
    parts = re.split(r'(\d+)', str(path))
    return [int(p) if p.isdigit() else p for p in parts]

@dataclass
class YumaConfig:
    config_path: Path
    plot_dir: Path
    tagger_defaults: dict[str, dict[str, str]]
    results_config: dict[str, dict[str, str]]
    taggers_config: dict
    inputs_config : dict = None
    taggers: list[str] | list[Tagger] | None = None
    flavours: list[str] = field(default_factory=lambda: ['bjets', 'cjets', 'ujets', 'taujets'])
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
        allowed_keys = ["signal", "plot_kwargs", "include_taggers", "exclude_taggers", "reference", "cuts"]
        for plots in self.plots.values():
            for p in plots:
                if not isinstance(p["signal"], list):
                    p["signal"] = [p["signal"]] 
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
        print("WOW", kwargs["perf_vars"])
        sample_path = kwargs.pop("sample_path", None)
        if self.base_path and sample_path:
            sample_path = self.base_path / sample_path

        backgrounds = [f for f in self.flavours if f != self.signal
        ]
        # Instantiate the results object
        results = Results(**kwargs, output_dir=self.plot_dir_final,
        regression_preds=self.regression_vars,)

        # Add taggers to results, then bulk load
        for key, t in self.taggers_config.items():
            if key not in self.taggers:
                logger.warning(f"Tagger {key} not in config, skipping")
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

            tdefault = copy.deepcopy(self.tagger_defaults)
            tdefault.update(t)
            results.add(Tagger(**tdefault))

        results.load()
        self.results = results

    @property
    def signals(self):
        """Iterates all plots in the config and returns a list of all signals."""
        return sorted({s for pt in self.plots.values() for p in pt for s in p["signal"]})

    @property
    def peff_vars(self):
        """Iterates plots and returns a list of all performance variables."""
        perf_vars = list({p["plot_kwargs"].get("perf_var", "pt") for p in self.plots.get("peff", [])})

        # Get also the truth variables of our regression plots
        for p in self.plots.get("regression_median_sigma", []):
            perf_vars += [p["plot_kwargs"].get("tagger_prediction_name", "pt")]
            perf_vars += [p["plot_kwargs"].get("truth_variable", "pt")]
            perf_vars += [p["plot_kwargs"].get("x_variable", "pt")]
            for pp in p["plot_kwargs"].get("global_predictions", []):
                perf_vars += [pp]

        return list(set(perf_vars))
    
    @property
    def regression_vars(self):
        vars = []
        median_plots = self.plots.get("regression_median_sigma", [])

        for plot in median_plots:
            vars += [plot["plot_kwargs"]["tagger_prediction_name"]]

        vars = list(set(vars))

        return vars

        
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
                if self.signal not in plot["signal"]:
                    continue
                self.results.taggers, all_taggers, inc_str = get_included_taggers(
                    self.results, plot
                )
                cuts = plot.get("cuts", None)
                
                if cuts:
                    self.results.set_cuts(cuts)

                plot_kwargs = copy.deepcopy(plot.get("plot_kwargs", {}))
                plot_kwargs["suffix"] = combine_suffixes([plot_kwargs.get("suffix", ""), inc_str])
                self.results.make_plot(plot_type, plot_kwargs)
                self.results.taggers = all_taggers
    
    def create_index(self, width=3):
        assert width >= 1, "Width must be at least 1"

        html = ['<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '<meta charset="UTF-8">',
            '<title>Plot Index</title>',
            '<style>',
            'body { font-family: sans-serif; margin: 20px; }',
            '.contents { margin-bottom: 40px; }',
            '.grid { display: flex; flex-wrap: wrap; margin: -10px; }',
            f'.plot {{ width: {100/width}%; padding: 10px; box-sizing: border-box; }}',
            'img { width: 100%; height: auto; display: block; }',
            'h2 { margin-top: 60px; border-bottom: 1px solid #ccc; padding-bottom: 5px; }',
            'h3 { margin-top: 40px; }',
            'ul { list-style-type: none; margin-left: 1em; }',
            '</style>',
            '</head>',
            '<body>',
            '<h1>Plot Index</h1>',
            '<div class="contents">',
            '<h2>Contents</h2>'
        ]

        image_paths = sorted(self.plot_dir_final.glob("**/*.png"), key=numeric_sort_key)
        rel_paths = [img.relative_to(self.plot_dir_final) for img in image_paths]
        rel_paths = [p for p in rel_paths if 'inputs' not in str(p)]

        # Group plots by top-level and subfolder
        from collections import defaultdict
        nested = defaultdict(lambda: defaultdict(list))
        for i, rel_path in enumerate(rel_paths):
            parts = rel_path.parts
            top = parts[0] if len(parts) >= 2 else "root"
            sub = "/".join(parts[:-1]) if len(parts) > 1 else top
            nested[top][sub].append((rel_path, f"plot{i}"))

        # Contents section
        html.append("<ul>")
        for top in sorted(nested):
            html.append(f"<li>{top}<ul>")
            for sub in sorted(nested[top]):
                html.append(f"<li>{sub}<ul>")
                for rel_path, anchor in nested[top][sub]:
                    html.append(f'<li><a href="#{anchor}">{rel_path}</a></li>')
                html.append("</ul></li>")
            html.append("</ul></li>")
        html.append("</ul>")
        html.append("</div>")  # End contents

        # Plot sections
        for top in sorted(nested):
            html.append(f'<h2>{top}</h2>')
            for sub in sorted(nested[top]):
                html.append(f'<h3>{sub}</h3>')
                html.append('<div class="grid">')
                for rel_path, anchor in nested[top][sub]:
                    plot_name = rel_path.stem
                    html.append(f'<div class="plot" id="{anchor}">')
                    html.append(f'<h4>{plot_name}</h4>')
                    html.append(f'<img src="{rel_path}">')
                    html.append('</div>')
                html.append('</div>')  # End grid for this subfolder

        html.append('</body></html>')

        index_path = self.plot_dir_final / "index.html"
        with open(index_path, "w") as f:
            f.write("\n".join(html))

    def plot_all_inputs(self):
        if not self.inputs_config:
            raise ValueError("No inputs_config defined in the config file")



        inputdistr = InputDistribution(
            output_path=self.plot_dir_final / "inputs",
            **self.inputs_config,
        )
        # for [norm, logy] in [(False, False), (True, False), (False, True), (True, True)]:
        #     inputdistr.plot(norm=norm, logy=logy)
        inputdistr.create_input_index(
            True
            )
        inputdistr.create_input_index(
            False
            )

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



    if args.inputs:
        yuma.plot_all_inputs()
        
    # return
    logger.info("Instantiating Results")
    yuma.signal = signals[0]
    yuma.get_results()  # only run once
    for signal in signals:
        logger.info(f"Plotting signal {signal}")
        yuma.signal = signal
        yuma.results.set_signal(signal)
        yuma.make_plots(plots)
    yuma.create_index()

if __name__ == "__main__":
    main()
