from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from yamlinclude import YamlIncludeConstructor

from puma.hlplots import PlotConfig, combine_suffixes, get_included_taggers
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


def make_plots(plots, plt_cfg):
    for plot in plt_cfg.plots:
        if not (plot["plot_type"] in plots and plot["signal"] == plt_cfg.signal):
            continue
        plt_cfg.results.taggers, all_taggers, inc_str = get_included_taggers(plt_cfg.results, plot)
        plot_kwargs = plot.get("plot_kwargs", {})
        plot_kwargs["suffix"] = combine_suffixes([plot_kwargs.get("suffix", ""), inc_str])
        plt_cfg.results.make_plot(plot["plot_type"], plot_kwargs)
        plt_cfg.results.taggers = all_taggers


def main(args=None):
    args = get_args(args)

    config_path = Path(args.config)
    YamlIncludeConstructor.add_to_loader_class(
        loader_class=yaml.SafeLoader, base_dir=config_path.parent
    )
    plt_cfg = PlotConfig.load_config(config_path, base_path=args.dir)

    # select and check plots
    plots = args.plots if args.plots else ALL_PLOTS
    if missing := [p for p in plots if p not in ALL_PLOTS]:
        raise ValueError(f"Unknown plot types {missing}, choose from {ALL_PLOTS}")

    # select and check signals
    signals = args.signals if args.signals else plt_cfg.signals
    if missing := [s for s in signals if s not in plt_cfg.signals]:
        raise ValueError(f"Unknown signals {missing}, choose from {plt_cfg.signals}")

    logger.info(f"Plotting in {plt_cfg.plot_dir_final}")

    logger.info("Instantiating Results")
    plt_cfg.signal = signals[0]
    plt_cfg.get_results()  # only run once
    for signal in signals:
        logger.info(f"Plotting signal {signal}")
        plt_cfg.signal = signal
        plt_cfg.results.set_signal(signal)
        plt_cfg.results.output_dir = plt_cfg.plot_dir_final / f"{signal[0]}tagging"
        os.makedirs(plt_cfg.results.output_dir, exist_ok=True)
        make_plots(plots, plt_cfg)


if __name__ == "__main__":
    main()
