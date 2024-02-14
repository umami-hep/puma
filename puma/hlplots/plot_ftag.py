from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ftag import Flavours

from puma.hlplots import (
    PlotConfig,
    get_included_taggers,
    get_plot_kwargs,
    get_signals,
    select_configs,
)
from puma.utils import logger

ALL_PLOTS = ["roc", "scan", "disc", "prob", "peff"]


def get_args(args):
    parser = argparse.ArgumentParser(description="YUMA: Plotting from Yaml in pUMA")
    parser.add_argument(
        "-c", 
        "--config", 
        required=True,
        type=str, help="Path to config")

    parser.add_argument(
        "-p",
        "--plots",
        nargs="+",
        help=f"Plot types to make. Allowed are: {ALL_PLOTS} ",
    )
    parser.add_argument(
        "-s",
        "--signals",
        nargs="+",
        help="Signals to plot",
    )
    return parser.parse_args(args)


def group_eff_vs_var_by_var(sel_configs):
    """
    Sorts eff vs var plots by the variable they plot, so that they can be plotted
    together
    """
    configs_var = []
    for c in sel_configs:
        configs_var.append(c["args"].get("peff_var", "pt"))

    grouped_configs = {
        var: [conf for conf in sel_configs if conf["args"].get("peff_var", "pt") == var]
        for var in set(configs_var)
    }

    return grouped_configs


def make_eff_vs_var_plots(plt_cfg):
    if not plt_cfg.eff_vs_var_plots:
        logger.warning("No eff vs var plots in config")
        return

    eff_vs_var_plots = select_configs(plt_cfg.eff_vs_var_plots, plt_cfg)

    for eff_vs_var in eff_vs_var_plots:
        perf_var = eff_vs_var["args"].get("perf_var", "pt")
        plt_cfg.results.taggers, all_taggers, inc_str = get_included_taggers(
            plt_cfg.results, eff_vs_var
        )
        plot_kwargs = get_plot_kwargs(eff_vs_var, suffix="_" + inc_str + "_" + perf_var)
        if not (bins := eff_vs_var["args"].get("bins", None)):
            if plt_cfg.sample["name"] == "ttbar":
                bins = [20, 30, 40, 60, 85, 110, 140, 175, 250]
            elif plt_cfg.sample["name"] == "zprime":
                bins = [250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5500]
            else:
                raise ValueError(
                    "No bins provided, and no default pt bins for sample"
                    f" {plt_cfg.sample}"
                )
        if plot_kwargs.get("fixed_rejections", False):
            plt_cfg.results.plot_flat_rej_var_perf(
                bins=bins, perf_var=perf_var, **plot_kwargs
            )
        else:
            plt_cfg.results.plot_var_perf(bins=bins, perf_var=perf_var, **plot_kwargs)
        plt_cfg.results.taggers = all_taggers


def make_prob_plots(plt_cfg):
    if not plt_cfg.prob_plots:
        logger.warning("No prob plots in config")
        return
    prob_plots = select_configs(plt_cfg.prob_plots, plt_cfg)

    for prob in prob_plots:
        plt_cfg.results.taggers, all_taggers, inc_str = get_included_taggers(
            plt_cfg.results, prob
        )
        plot_kwargs = get_plot_kwargs(prob, suffix="_" + inc_str)
        plt_cfg.results.plot_probs(**plot_kwargs)
        plt_cfg.results.taggers = all_taggers


def make_disc_plots(plt_cfg):
    if not plt_cfg.disc_plots:
        logger.warning("No disc plots in config")
        return
    disc_plots = select_configs(plt_cfg.disc_plots, plt_cfg)

    for disc in disc_plots:
        plt_cfg.results.taggers, all_taggers, inc_str = get_included_taggers(
            plt_cfg.results, disc
        )
        plot_kwargs = get_plot_kwargs(disc, suffix="_" + inc_str)
        plt_cfg.results.plot_discs(**plot_kwargs)
        plt_cfg.results.taggers = all_taggers


def make_fracscan_plots(plt_cfg):
    if not plt_cfg.fracscan_plots:
        logger.warning("No fracscan plots in config")
        return
    fracscan_plots = select_configs(plt_cfg.fracscan_plots, plt_cfg)

    for fracscan in fracscan_plots:
        # Ideally, long term we'd introduce the ability to also plot tau rejection, in
        # which case we'd be selecting these backgrounds. However, for now they'll
        # always be [cjets, ujets] or [bjets, ujets]
        backgrounds = [Flavours[k] for k in fracscan["args"].get("backgrounds", [])]
        if len(backgrounds) != 2:
            raise ValueError(
                f"Background must be a list of two flavours, got {backgrounds}"
            )

        tmp_backgrounds = plt_cfg.results.backgrounds
        plt_cfg.results.backgrounds = backgrounds

        efficiency = fracscan["args"]["efficiency"]
        frac_flav = fracscan["args"]["frac_flav"]
        if frac_flav not in ["b", "c"]:
            raise ValueError(f"Unknown flavour {frac_flav}")
        info_str = f"$f_{frac_flav}$ scan" if frac_flav != "tau" else "$f_{\\tau}$ scan"
        # info_str += f" {round(efficiency*100)}% {plt_cfg.results.signal.label} WP"
        plt_cfg.results.atlas_second_tag = (
            plt_cfg.default_second_atlas_tag + "\n" + info_str
        )

        eff_str = str(round(efficiency * 100, 3)).replace(".", "p")
        back_str = "_".join([f.name for f in backgrounds])
        suffix = f"_back_{back_str}_eff_{eff_str}_change_f_{frac_flav}"

        # TODO - we have a 'frac flav' which can be used in cases where there are more
        # than 2 backgrounds, such as if we want to extend to tau-jets. It might also be
        # useful for making frac scan plots for X->bb
        plt_cfg.results.taggers, all_taggers, inc_str = get_included_taggers(
            plt_cfg.results, fracscan
        )
        plot_kwargs = get_plot_kwargs(fracscan, suffix=suffix + "_" + inc_str)
        plt_cfg.results.plot_fraction_scans(efficiency=efficiency, **plot_kwargs)
        plt_cfg.results.taggers = all_taggers
        plt_cfg.results.backgrounds = tmp_backgrounds
        plt_cfg.results.atlas_second_tag = plt_cfg.default_second_atlas_tag


def make_roc_plots(plt_cfg):
    if not plt_cfg.roc_plots:
        logger.warning("No ROC plots in config")
        return

    roc_config = select_configs(plt_cfg.roc_plots, plt_cfg)

    for roc in roc_config:
        x_range = roc["args"].get("x_range", [0.5, 1.0])
        if len(x_range) <= 1 or len(x_range) > 3:
            raise ValueError(f"Invalid x_range {x_range}")
        elif len(x_range) == 2:
            x_range = [x_range[0], x_range[1], 20]
        plt_cfg.results.sig_eff = np.linspace(*x_range)

        plt_cfg.results.taggers, all_taggers, inc_str = get_included_taggers(
            plt_cfg.results, roc
        )
        plot_kwargs = get_plot_kwargs(roc, inc_str)

        plt_cfg.results.plot_rocs(**plot_kwargs)
        plt_cfg.results.taggers = all_taggers


def make_plots(args, plt_cfg):
    if "roc" in args.plots:
        make_roc_plots(plt_cfg)

    if "scan" in args.plots:
        make_fracscan_plots(plt_cfg)

    if "disc" in args.plots:
        make_disc_plots(plt_cfg)

    if "prob" in args.plots:
        make_prob_plots(plt_cfg)

    if "peff" in args.plots:
        make_eff_vs_var_plots(plt_cfg)


def main(args=None):
    args = get_args(args)

    # Allow selection of subset of plots, if not, plot all plots
    if args.plots:
        for p in args.plots:
            if p not in ALL_PLOTS:
                raise ValueError(f"Plot type {p} not in allowed list {ALL_PLOTS}")
    else:
        args.plots = ALL_PLOTS

    config_path = Path(args.config)
    plt_cfg = PlotConfig.load_config(config_path)

    signals = get_signals(plt_cfg)

    if args.signals:
        assert all(
            [s in signals for s in args.signals]
        ), f"Signal {args.signals} not in config"
    else:
        args.signals = signals

    logger.info(f"Plotting in {plt_cfg.plot_dir_final}")

    for signal in args.signals:
        plt_cfg.signal = signal
        plt_cfg.get_results()
        make_plots(args, plt_cfg)


if __name__ == "__main__":
    main()
