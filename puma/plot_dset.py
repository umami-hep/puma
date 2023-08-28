import argparse
from pathlib import Path
from copy import deepcopy
from puma.utils import logger, set_log_level, get_good_linestyles
from puma.yuma import VariablePlotConfig
import numpy as np
from puma import (
    Histogram,
    HistogramPlot,)

ALL_PLOTS=["jets", "tracks"]
def get_args():
    args = argparse.ArgumentParser(description='YUMA: Plotting from Yaml in pUMA')
    args.add_argument('-c', '--config', type=str, help='Path to config')
    
    args.add_argument(
                        '-p',
                        '--plots', 
                        nargs='+', 
                        help=f'Plot types to make. Allowed are: {ALL_PLOTS} ',
                        )
    args.add_argument(
        "-n",
        "--num_jets",
        required=False,
        type=int,
        help='Maximum number of jets to use in plotting, per model.'
    )
    args.add_argument(
        '-d',
        '--debug',
        action='store_true',
        help='Set logger level to debug.'
    )
    return args.parse_args()

SHOULD_LOGY=[
    'd0',
    'z0'
]
def make_var_hist_plot(plt_cfg, plot_type, var, flavours=None):
    plot_path = plt_cfg.plot_dir_final / plt_cfg.sample / plot_type / var
    plot_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Plotting {plot_type}/{var}")
    histo = HistogramPlot(
        n_ratio_panels=1,
        xlabel=var,
        ylabel="Normalised number of jets",
    )
    has_data = False
    ls = list(get_good_linestyles())
    
    flavour_str = '' if flavours is None else '_' + '_'.join(flavours)
    if flavours is None:
        flavours = [f.name for f in plt_cfg.flavours]
        
    
    for i, dset in enumerate(plt_cfg.datasets.keys()):
        style = plt_cfg.datasets[dset].style
        data = plt_cfg.loaded_datasets[dset][plot_type]

        for j, flav in enumerate(flavours):
            # data[flav][var] = data[flav][var][~np.isnan(data[flav][var])]
            no_nan = data[flav][var][~np.isnan(data[flav][var])]
            if len(no_nan) == 0:
                logger.warning(f"No data for {dset} {flav} {var}")
                continue

            hist_plt = Histogram(
                no_nan,
                label=style['label'] if j == 0 else None,
                ratio_group=flav ,
                colour=style['colour'], 
                linestyle=ls[j])

            histo.add(hist_plt,
                      reference=dset == plt_cfg.denominator,)
            has_data = True
    
    if not has_data:
        logger.warning(f"No data for {plot_type}/{var} {flavour_str}")
        return
    histo.draw()
    if len(flavours) > 1:
        histo.make_linestyle_legend(
                linestyles=ls[:len(flavours)],
                labels=flavours,
                bbox_to_anchor=(0.55, 1),
            )
    histo.savefig(plot_path / f'{var}_hist{flavour_str}.png')

def make_plot(plt_cfg, plot):
    plt_type, plt_var = plot['var'].split('/')    
    flavours = [[f.name] for f in plt_cfg.flavours] + [None]

    for f in flavours:
        if plot['type'] == 'histogram':
            make_var_hist_plot(plt_cfg, plt_type, plt_var, f)

def make_plots(plt_cfg):

    for plot in plt_cfg.plots:
        if plot['var'] == 'all':
            if plt_cfg.keys is None:
                plt_cfg.keys = plt_cfg.variables.keys()
            for key in plt_cfg.keys:
                for var in plt_cfg.variables[key]:
                    plot_copy = deepcopy(plot)
                    plot_copy['var'] = f"{key}/{var}"
                    make_plot(plt_cfg, plot_copy)
        else:
            make_plot(plt_cfg, plot)

if __name__ == '__main__':

    args = get_args()
    if args.debug:
        set_log_level(logger, 'DEBUG')
    # if args.plots:
    #     for p in args.plots:
    #         if p not in ALL_PLOTS:
    #             raise ValueError(f"Plot type {p} not in allowed list {ALL_PLOTS}")
    # else:
    #     args.plots = ALL_PLOTS

    config_path = Path(args.config)

    plt_cfg = VariablePlotConfig.load_config(config_path)
    logger.info(f"Making plots for {plt_cfg.sample} with {plt_cfg.num_jets} jets")
    if args.num_jets:
        plt_cfg.num_jets = args.num_jets
    if args.plots:
        plt_cfg.keys = args.plots

    for sample in plt_cfg.samples:
        logger.info(f"Making variable plots for {sample}")
        plt_cfg.sample = sample
        plt_cfg.load_datasets()
        make_plots(plt_cfg)