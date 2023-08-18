
import argparse
from pathlib import Path

import numpy as np

from ftag import Cuts, Flavour, Flavours

from puma.yuma import ModelConfig, PlotConfig
from puma.utils import logger, set_log_level

ALL_PLOTS=['roc', 'scan', 'disc', 'prob', 'peff']
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
                        '-s',''
                        '--signal',
                        nargs='+',
                        default=['bjets', 'cjets'],
                        help='Signal to plot, by default bjets and cjets',)
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

def select_configs(configs, plt_cfg):
    return   [c for c in configs 
                    if (c['sample'] == plt_cfg.sample
                    and c['args']['signal'] == plt_cfg.signal)]

def make_fracscan_plots(plt_cfg):

    if not plt_cfg.fracscan_plots:
        logger.warning("No fracscan plots in config")
        return
    fracscan_plots = select_configs(plt_cfg.fracscan_plots, plt_cfg)

    for fracscan in fracscan_plots:
        plot_kwargs = fracscan['args'].get('plot_kwargs', {})
        backgrounds = [Flavours[k] for k in fracscan['args'].get('backgrounds', [])]
        if len(backgrounds) != 2:
            raise ValueError(f"Background must be a list of two flavours, got {backgrounds}")

        tmp_backgrounds = plt_cfg.results.backgrounds
        plt_cfg.results.backgrounds = backgrounds

        efficiency = fracscan['args']['efficiency']
        frac_flav = fracscan['args']['frac_flav']
        if frac_flav not in ['b', 'c']:
            raise ValueError(f"Unknown flavour {frac_flav}")
        atlas_second_tag = plt_cfg.default_second_atlas_tag + f"\n $f_{frac_flav}$-tagging"
        info_str = (f"$f_{frac_flav}$ scan" 
                            if frac_flav != 'tau' 
                            else "$f_{\\tau}$ scan")
        info_str += f" {round(efficiency*100)}% {plt_cfg.results.signal.label} WP"
        plt_cfg.results.second_atlas_tag = atlas_second_tag + "\n" + info_str

        eff_str = str(round(efficiency*100,3)).replace('.', 'p')
        back_str = '_'.join([f.name for f in backgrounds])
        suffix=f"_back_{back_str}_eff_{eff_str}_change_f_{frac_flav}"
        plot_kwargs['suffix'] = plot_kwargs.get('suffix', '') + suffix

        plt_cfg.results.plot_fraction_scans(efficiency=efficiency, 
                                                         **plot_kwargs)
        
        plt_cfg.results.backgrounds = tmp_backgrounds

def make_roc_plots(plt_cfg):
    
    if not plt_cfg.roc_plots:
        logger.warning("No ROC plots in config")
        return
    
    roc_config = select_configs(plt_cfg.roc_plots, plt_cfg)
    
    for roc in roc_config:
        
        x_range = roc['args'].get('x_range', [0.5, 1.0])

        plot_kwargs = roc['args'].get('plot_kwargs', {})
        plt_cfg.results.sig_eff = np.linspace(*x_range)
        plt_cfg.results.plot_rocs(**plot_kwargs)


def make_plots(args, plt_cfg):

    if 'roc' in args.plots:
        make_roc_plots(plt_cfg)   
    
    if 'scan' in args.plots:
        make_fracscan_plots(plt_cfg)

if __name__ == '__main__':

    args = get_args()
    if args.debug:
        set_log_level(logger, 'DEBUG')
    if args.plots:
        for p in args.plots:
            if p not in ALL_PLOTS:
                raise ValueError(f"Plot type {p} not in allowed list {ALL_PLOTS}")
    else:
        args.plots = ALL_PLOTS

    config_path = Path(args.config)

    plt_cfg = PlotConfig.load_config(config_path)
    if args.num_jets:
        plt_cfg.num_jets = args.num_jets

    for signal in args.signal:
        for sample in plt_cfg.samples:
            logger.info(f"Making {signal}-tagging plots for {sample}")
            plt_cfg.sample = sample
            plt_cfg.signal = signal
            plt_cfg.get_results()
            make_plots(args, plt_cfg)
    