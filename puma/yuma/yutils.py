from puma.utils import logger

def get_signals(plt_cfg):
    """Iterates all plots in the config and returns a list of all signals.
    """
    all_plots = [
        *plt_cfg.roc_plots,
        *plt_cfg.fracscan_plots,
        *plt_cfg.disc_plots,
        *plt_cfg.prob_plots,
        *plt_cfg.eff_vs_var_plots,
    ]
    all_sig = list(set([p['args']['signal'] for p in all_plots]))
    return all_sig

def select_configs(configs, plt_cfg):
    '''
    Selects only configs that match the current sample and signal
    '''
    # return configs
    return   [c for c in configs 
                    if c['args']['signal'] == plt_cfg.signal]
    #                 and c['args']['signal'] == plt_cfg.signal)]

def get_plot_kwargs(plt_cfg, config, suffix=''):
    plot_kwargs = config['args'].get('plot_kwargs', {})
    chosen_suffix = config['args'].get('suffix', '')
    plot_kwargs['suffix'] = '_'.join([s for s in [plot_kwargs.get('suffix', ''), suffix, chosen_suffix] if s != ''])
    return plot_kwargs

def get_include_exclude_str(include_taggers, all_taggers):
    
    if len(include_taggers)  == len(all_taggers):
        return ""

    return  'taggers_' + '_'.join([t.yaml_name for t in include_taggers.values()])

def get_included_taggers(results, plot_config):

    all_taggers = results.taggers
    if 'args' not in plot_config or not 'include_taggers' in plot_config['args'] and not 'exclude_taggers' in plot_config['args']:
        include_taggers = results.taggers
    elif 'include_taggers' in plot_config['args']:
        
        # TODO write function that checks all models in icnlude/exclude actually exist...
        # include_taggers_ = plot_config['args']['include_taggers']
        include_taggers = {t:v for t, v in results.taggers.items() if v.yaml_name in plot_config['args']['include_taggers']}
        # exclude_taggers = [t for t in results.taggers.values() if t.filereference not in plot_config['args']['include_taggers']]

    elif 'exclude_taggers':
        include_taggers = {t:v for t,v in results.taggers.items() if v.yaml_name not in plot_config['args']['exclude_taggers']}
        # exclude_taggers = [t for t in results.taggers.values() if t.filereference in plot_config['args']['exclude_taggers']]
    else:
        raise ValueError("Should not be here...")
    logger.debug("Include taggers: %s", include_taggers)
    return include_taggers, all_taggers, get_include_exclude_str(include_taggers, all_taggers)

