# YUMA

YUMA (Yaml for pUMA) removes boilerplate required for making simple flavour tagging plots, instead moving everything required into yaml config files, and acting as a wrapper around the high-level plotting functionality included in puma. To make plots, two config files are required, a ```taggers.yaml``` and ```plot_cfg.yaml```. Examples of these can be found in ```puma/examples```.

To create plots, after modifying the two files, run

```bash
python puma/plot_ftag.py --config examples/plot_cfg.yaml 
```

Additional arguments can be included:
- ```--plots [roc, scan, disc, prob, peff]``` Select one or more type of plots to produce.
- ```--signals [bjets, cjets]``` what signals to plot
- ```--num_jets [n]``` number of jets to load per tagger (before cuts are applied)

## taggers.yaml

The ```taggers.yaml``` file contains required information for taggers we wish to load. A section 'tagger_defaults' can be used to include default values, such as f_c, f_b, or cuts. For all taggers that are loaded, these values will be used by default, unless overwritten by the tagger.

Under 'taggers' are the defined taggers. Each should be assigned a tag, which is used as the 'name' of the tagger and its settings. Within in tagger, arguments that can be parsed to the Tagger high-level class can be used.

## plot_cfg.yaml

This file contains info on what taggers to load, what plots to make, and where to save them.

### Global options

- ```plot_dir:``` - The base directory to write plots to. The pltos will be saved to a directory of the form ```plot_dir/plt_cfg```.
- ```timestamp: False``` - If True, will create a new directory each time the script is run, with a timestamp included in the name. If False, then will save to the default directory, and overwrite any files.
- ```results_default: ``` - Arguments to parse to the 'Results' class.
- ```taggers_config: ``` - Path to the ```taggers.yaml``` file.
- ```taggers: ``` - List of tagger names that we wish to plot.
- ```reference_tagger: ``` - Tagger name that shall be the 'reference' tagger. Any ratios by default are with respect to this tagger.
- ```sample: ``` - Information on the sample plotted, including the name, any cuts for the sample, and a sample string to include in the plot info.

### Plots
Each plot type should come under its own section. The allowed plots are:
- roc_plots
- fracscan_plots
- disc_plots
- prob_plots
- eff_vs_var_plots

