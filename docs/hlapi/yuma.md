# YUMA (Yaml Plotting)

YUMA (Yaml for pUMA) removes boilerplate required for making simple flavour tagging plots, instead moving everything required into yaml config files, and acting as a wrapper around the high-level plotting functionality included in puma. To make plots, two config files are required, a ```taggers.yaml``` and ```plot_cfg.yaml```. Examples of these can be found in ```puma/examples```.

To create plots, after modifying the two files, run

```bash
yuma --config examples/plot_cfg.yaml 
```

Additional arguments can be included:

- ```--plots [roc, scan, disc, prob, peff]``` Select one or more type of plots to produce.
- ```--signals [bjets, cjets]``` what signals to plot
- ```--num_jets [n]``` number of jets to load per tagger (before cuts are applied)

## taggers.yaml

The ```taggers.yaml``` file contains required information for taggers we wish to load.
Under 'taggers' are the defined taggers. Each should be assigned a tag, which is used as the 'name' of the tagger and its settings. Within in tagger, arguments that can be parsed to the Tagger high-level class can be used.
If you wish to plot the same tagger multiple times, for example with different `fc` values, you can also include a name:

```yaml
GN2_fc0:
    name: GN2v01
    [args]
GN2_fc0p1:
    name: GN2v01
    [args]
```

In this case, both models will be the GN2v01 from whichever sample is specified. You should add `GN2_fc0` and `GN2_fc0p1` as the taggers in the plot config to plot both.

## plot_cfg.yaml

This file contains info on what taggers to load, what plots to make, and where to save them.

### Global options

- ```plot_dir:``` - The base directory to write plots to. The pltos will be saved to a directory of the form ```plot_dir/plt_cfg```.
- ```timestamp: False``` - If True, will create a new directory each time the script is run, with a timestamp included in the name. If False, then will save to the default directory, and overwrite any files.
- ```results_config:``` - Arguments to parse to the 'Results' class.
- ```taggers_config:``` - Path to the ```taggers.yaml``` file.
- ```taggers:``` - List of tagger names that we wish to plot.
- ```reference_tagger:``` - Tagger name that shall be the 'reference' tagger. Any ratios by default are with respect to this tagger.
- ```sample:``` - Information on the sample plotted, including the name, any cuts for the sample, and a sample string to include in the plot info.
- ```all_flavours``` - A list which defines all the flavours available. If defined, then background flavours are defined as `all_flavours - signal`, if not defined, then default backgrounds are assigned (e.g, cjets and lightjets if signal=bjets)
-```label_var``` - What variable define the class labels. By default `HadronConeExclTruthLabelID`

### Plots

Each plot type should come under its own section. The allowed plots are:

- roc_plots
- fracscan_plots
- disc_plots
- prob_plots
- eff_vs_var_plots

Within each section, a list of plots should be included. Each plot requires a 'signal'. Additional options that work for all plot types are:

- suffix: Allows a custom suffix for the plot name.
- include_taggers: Only include the specified taggers for this plot
- exclude_taggers: Plot all taggers, except those in 'exclude_taggers'
- plot_kwargs: arguments parsed to the base plot object, such as figsize.

Other specific plot types have additional arguments that can be included:

#### ROC plots

- x_range: The x-range of the signal efficiency

#### Fraction Scan plots

- frac_flav: The x in 'f_x', for example 'c' for standard f_c plots
- efficiency: The signal efficiency to plot at
- backgrounds: List of two backgrounds to plot the scan over.

#### Efficiency Vs Var plots

- peff_var: The variable to bin on the x-axis, default is pt
bins: The bin edges for the x-axis, default for pT depends on if sample is ttbar/zprime
