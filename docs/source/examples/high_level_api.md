# High level API

To set up the inputs for the plots, have a look [here](./index.md).

The following examples use the dummy data which is described [here](./dummy_data.md)

All the previous examples show how to use the plotting of individual plots often requiring
a fair amount of code to produce ROC curves etc.

This high level API facilitates several steps and is designed to quickly plot b- and c-jet
performance plots.


## Initialising the taggers

```py
§§§examples/high_level_plots.py:1:56§§§
```
WARNING: when using 2 different data frames you cannot just use one `tagger_args` but you need
as many as you have data frames defining the flavour classes and performance variables.


## Discriminant plots
To plot the discriminant, you can now simply call one function and everything else is handled automatically,
here for the _b_-jet discriminant
```py
§§§examples/high_level_plots.py:58:60§§§
```

<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_disc_b.png width=500>

and similar for the _c_-jet discriminant
```py
§§§examples/high_level_plots.py:51§§§
```

<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_disc_c.png width=500>


## ROC plots

In the same manner you can plot ROC curves, here for the _b_-tagging performance
```py
§§§examples/high_level_plots.py:64:66§§§
```
<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_roc_b.png width=500>

and similar for the _c_-tagging performance
```py
§§§examples/high_level_plots.py:68§§§
```

<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_roc_c.png width=500>


## Performance vs a variable
In this case we plot the performance as a function of the jet pT with the same syntax as above for an inclusive working point of 70%
```py
§§§examples/high_level_plots.py:71:84§§§
```
<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_dummy_tagger_pt_b_eff.png width=500>
<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_dummy_tagger_pt_c_rej.png width=500>
<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_dummy_tagger_pt_light_rej.png width=500>

and similar for a fixed b-efficiency per bin.
```py
§§§examples/high_level_plots.py:86:96§§§
```

<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_dummy_tagger_fixed_per_bin_pt_b_eff.png width=500>
<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_dummy_tagger_fixed_per_bin_pt_c_rej.png width=500>
<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_dummy_tagger_fixed_per_bin_pt_light_rej.png width=500>


Similar to above you can also do these plots for _c_-tagging by changing the `signal_class` to `cjets`.
