# High level API

To set up the inputs for the plots, have a look [here](./index.md).

The following examples use the dummy data which is described [here](./dummy_data.md)

All the previous examples show how to use the plotting of individual plots often requiring
a fair amount of code to produce ROC curves etc.

This high level API facilitates several steps and is designed to quickly plot b- and c-jet
performance plots.


## Initialising the taggers

```py
§§§examples/high_level_plots.py:1:55§§§
```
WARNING: when using 2 different data frames you cannot just use one `tagger_args` but you need
as many as you have data frames defining the flavour classes and performance variables.


## Discriminant plots
To plot the discriminant, you can now simply call one function and everything else is handled automatically,
here for the _b_-jet discriminant
```py
§§§examples/high_level_plots.py:56:58§§§
```

<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_disc_b.png width=500>

and similar for the _c_-jet discriminant
```py
§§§examples/high_level_plots.py:59§§§
```

<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_disc_c.png width=500>


## ROC plots

In the same manner you can plot ROC curves, here for the _b_-tagging performance
```py
§§§examples/high_level_plots.py:62:64§§§
```
<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_roc_b.png width=500>

and similar for the _c_-tagging performance
```py
§§§examples/high_level_plots.py:65§§§
```

<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_roc_c.png width=500>


## Performance vs a variable
In this case we plot the performance as a function of the jet pT with the same syntax as above
```py
§§§examples/high_level_plots.py:69:82§§§
```
<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_dummy_tagger_pt_b_eff.png width=500>
<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_dummy_tagger_pt_c_rej.png width=500>
<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_dummy_tagger_pt_light_rej.png width=500>

and similar for the _c_-tagging performance
```py
§§§examples/high_level_plots.py:84:94§§§
```

<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_dummy_tagger_fixed_per_bin_pt_b_eff.png width=500>
<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_dummy_tagger_fixed_per_bin_pt_c_rej.png width=500>
<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_dummy_tagger_fixed_per_bin_pt_light_rej.png width=500>
