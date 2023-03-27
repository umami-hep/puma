# High level API

To set up the inputs for the plots, have a look [here](./index.md).

The following examples use the dummy data which is described [here](./dummy_data.md)

All the previous examples show how to use the plotting of individual plots often requiring
a fair amount of code to produce ROC curves etc.

This high level API facilitates several steps and is designed to quickly plot b- and c-jet
performance plots.


## Initialising the taggers

The `Results` object is initialised with the signal class, by default this is `bjets` but can be changed to `cjets`
to produce the c-tagging plots, or `Hbb`/`Hcc` for Xbb tagging.

```py
§§§examples/high_level_plots.py:1:48§§§
```

## Discriminant plots
To plot the discriminant, you can now simply call one function and everything else is handled automatically,
here for the _b_-jet discriminant
```py
§§§examples/high_level_plots.py:50:52§§§
```

<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_disc_b.png width=500>


## ROC plots

In the same manner you can plot ROC curves, here for the _b_-tagging performance
```py
§§§examples/high_level_plots.py:54:56§§§
```
<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_roc_b.png width=500>



## Performance vs a variable
In this case we plot the performance as a function of the jet pT with the same syntax as above for an inclusive working point of 70%
```py
§§§examples/high_level_plots.py:59:72§§§
```
<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_dummy_tagger_pt_bjets_eff.png width=500>
<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_dummy_tagger_pt_cjets_rej.png width=500>
<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_dummy_tagger_pt_ujets_rej.png width=500>

and similar for a fixed b-efficiency per bin.
```py
§§§examples/high_level_plots.py:74:84§§§
```

<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_dummy_tagger_fixed_per_bin_pt_bjets_eff.png width=500>
<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_dummy_tagger_fixed_per_bin_pt_cjets_rej.png width=500>
<img src=https://github.com/umami-hep/puma/raw/examples-material/hlplots_dummy_tagger_fixed_per_bin_pt_ujets_rej.png width=500>


Similar to above you can also do these plots for _c_-tagging by changing the `signal_class` to `cjets`.
