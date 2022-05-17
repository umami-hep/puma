# Histograms

To set up the inputs for the plots, have a look [here](./index.md).

The following examples use the dummy data which is described [here](./dummy_data.md)

## _b_-tagging discriminant plot

<img src=https://github.com/umami-hep/puma/raw/examples-material/histogram_discriminant.png width=500>

```py
§§§examples/plot_discriminant_scores.py§§§
```

## Flavour probabilities plot

<img src=https://github.com/umami-hep/puma/raw/examples-material/histogram_bjets_probability.png width=500>

```py
§§§examples/plot_flavour_probabilities.py§§§
```

## Example for basic untypical histogram

In most cases you probably want to plot histograms with the different flavours
like in the examples above.
However, the `puma` API allows to plot any kind of data. As an example, you
could also produce a `MC` vs `data` plot with the following example code:

<img src=https://github.com/umami-hep/puma/raw/examples-material/histogram_basic_example.png width=500>

```py
§§§examples/plot_basic_histogram.py§§§
```
