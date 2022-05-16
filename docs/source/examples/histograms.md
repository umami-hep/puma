# Histograms

To set up the inputs for the plots, have a look [here](./index.md).

The following examples use the dummy data which is described [here](./dummy_data.md)

## _b_-tagging discriminant plot

![discriminant](https://github.com/umami-hep/puma/raw/examples-material/histogram_discriminant.png)

```py linenums="1"
§§§examples/plot_discriminant_scores.py§§§
```

## Flavour probabilities plot

![b-jets probability](https://github.com/umami-hep/puma/raw/examples-material/histogram_bjets_probability.png)

```py linenums="1"
§§§examples/plot_flavour_probabilities.py§§§
```

## Example for basic untypical histogram

In most cases you probably want to plot histograms with with the different flavours
like in the examples above.
However, the python plotting API allows to plot any kind of data. As an example, you
could e.g. produce a `MC` vs `data` plot with the following example code:

![non-ftag example](https://github.com/umami-hep/puma/raw/examples-material/histogram_basic_example.png)

```py
§§§examples/plot_basic_histogram.py§§§
```
