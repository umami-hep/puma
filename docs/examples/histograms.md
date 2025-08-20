# Histograms

The following examples use the dummy data which is described [here](./dummy_data.md)

## _b_-tagging Discriminant Plot

<img src=https://github.com/umami-hep/puma/raw/examples-material/histogram_discriminant.png width=500>

```py
--8<-- "examples/plot_discriminant_scores.py"
```

## Flavour Probabilities Plot

<img src=https://github.com/umami-hep/puma/raw/examples-material/histogram_bjets_probability.png width=500>

```py
--8<-- "examples/plot_flavour_probabilities.py"
```

## More General Example

In most cases you probably want to plot histograms with the different flavours
like in the examples above.
However, the `puma` API allows to plot any kind of data. As an example, you
could also produce a `MC` vs `data` plot with the following example code:

<img src=https://github.com/umami-hep/puma/raw/examples-material/histogram_basic_example.png width=500>

```py
--8<-- "examples/plot_basic_histogram.py"
```

## Weighted Histograms

`puma` also supports weighted histograms by specifying the optional argument `weights`.
An example is given below:

<img src=https://github.com/umami-hep/puma/raw/examples-material/histogram_weighted.png width=500>

```py
--8<-- "examples/plot_weighted_histograms.py"
```

## Underflow/Overflow Bins

Underflow and overflow bins are enabled by default, but can be deactivated using the
`underoverflow` attribute of `puma.HistogramPlot`.
Below an example of the same Gaussian distribution plotted with and without
underflow/overflow bins.

<img src=https://github.com/umami-hep/puma/raw/examples-material/hist_without_underoverflow.png width=500>
<img src=https://github.com/umami-hep/puma/raw/examples-material/hist_with_underoverflow.png width=500>

```py
--8<-- "examples/plot_histogram_underoverflow.py"
```

## Iteratively Filling a Histogram

Sometimes, the amount of data/jets you want to histogram, is too large to fit in your RAM. To avoid issues here,
you can use the `update()` function of the `Histogram` class.

<img src=https://github.com/umami-hep/puma/raw/examples-material/Iterative_histogram.png width=500>

```py
--8<-- "examples/plot_iterative_histogram.py"
```

## Data/MC Histograms

To visualize the agreement of the Monte-Carlo with data, `puma` is also able to produce
so-called Data/MC histograms. They show the data as a dot histogram while the MC is still
a stacked histogram. An example of this plot can be seen here:

<img src=https://github.com/umami-hep/puma/raw/examples-material/data_mc_example.png width=500>

The code to create this example can be found in the `examples` folder in the `plot_data_mc.py`.
Similar to the rest of the `Puma.HistogramPlot` examples shown here, a lot of more optional
argument can be passed.

```py
--8<-- "examples/plot_data_mc.py"
```
