# High level API

To set up the inputs for the plots, have a look [here](../examples/index.md).

The following examples use the dummy data which is described [here](../examples/dummy_data.md)

All the previous examples show how to use the plotting of individual plots often requiring
a fair amount of code to produce ROC curves etc.

This high level API facilitates several steps and is designed to quickly plot b- and c-jet
performance plots.


## Initialising the taggers

The `Results` object is initialised with the signal class, by default this is `bjets` but can be changed to `cjets`
to produce the c-tagging plots, or `Hbb`/`Hcc` for Xbb tagging.

```py
--8<-- "examples/high_level_plots.py:1:46"
```


## Probability distributions
You can get the output probability distributions just run
```py
--8<-- "examples/high_level_plots.py:47:48"
```


## Discriminant plots
To plot the discriminant, you can now simply call one function and everything else is handled automatically,
here for the _b_-jet discriminant
```py
--8<-- "examples/high_level_plots.py:50:53"
```



## ROC plots

In the same manner you can plot ROC curves, here for the _b_-tagging performance
```py
--8<-- "examples/high_level_plots.py:55:57"
```



## Performance vs a variable
In this case we plot the performance as a function of the jet pT with the same syntax as above for an inclusive working point of 70%
```py
--8<-- "examples/high_level_plots.py:59:69"
```

and similar for a fixed b-efficiency per bin.
```py
--8<-- "examples/high_level_plots.py:71:78"
```



Similar to above you can also do these plots for _c_-tagging by changing the `signal_class` to `cjets`.


## Fraction scans

Plot the two background efficiencies as a function of the $f_c$ or $f_b$ value.

```py
--8<-- "examples/high_level_plots.py:87:90"
```
