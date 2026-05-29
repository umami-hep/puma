# Puma Plotting Tutorial

## Introduction

In this tutorial you will learn how to use `puma`, the _Plotting UMami API_. The idea behind `puma`
is to provide a plotting API that is easy to use but at the same time highly configurable. This
means that the user has full control over the plot while things like uncertainties, labels, ratio
panels can be added easily.

You can find the `puma` documentation [here](https://umami-hep.github.io/puma/).

`puma` is based on `matplotlib` and helps you to produce most of the types of plots that are
commonly used in flavour tagging like the ones shown below:

|                                     ROC curves                                      |                                            Histogram plots                                             |                                    Variable vs efficiency                                    |
| :---------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: |
| <img src=https://github.com/umami-hep/puma/raw/examples-material/roc.png width=200> | <img src=https://github.com/umami-hep/puma/raw/examples-material/histogram_discriminant.png width=220> | <img src=https://github.com/umami-hep/puma/raw/examples-material/pt_light_rej.png width=220> |

In this tutorial you will learn how to:

0. A small introduction to the different metrics used in flavour tagging.
1. Plot histograms with `puma`, where you will produce plots of both jet- and track-variables.
2. Plot ROC curves of two taggers, with ratio panels that compare the curves.
3. Plot the efficiency/rejection of a tagger as a function of $p_\text{T}$.
4. Stream large files in batches and use the high-level `puma` plotting interface.
5. Try a few other plot types supported by `puma`.

The tutorial is meant to be followed in a self-guided manner. You will be prompted to do certain
tasks by telling you what the desired outcome will be, without telling you how to do it. Using the
documentation of `puma`, you can find out how to achieve your goal. In case you are stuck, you can
click on the "hint" toggle box to get a hint. If you tried for more than 10 min at a problem, feel
free to toggle also the solution with a worked example.

## Setting Up Everything You Need

For the tutorial, you will need `puma`. Probably the easiest way to get `puma` is via `PyPI`.
You can install it like any other Python package using the following command:

```bash
pip install puma-hep
```

You can also use so-called containers for the tutorial or in general for `puma`. This is the
recommended way to run `puma` if you have different python versions and dependencies running on your
system. You can read more about this in the [FTAG Docs](https://ftag.docs.cern.ch/software/container_usage/).

For this tutorial, two `.h5` files were prepared, one $t\bar{t}$ and one $Z'$ file. You can download
them from the `eos` directly. The path to the directory where the files are stored is
`/eos/user/u/umamibot/tutorials/`. If you don't have access to `eos`, you can simply download
the files via the following two commands:

```bash
wget https://umami-ci-provider.web.cern.ch/puma/ttbar.h5
wget https://umami-ci-provider.web.cern.ch/puma/zpext.h5
```

## Tutorial Tasks

The tasks are divided into several subtasks. You don't have to do all of them in case you are more
interested in the other tasks. However, the subtasks depend on each other, so you should finish a
subtask before proceeding to the next one (also the solutions assume that you already have the
previous subtasks completed).

### Task 0: Flavour Tagging Metrics

To get started with the plotting, we will first have a quick look into different metrics
used in flavour tagging to evaluate the performance.

#### Task 0.1: Generating Dummy Data

The typical output of our ML-based taggers, like DIPS or DL1d, are 3 scores indicating the
probabilities of being a light-flavour jet, a $c$-jet and a $b$-jet. For example the scores
`[0.1, 0.3, 0.6]` would indicate that we have most probably a $b$-jet while `[0.7, 0.2, 0.1]` would
indicate a light-flavour jet.

Even though we always have MC simulation, it is sometimes useful to have dummy data, for instance if
we want to test certain plotting features etc. (and to understand the actual underlying
distributions).

Now, it is up to you. **Generate a dummy multi-class output of a neural network.**

??? info "Hint: Where can I find such a function?"
    You can have a look at the [puma documentation](https://umami-hep.github.io/puma/)
    and search in the API reference.

??? info "Hint: Which exact function?"
    The `puma.utils` module contains the desired functions

??? warning "Solution"
    Dummy data are explained [here](https://umami-hep.github.io/puma/examples/dummy_data/)

    ```py
    from puma.utils import get_dummy_multiclass_scores

    # Generating 10_000 jets
    dummy_jets, dummy_labels = get_dummy_multiclass_scores()
    print(dummy_jets)
    print(dummy_labels)
    ```

#### Task 0.2: Defining Working Points - Likelihood Ratio

Since we are not (yet) able to calibrate the entire spectrum of the different multi-class outputs,
we need to define so-called working points (or operating points). In the case of ATLAS, we have
four/five different $b$-tagging working points (WPs) which are defined covering various needs of the
physics analyses. The efficiency of a specific flavour $j$ ($b$, $c$ or light) is defined as

$$
\varepsilon^j = \frac{N_\text{pass}^j(\mathcal{D}>T_f)}{N_\text{total}^j},
$$

where $N_\text{pass}^j(\mathcal{D}>T_f)$ are the number of jets of flavour $j$ passing the cut $T_f$
on the tagger discriminant $\mathcal{D}$ and $N_\text{total}^j$ are all jets of flavour $j$ before
the cut.

The final output score is calculated from the multi-class output and results for the $b$-tagging
discriminant into the log-likelihood

$$
\mathcal{D}_\text{b}(f_c) = \log \left(  \frac{p_b}{f_c\cdot p_c+(1-f_c)\cdot p_u} \right),
$$

with $p_b$, $p_c$ and $p_u$ being the probabilities for the jet to be a $b$-jet, $c$-jet or
light-flavour jet (often referred to as $u$-jets), respectively. The $c$-jet fraction value $f_c$
allows to tune how much emphasis is given to the $c$-jet or to the light-flavour performance. While
the $c$-jet rejection increases as a function of $f_c$, the light-flavour jet rejection decreases.
This parameter has to be tuned separately for each tagger and depends on the needs of the physics
analyses.

The advantage of the multi-class output is that this tuning is possible after the training and the
$c$-jet fraction value in the training sample does not have to be adapted. Another advantage of the
multi-class output is that one can by changing the log-likelihood to

$$
\mathcal{D}_\text{c}(f_b) = \log \left(  \frac{p_c}{f_b\cdot p_b+(1-f_b)\cdot p_u} \right),
$$

perform $c$-tagging without the need of retraining the tagger. Here $f_b$ is now the $b$-jet
fraction value.

**Define a function which calculates the log-likelihood, when giving it the 3 scores and the $f_c$ value as input.**

??? info "Hint 1"
    You can either use a python function `def` or a `lambda` function

??? info "Hint 2"
    You can find the solution in the
    [atlas-ftag-tools](https://github.com/umami-hep/atlas-ftag-tools/tree/main) package.

??? warning "Solution"
    This implementation here is hardcoded to explain how it works. In general, it is recommended
    to use the `get_discriminant` function from the 
    [atlas-ftag-tools](https://github.com/umami-hep/atlas-ftag-tools/tree/main) package

    ```py
    import numpy as np
    def disc_fct(arr: np.ndarray, f_c: float = 0.018) -> np.ndarray:
        """Tagger discriminant

        Parameters
        ----------
        arr : numpy.ndarray
            array with shape (3,)
        f_c : float, optional
            f_c value in the discriminant (weight for $c$-jets rejection)

        Returns
        -------
        np.ndarray
            Array with the discriminant values inside.
        """
        # you can adapt this for your needs
        return np.log(arr[2] / (f_c * arr[1] + (1 - f_c) * arr[0]))


    # you can also use a lambda function
    # fc = 0.018
    # lambda a: np.log(a[2] / (fc * a[1] + (1 - fc) * a[0]))

    ```

**Using the `dummy data` from task 0.1, calculate the log-likelihood with $f_c=0.018$ and
retrieve the working point cut value for 70% $b$-jet efficiency.**

??? info "Hint: Where to get the Labels from?"
    The `labels` from task 0.1 have the same values as the `HadronConeExclTruthLabelID`
    described in the [FTAG algo docs](https://ftag.docs.cern.ch/algorithms/labelling/jet_labels/#delta-r-matching-default-scheme-in-ftag).

??? info "Hint: Which function to use?"
    You can have a look at the `percentile` function from `numpy`.
    Be aware from which side we need to integrate!
    And the `apply_along_axis` function to evaluate an entire array.

??? warning "Solution"
    ```py
    import numpy as np

    bjets = dummy_labels == 5
    dummy_jets_2d = np.column_stack((dummy_jets['ujets'], dummy_jets['cjets'], dummy_jets['bjets']))
    scores = np.apply_along_axis(disc_fct, axis=1, arr=dummy_jets_2d)
    target_eff = 0.7
    cutvalue = np.percentile(scores[bjets], 100.0 * (1.0 - target_eff))
    print("cut value for 70% $b$-jet efficiency:", cutvalue)
    ```

    The current way to implement this is using the [atlas-ftag-tools metrics functions](https://github.com/umami-hep/atlas-ftag-tools/blob/main/ftag/utils/metrics.py)

#### Task 0.3: Performance Metrics - Efficiency and Rejection

To quantify the performance of a tagger at a given working point, the background rejection is a good
measure. The rejection is simply the inverse of the efficiency $\frac{1}{\varepsilon^j}$.

**Calculate the light-flavour jet and $c$-jet rejection for the 70% working point from task 0.2.**

??? warning "Solution"
    ```py
    ljets = dummy_labels == 0
    cjets = dummy_labels == 4

    ljets_eff = ((scores > cutvalue) & ljets).sum() / ljets.sum()
    print("light-flavour jets efficiency:", ljets_eff)
    print("light-flavour jets rejection:", 1 / ljets_eff)

    cjets_eff = ((scores > cutvalue) & cjets).sum() / cjets.sum()
    print("$c$-flavour jets efficiency:", cjets_eff)
    print("$c$-flavour jets rejection:", 1 / cjets_eff)
    ```

    Alternatively, all this functionality is also provided by the `atlas-ftag-tools` package.
    In that case this would simplify to
    ```py
    from ftag.utils import calculate_rejection
    ljets = dummy_labels == 0
    cjets = dummy_labels == 4
    rej = calculate_rejection(scores[bjets], scores[ljets], target_eff=0.7)
    print("light-flavour jets rejection:", rej)
    ```

Starting from these metrics, we can plot for instance:

- **ROC curves**: which show the background rejection as a function of the $b$-jet efficiency
- **Efficiency vs $p_\text{T}$**: where one fixes a working point and calculates the background rejection in bins of $p_\text{T}$

### Task 1: Histogram Plots

#### Task 1.1: Loading the h5 File

Before starting with the different plotting exercises, you have to load the h5 file that was
prepared for this tutorial. The expected outcome of this is that you have access to the jet
variables as well as to the track variables. You can put the jet variables in a `pandas.DataFrame`
in case you feel more comfortable with that, but this will not be possible for the tracks, since the
`tracks_loose` dataset in the h5 file has an extra dimension for the tracks (for each jet we store
the information of up to 40 tracks).

1. Write a little python script that loads the jet and track variables.
2. Have a look at how large the dataset is, and what shape the loaded arrays have.

For the following tasks you can re-use the code that loads the h5 file or just extend your python
script from this task.

??? info "Hint: How can I load an h5 file with Python?"

    You can find the documentation of `h5py` [here](https://docs.h5py.org/en/stable/quick.html).

??? warning "Solution"

    ```py
    import h5py

    # Use directly the eos path or use the path to your downloaded files
    ttbar_filepath = "/eos/user/u/umamibot/tutorials/ttbar.h5"

    # load the "jets" and "tracks_loose" dataset from the h5 file
    with h5py.File(ttbar_filepath, "r") as h5file:
        jets = h5file["jets"][:]
        tracks = h5file["tracks_loose"][:]

    # print the shape and the field names of the datasets
    print(jets.shape)
    print(jets.dtype.names)
    print(tracks.shape)
    print(tracks.dtype.names)
    ```

#### Task 1.2: Plotting the $p_\text{T}$ Distribution for Jets of Different Flavours

As a next step, you will produce a histogram plot that shows the $p_\text{T}$ distribution of light-flavour
jets, $c$-jets and $b$-jets.

??? info "Hint: How do I Create a Histogram Plot with `puma`?"

    You can find the examples of histogram plots
    [here](https://umami-hep.github.io/puma/examples/histograms/) and the
    documentation for histogram plots with `puma`
    [here](https://umami-hep.github.io/puma/api/histogram/).

??? warning "Solution"

    ```py
    import h5py
    import numpy as np
    from ftag import Flavours
    from puma import Histogram, HistogramPlot

    # Use directly the eos path or use the path to your downloaded files
    ttbar_filepath = "/eos/user/u/umamibot/tutorials/ttbar.h5"

    # load the jets dataset from the h5 file
    with h5py.File(ttbar_filepath, "r") as h5file:
        jets = h5file["jets"][:]

    # defining boolean arrays to select the different flavour classes
    is_light = jets["HadronConeExclTruthLabelID"] == 0
    is_c = jets["HadronConeExclTruthLabelID"] == 4
    is_b = jets["HadronConeExclTruthLabelID"] == 5

    # initialise the plot
    pt_plot = HistogramPlot(
        xlabel=r"$p_\text{T}$ [GeV]",
        ylabel="Normalised number of jets",
    )

    # add the histograms (Note that the Histogram objects need to have the same bins!)
    bins = np.linspace(0, 250, 50)
    pt_plot.add(Histogram(jets[is_light]["pt"] / 1000, flavour=Flavours["ujets"], bins=bins))
    pt_plot.add(Histogram(jets[is_c]["pt"] / 1000, flavour=Flavours["cjets"], bins=bins))
    pt_plot.add(Histogram(jets[is_b]["pt"] / 1000, flavour=Flavours["bjets"], bins=bins))

    pt_plot.draw()
    pt_plot.savefig("tutorial_histogram_pT.png")
    ```

#### Task 1.3: Plot the $b$-jets Probability Output of two Different Taggers

In this task you will plot the $b$-jets probability of two different taggers - RNNIP and DIPS.

1.  Create the histogram plot (similar to the one from the previous task) and the different
    histograms. If you plot this for light-flavour jets, $c$-jets and $b$-jets, you should have 6
    histograms.
2.  Make sure that you use a different linestyle for the histograms of your second tagger.
3.  Add a ratio panel to the plot
4.  Make your plot look pretty. Have a look at the arguments that are supported by
    [`puma.PlotObject`](https://umami-hep.github.io/puma/api/plot_base/#puma.plot_base.PlotObject).

??? info "Hint 1: Histogram and HistogramPlot Objects"

    After you defined your HistogramPlot object, you can start adding lines to it.
    This lines are the Histogram objects you need to define.

??? info "Hint 2: Linestyle"

    The `linestyle` can be set when the different `Histogram` lines are initialised.

??? info "Hint 3: Ratio Panel"

    The ratio for the ratio panel is calculated in this case between the same flavours. But you
    need to tell the plot which of the Histogram objects is the reference. Try to look up the
    `reference` option in the `add()` function.

??? warning "Solution"

    ```py
    tagger_output_plot = HistogramPlot(
        n_ratio_panels=1,
        xlabel="$b$-jets probability",
        ylabel="Normalised number of jets",
        # optional:
        # figsize=(6, 4.5),
        # leg_ncol=2,
        # atlas_second_tag="$t\\bar{t}$ R22 sample",
        # logy=True,
    )

    # add the histograms
    tagger_output_plot.add(
        Histogram(
            jets[is_light]["rnnip_pb"],
            ratio_group="ujets",
            flavour=Flavours["ujets"],
            bins=np.linspace(0, 1, 50),
        ),
        reference=True,
    )
    tagger_output_plot.add(
        Histogram(
            jets[is_c]["rnnip_pb"],
            ratio_group="cjets",
            flavour=Flavours["cjets"],
            bins=np.linspace(0, 1, 50),
        ),
        reference=True,
    )
    tagger_output_plot.add(
        Histogram(
            jets[is_b]["rnnip_pb"],
            ratio_group="bjets",
            flavour=Flavours["bjets"],
            bins=np.linspace(0, 1, 50),
        ),
        reference=True,
    )

    # add the histograms
    tagger_output_plot.add(
        Histogram(
            jets[is_light]["dipsLoose20220314v2_pb"],
            ratio_group="ujets",
            flavour=Flavours["ujets"],
            bins=np.linspace(0, 1, 50),
            linestyle="--",
        )
    )
    tagger_output_plot.add(
        Histogram(
            jets[is_c]["dipsLoose20220314v2_pb"],
            ratio_group="cjets",
            flavour=Flavours["cjets"],
            bins=np.linspace(0, 1, 50),
            linestyle="--",
        )
    )
    tagger_output_plot.add(
        Histogram(
            jets[is_b]["dipsLoose20220314v2_pb"],
            ratio_group="bjets",
            flavour=Flavours["bjets"],
            bins=np.linspace(0, 1, 50),
            linestyle="--",
        )
    )

    tagger_output_plot.draw()
    tagger_output_plot.savefig("tutorial_histogram_tagger_pb_comparison.png")
    ```

#### Task 1.4: Plot a Track Variable of Your Choice

In this task you are asked to make a histogram plot of a _track variable_. This is slightly more
tricky, since the array that you load from the h5 file has a different shape compared to the array
storing the jet information. In addition to that, many entries might be filled with `nan` values,
which is challenging here and there.

1.  Choose a track variable that you want to plot.
2.  Create a histogram plot (maybe again for multiple flavours, but that is up to you).

??? info "Hint 1: NaN-Values in Binning"

    If you encounter an issue with NaN values in the binning, you need to set the
    `bins_range` correctly, because with NaN values it cannot be calculated automatically.

??? info "Hint 2: Difference in Shape"

    Due to the dimensionality of tracks, you need to get rid of one of the dimensions.
    Try the `.flatten()` option of `numpy.ndarray`'s

??? warning "Solution"

    ```py
    with h5py.File(ttbar_filepath, "r") as h5file:
        tracks = h5file["/tracks_loose"][:, :]
        print(tracks.shape)

    d0_plot = HistogramPlot(
        xlabel="$d_0$ significance",
        ylabel="Normalised number of tracks",
        figsize=(6, 4.5),
    )

    d0_plot.add(
        Histogram(
            tracks["IP3D_signed_d0_significance"][is_light, :].flatten(),
            flavour=Flavours["ujets"],
            bins=np.linspace(-3, 3, 50),
        )
    )
    d0_plot.add(
        Histogram(
            tracks["IP3D_signed_d0_significance"][is_c, :].flatten(),
            flavour=Flavours["cjets"],
            bins=np.linspace(-3, 3, 50),
        )
    )
    d0_plot.add(
        Histogram(
            tracks["IP3D_signed_d0_significance"][is_b, :].flatten(),
            flavour=Flavours["bjets"],
            bins=np.linspace(-3, 3, 50),
        )
    )

    d0_plot.draw()
    d0_plot.savefig("tutorial_histogram_track_variable.png")
    ```

### Task 2: ROC Plots

In this task, you will plot a ROC comparison for the two taggers _RNNIP_ and _DIPS_.

#### Task 2.1: Calculate the Rejections as a Function of the $b$-jets Efficiency

Before you can actually plot the ROC curves, you have to calculate the light-flavour and $c$-jets
rejection for a range of $b$-jets efficiencies.

1.  Define a function that calculates the $b$-jets discriminant from the tagger output.
2.  Calculate the light-flavour jets rejection as a function of the $b$-jets efficiency.

??? info "Hint: Look at Examples"

    Multiple examples (also for ROCs) are provided in the `puma` examples.

??? warning "Solution"

    ```py
    import numpy as np
    from puma import Roc, RocPlot
    from ftag.utils import calculate_rejection, get_discriminant
    from ftag import Flavours
    from ftag.hdf5 import H5Reader

    ttbar_filepath = "/eos/user/u/umamibot/tutorials/ttbar.h5"

    # load the jets dataset from the h5 file
    reader = H5Reader(ttbar_filepath, batch_size=100)
    data = reader.load()
    jets = data["jets"]

    # Calculate the discriminant for both taggers
    discs_dips = get_discriminant(
        jets=jets,
        tagger="dipsLoose20220314v2",
        signal=Flavours["bjets"],
        flavours=Flavours.by_category("single-btag"),
        fraction_values={
            "fc": 0.018,
            "fu": 0.982,
            "ftau": 0,
        },
    )
    discs_rnnip = get_discriminant(
        jets=jets,
        tagger="rnnip",
        signal=Flavours["bjets"],
        flavours=Flavours.by_category("single-btag"),
        fraction_values={
            "fc": 0.018,
            "fu": 0.982,
            "ftau": 0,
        },
    )

    # defining target efficiency
    sig_eff = np.linspace(0.49, 1, 20)

    # defining boolean arrays to select the different flavour classes
    is_light = jets["HadronConeExclTruthLabelID"] == 0
    is_c = jets["HadronConeExclTruthLabelID"] == 4
    is_b = jets["HadronConeExclTruthLabelID"] == 5

    n_jets_light = sum(is_light)
    n_jets_c = sum(is_c)

    rnnip_ujets_rej = calculate_rejection(discs_rnnip[is_b], discs_rnnip[is_light], sig_eff)
    dips_ujets_rej = calculate_rejection(discs_dips[is_b], discs_dips[is_light], sig_eff)
    ```

#### Task 2.2: Plot the Rejections as a Function of the $b$-jets Efficiency

1.  Plot the light-flavour jets rejection as a function of the $b$-jets efficiency. Use
    `n_ratio_panels=1` to also get the ratio of the two rejection curves.

??? info "Hint 1: How do I Initialise a ROC Curve Plot?"

    Plotting ROC curves with `puma` is similar to plotting histograms. The main difference
    is that you are using the `puma.RocPlot` and `puma.Roc` classes. Search the
    [puma docs](https://umami-hep.github.io/puma/examples/rocs/) for "roc" to have a look at an example and the API reference.

??? info "Hint 2: I Initialised the Plot and Added the ROC Curves - Is there anything else to do?"

    For ROC curves you also have to define the class which is drawn in the ratio panel.
    The method you need to use here is `RocPlot.set_ratio_class()`.

??? warning "Solution"

    Add this part under the calculation of the rejection values.

    ```py
    # Define the ROC class instances
    rnnip_ujets_roc = Roc(
        sig_eff=sig_eff,
        bkg_rej=rnnip_ujets_rej,
        n_test=n_jets_light,
        rej_class=Flavours["ujets"],
        signal_class="bjets",
        label="RNNIP",
    )
    dips_ujets_roc = Roc(
        sig_eff=sig_eff,
        bkg_rej=dips_ujets_rej,
        n_test=n_jets_light,
        rej_class=Flavours["ujets"],
        signal_class="bjets",
        label="DIPS r22",
    )

    # Set up a plot
    plot_roc = RocPlot(
        n_ratio_panels=1,
        ylabel="Background rejection",
        xlabel="$b$-jet efficiency",
        atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets \ndummy sample, $f_{c}=0.018$",
        figsize=(6.5, 6),
        y_scale=1.4,
    )

    # Add the ROC objects to the plot
    plot_roc.add_roc(roc_curve=rnnip_ujets_roc, reference=True)
    plot_roc.add_roc(roc_curve=dips_ujets_roc)

    # Set the ratio class for the ratio panels
    plot_roc.set_ratio_class(1, Flavours["ujets"])

    # Draw and save the plot
    plot_roc.draw()
    plot_roc.savefig("tutorial_roc.png", transparent=False)
    ```

#### Task 2.3: Add the $c$-rejection to Your Plot

1.  Repeat the calculation of the rejection for $c$-jets
2.  Add the corresponding ROC curves to the plot. Don't forget to increase `n_ratio_panels`
    of your `puma.RocPlot`.

??? warning "Solution"

    ```py
    # Calculate the c-jet rejection values
    rnnip_cjets_rej = calculate_rejection(discs_rnnip[is_b], discs_rnnip[is_c], sig_eff)
    dips_cjets_rej = calculate_rejection(discs_dips[is_b], discs_dips[is_c], sig_eff)

    # Define the c-jet ROC objects
    rnnip_cjets_roc = Roc(
        sig_eff=sig_eff,
        bkg_rej=rnnip_cjets_rej,
        n_test=n_jets_c,
        rej_class=Flavours["cjets"],
        signal_class="bjets",
        label="RNNIP",
    )
    dips_cjets_roc = Roc(
        sig_eff=sig_eff,
        bkg_rej=dips_cjets_rej,
        n_test=n_jets_c,
        rej_class=Flavours["cjets"],
        signal_class="bjets",
        label="DIPS r22",
    )

    # Create a new ROC plot with one ratio panel per rejection class
    plot_roc = RocPlot(
        n_ratio_panels=2,
        ylabel="Background rejection",
        xlabel="$b$-jet efficiency",
        atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets \ndummy sample, $f_{c}=0.018$",
        figsize=(6.5, 6),
        y_scale=1.4,
    )

    # Add both the light-jet and c-jet ROC objects
    plot_roc.add_roc(roc_curve=rnnip_ujets_roc, reference=True)
    plot_roc.add_roc(roc_curve=dips_ujets_roc)
    plot_roc.add_roc(roc_curve=rnnip_cjets_roc, reference=True)
    plot_roc.add_roc(roc_curve=dips_cjets_roc)

    # Set one ratio panel for light jets and one for c-jets
    plot_roc.set_ratio_class(1, Flavours["ujets"])
    plot_roc.set_ratio_class(2, Flavours["cjets"])

    plot_roc.draw()
    plot_roc.savefig("tutorial_roc_with_cjets.png", transparent=False)
    ```

### Task 3: $p_\text{T}$ vs. Efficiency

In this task, you will plot both the $b$-jets efficiency and the light-flavour jets rejection for
specific bins of $p_\text{T}$.

#### Task 3.1: Calculate the discriminant values

Just like you did in Task 2.1, calculate the discriminant scores for _RNNIP_ and _DIPS_. You can
reuse the code from task 2.1. If you are putting everything in one python script you can just reuse
the values that are already calculated.

#### Task 3.2: Create a $p_\text{T}$ vs. $b$-efficiency Plot

For a fixed inclusive $b$-efficiency, you plot the $b$-efficiency for different bins of $p_\text{T}$.

??? info "Hint 1: I'm not sure which type of Plot I need"

    The plot type you are looking for is `VarVsEff` and `VarVsEffPlot`.

??? warning "Solution"

    ```py
    import numpy as np
    from puma import VarVsEff, VarVsEffPlot
    from ftag.utils import get_discriminant
    from ftag import Flavours
    from ftag.hdf5 import H5Reader

    ttbar_filepath = "/eos/user/u/umamibot/tutorials/ttbar.h5"

    # load the jets dataset from the h5 file
    reader = H5Reader(ttbar_filepath, batch_size=100)
    data = reader.load()
    jets = data["jets"]

    # Calculate the discriminant for both taggers
    discs_dips = get_discriminant(
        jets=jets,
        tagger="dipsLoose20220314v2",
        signal=Flavours["bjets"],
        flavours=Flavours.by_category("single-btag"),
        fraction_values={
            "fc": 0.018,
            "fu": 0.982,
            "ftau": 0,
        },
    )
    discs_rnnip = get_discriminant(
        jets=jets,
        tagger="rnnip",
        signal=Flavours["bjets"],
        flavours=Flavours.by_category("single-btag"),
        fraction_values={
            "fc": 0.018,
            "fu": 0.982,
            "ftau": 0,
        },
    )

    # Getting jet pt in GeV (in the files, they are stored in MeV)
    pt = jets["pt"] / 1e3

    # defining target efficiency
    sig_eff = np.linspace(0.49, 1, 20)

    # defining boolean arrays to select the different flavour classes
    is_light = jets["HadronConeExclTruthLabelID"] == 0
    is_c = jets["HadronConeExclTruthLabelID"] == 4
    is_b = jets["HadronConeExclTruthLabelID"] == 5

    # here the plotting starts

    # define the curves
    rnnip_light = VarVsEff(
        x_var_sig=pt[is_b],
        disc_sig=discs_rnnip[is_b],
        x_var_bkg=pt[is_light],
        disc_bkg=discs_rnnip[is_light],
        bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
        working_point=0.7,
        disc_cut=None,
        flat_per_bin=False,
        label="RNNIP",
    )
    dips_light = VarVsEff(
        x_var_sig=pt[is_b],
        disc_sig=discs_dips[is_b],
        x_var_bkg=pt[is_light],
        disc_bkg=discs_dips[is_light],
        bins=[20, 30, 40, 60, 85, 110, 140, 175, 250],
        working_point=0.7,
        disc_cut=None,
        flat_per_bin=False,
        label="DIPS",
    )

    # You can choose between different modes: "sig_eff", "bkg_eff", "sig_rej", "bkg_rej"
    plot_sig_eff = VarVsEffPlot(
        mode="sig_eff",
        ylabel="$b$-jets efficiency",
        xlabel=r"$p_{T}$ [GeV]",
        logy=False,
        atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow jets, \n$t\\bar{t}$ sample, $f_{c}=0.018$",
        figsize=(6, 4.5),
        n_ratio_panels=1,
    )
    plot_sig_eff.add(rnnip_light, reference=True)
    plot_sig_eff.add(dips_light)

    plot_sig_eff.atlas_second_tag += "\n" + r"Inclusive $\epsilon_b=70\%$"

    # If you want to inverse the discriminant cut you can enable it via
    # plot_sig_eff.set_inverse_cut()
    plot_sig_eff.draw()
    # Drawing a hline indicating inclusive efficiency
    plot_sig_eff.draw_hline(0.7)
    plot_sig_eff.savefig("tutorial_pt_b_eff.png", transparent=False)
    ```

#### Task 3.3: Create a $p_\text{T}$ vs. light-flavour jets rejection plot

??? warning "Solution"

    ```py
    # reuse the VarVsEff objects that were defined for the previous exercise
    plot_bkg_rej = VarVsEffPlot(
        mode="bkg_rej",
        ylabel="Light-flavour jets rejection",
        xlabel=r"$p_{T}$ [GeV]",
        logy=False,
        atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow jets \n$t\\bar{t}$ sample, $f_{c}=0.018$",
        figsize=(6, 4.5),
        n_ratio_panels=1,
    )
    plot_bkg_rej.atlas_second_tag += "\n" + r"Inclusive $\epsilon_b=70\%$"
    plot_bkg_rej.add(rnnip_light, reference=True)
    plot_bkg_rej.add(dips_light)

    plot_bkg_rej.draw()
    plot_bkg_rej.savefig("tutorial_pt_light_rej.png")
    ```

### Task 4: Scaling Up and Using the High-Level API

The previous tasks intentionally used the low-level plotting objects, because they make the
ingredients of each plot explicit. Once you understand these ingredients, two patterns are useful in
day-to-day plotting:

- stream large files in batches when a full `h5py` array would use too much memory;
- use the high-level API to make standard tagger-comparison plots with less boilerplate.

#### Task 4.1: Fill a Histogram from an h5 File in Batches

In Task 1 you loaded the full `jets` and `tracks_loose` datasets into memory. This is convenient for
small files, but many production files are too large for this. The `ftag.hdf5.H5Reader` can stream
only the variables you need, and `puma.Histogram.update()` can fill the same histogram batch by
batch.

**Create the same $p_\text{T}$ histogram as in Task 1.2, but fill it batch by batch.**

??? info "Hint: Which methods are needed?"

    Use `H5Reader.stream()` to loop over batches and `Histogram.update()` after the first batch.

??? warning "Solution"

    ```py
    import numpy as np
    from ftag import Flavours
    from ftag.hdf5 import H5Reader
    from puma import Histogram, HistogramPlot

    ttbar_filepath = "/eos/user/u/umamibot/tutorials/ttbar.h5"

    reader = H5Reader(ttbar_filepath, batch_size=100_000, shuffle=False)
    variables = {"jets": ["pt", "HadronConeExclTruthLabelID"]}
    bins = np.linspace(0, 250, 50)

    histograms = {"ujets": None, "cjets": None, "bjets": None}

    for batch in reader.stream(variables=variables):
        jets = batch["jets"]
        masks = {
            "ujets": jets["HadronConeExclTruthLabelID"] == 0,
            "cjets": jets["HadronConeExclTruthLabelID"] == 4,
            "bjets": jets["HadronConeExclTruthLabelID"] == 5,
        }

        for flavour, mask in masks.items():
            values = jets["pt"][mask] / 1000
            if histograms[flavour] is None:
                histograms[flavour] = Histogram(
                    values,
                    flavour=Flavours[flavour],
                    bins=bins,
                )
            else:
                histograms[flavour].update(values)

    pt_plot = HistogramPlot(
        xlabel=r"$p_\text{T}$ [GeV]",
        ylabel="Normalised number of jets",
    )
    for histogram in histograms.values():
        pt_plot.add(histogram)

    pt_plot.draw()
    pt_plot.savefig("tutorial_histogram_pT_streamed.png")
    ```

#### Task 4.2: Use the High-Level API for Standard Tagger Comparisons

The low-level `Histogram`, `Roc`, and `VarVsEff` objects are useful when you need full control.
For repeated tagger comparisons, the high-level API can load taggers once and produce common plots
with short method calls.

**Use `Results` and `Tagger` to make discriminant and ROC comparison plots for RNNIP and DIPS.**

??? info "Hint: Where can I find more examples?"

    The high-level API is documented in the [high-level API docs](../hlapi/index.md). The full
    example script is in `examples/high_level_plots.py`.

??? warning "Solution"

    ```py
    from pathlib import Path

    from puma.hlplots import Results, Tagger

    ttbar_filepath = "/eos/user/u/umamibot/tutorials/ttbar.h5"

    dips = Tagger(
        name="dipsLoose20220314v2",
        output_flavours=["ujets", "cjets", "bjets"],
        label="DIPS r22",
        fxs={"fc": 0.018},
        colour="#AA3377",
    )
    rnnip = Tagger(
        name="rnnip",
        output_flavours=["ujets", "cjets", "bjets"],
        label="RNNIP",
        fxs={"fc": 0.018},
        colour="#4477AA",
        reference=True,
    )

    results = Results(
        signal="bjets",
        sample="ttbar",
        output_dir=Path("tutorial_high_level_plots"),
        extension="png",
    )
    results.load_taggers_from_file([dips, rnnip], ttbar_filepath)
    results.atlas_second_tag = "$\\sqrt{s}=13$ TeV, PFlow jets\n$t\\bar{t}$ sample"

    results.plot_discs(bins=40, bins_range=(-10, 10))
    results.plot_rocs()
    ```

    The high-level API writes plots and cached intermediate `Histogram`/`Roc` objects into the
    output directory. If you only want a fast test while developing a script, pass `num_jets=50_000`
    to `load_taggers_from_file()`.

### Task 5: Other Supported Plot Types

The main flavour-tagging performance plots are histograms, ROC curves and efficiency/rejection
profiles. `puma` also supports several more general plot types that are useful for quick checks,
summary figures, and classifier diagnostics.

#### Task 5.1: Plot the Flavour Composition with `PiePlot`

Pie charts are not usually the right tool for detailed performance comparisons, but they are useful
for quick sample-composition checks.

**Plot the fraction of light-flavour jets, $c$-jets and $b$-jets in the tutorial file.**

??? warning "Solution"

    ```py
    import h5py
    from ftag import Flavours
    from puma.pie import PiePlot

    ttbar_filepath = "/eos/user/u/umamibot/tutorials/ttbar.h5"

    with h5py.File(ttbar_filepath, "r") as h5file:
        labels = h5file["jets"]["HadronConeExclTruthLabelID"][:]

    counts = [
        (labels == 0).sum(),
        (labels == 4).sum(),
        (labels == 5).sum(),
    ]
    plot = PiePlot(
        wedge_sizes=counts,
        labels=["Light-flavour jets", "$c$-jets", "$b$-jets"],
        colours=[
            Flavours["ujets"].colour,
            Flavours["cjets"].colour,
            Flavours["bjets"].colour,
        ],
        atlas_second_tag="$t\\bar{t}$ sample",
        figsize=(5.5, 3.5),
    )
    plot.savefig("tutorial_flavour_composition.png")
    ```

#### Task 5.2: Plot a Confusion Matrix with `MatshowPlot`

For multi-class taggers, a confusion matrix is a compact way to see which class has the largest
output probability for each true jet flavour. Here we use the DIPS output probabilities directly.

**Create a row-normalised confusion matrix for DIPS.**

??? warning "Solution"

    ```py
    import h5py
    import numpy as np
    from puma.matshow import MatshowPlot
    from puma.utils.confusion_matrix import confusion_matrix

    ttbar_filepath = "/eos/user/u/umamibot/tutorials/ttbar.h5"

    with h5py.File(ttbar_filepath, "r") as h5file:
        jets = h5file["jets"][:]

    is_used = np.isin(jets["HadronConeExclTruthLabelID"], [0, 4, 5])
    true_labels = np.select(
        [
            jets["HadronConeExclTruthLabelID"][is_used] == 0,
            jets["HadronConeExclTruthLabelID"][is_used] == 4,
            jets["HadronConeExclTruthLabelID"][is_used] == 5,
        ],
        [0, 1, 2],
    )
    probabilities = np.column_stack(
        [
            jets["dipsLoose20220314v2_pu"][is_used],
            jets["dipsLoose20220314v2_pc"][is_used],
            jets["dipsLoose20220314v2_pb"][is_used],
        ]
    )
    predicted_labels = np.argmax(probabilities, axis=1)

    matrix = confusion_matrix(true_labels, predicted_labels, normalize="rownorm")

    plot = MatshowPlot(
        x_ticklabels=["Pred. light", "Pred. $c$", "Pred. $b$"],
        y_ticklabels=["True light", "True $c$", "True $b$"],
        show_entries=True,
        show_percentage=True,
        cbar_label="Fraction of true class",
        atlas_second_tag="DIPS r22",
    )
    plot.draw(matrix)
    plot.savefig("tutorial_dips_confusion_matrix.png")
    ```

#### Task 5.3: Plot Integrated Efficiencies

Integrated-efficiency plots show the signal and background efficiencies as a function of the
discriminant cut. They are a useful diagnostic when you want to inspect where a working point sits
on the discriminant distribution.

**Create integrated-efficiency curves for RNNIP and DIPS.**

??? warning "Solution"

    ```py
    from ftag import Flavours
    from ftag.hdf5 import H5Reader
    from ftag.utils import get_discriminant
    from puma import IntegratedEfficiency, IntegratedEfficiencyPlot

    ttbar_filepath = "/eos/user/u/umamibot/tutorials/ttbar.h5"

    reader = H5Reader(ttbar_filepath, batch_size=100)
    jets = reader.load()["jets"]

    discs_dips = get_discriminant(
        jets=jets,
        tagger="dipsLoose20220314v2",
        signal=Flavours["bjets"],
        flavours=Flavours.by_category("single-btag"),
        fraction_values={
            "fc": 0.018,
            "fu": 0.982,
            "ftau": 0,
        },
    )
    discs_rnnip = get_discriminant(
        jets=jets,
        tagger="rnnip",
        signal=Flavours["bjets"],
        flavours=Flavours.by_category("single-btag"),
        fraction_values={
            "fc": 0.018,
            "fu": 0.982,
            "ftau": 0,
        },
    )

    is_light = jets["HadronConeExclTruthLabelID"] == 0
    is_c = jets["HadronConeExclTruthLabelID"] == 4
    is_b = jets["HadronConeExclTruthLabelID"] == 5

    plot = IntegratedEfficiencyPlot(
        ylabel="Integrated efficiency",
        xlabel="Discriminant",
        atlas_second_tag="$\\sqrt{s}=13$ TeV, PFlow jets\n$t\\bar{t}$ sample",
        figsize=(6.5, 6),
    )
    plot.add(
        IntegratedEfficiency(
            discs_rnnip[is_b],
            discs_rnnip[is_light],
            flavour="ujets",
            tagger="RNNIP",
        )
    )
    plot.add(
        IntegratedEfficiency(
            discs_rnnip[is_b],
            discs_rnnip[is_c],
            flavour="cjets",
            tagger="RNNIP",
        )
    )
    plot.add(
        IntegratedEfficiency(
            discs_dips[is_b],
            discs_dips[is_light],
            flavour="ujets",
            tagger="DIPS r22",
        )
    )
    plot.add(
        IntegratedEfficiency(
            discs_dips[is_b],
            discs_dips[is_c],
            flavour="cjets",
            tagger="DIPS r22",
        )
    )

    plot.draw()
    plot.savefig("tutorial_integrated_efficiency.png", transparent=False)
    ```

## Bonus tasks

### Run over a Run-3 MC Sample and Compare the Pileup Distributions

This task will extend over the simple histogram plotting you already encountered in Task 1. You are
asked to compare distributions from two different files: the Run-2 MC for the Z' sample and the
Run-3 MC for the Z' sample.

For this task, you will:

1. Download the Z' sample for the Run-3 MC `zpext_run3.h5` from `/eos/user/u/umamibot/tutorials/`
   or from `https://umami-ci-provider.web.cern.ch/puma/zpext_run3.h5`.
2. Write a plotting script to compare the `averageInteractionsPerCrossing` between the two samples.

??? warning "Solution"

    Copy the Run-3 MC file (assuming you work on lxplus):

    ```bash
    cp /eos/user/u/umamibot/tutorials/zpext_run3.h5 </path/to/tutorial/data/>
    ```

    If you don't work on lxplus, download it via

    ```bash
    wget https://umami-ci-provider.web.cern.ch/puma/zpext_run3.h5
    ```

    You should provide a path for the dummy `</path/to/tutorial/data/>` in the command above and in the
    python example below:


    ```python
    import numpy as np
    from puma import Histogram, HistogramPlot
    from ftag.hdf5 import H5Reader

    # load the "jets" datasets from the h5 files
    filepath_run2 = "zpext.h5"
    reader = H5Reader(filepath_run2, batch_size=100)
    data = reader.load()
    jets_run2 = data["jets"]
    
    filepath_run3 = "zpext_run3.h5"
    reader = H5Reader(filepath_run3, batch_size=100)
    data = reader.load()
    jets_run3 = data["jets"]

    variable = "averageInteractionsPerCrossing"
    run_2 = Histogram(jets_run2[variable], label="Run 2 MC", bins=np.linspace(10, 70, 60), norm=True)
    run_3 = Histogram(jets_run3[variable], label="Run 3 MC", bins=np.linspace(10, 70, 60), norm=True)

    # Initialise histogram plot
    plot_histo = HistogramPlot(
        ylabel="Number of events",
        xlabel=r"average interactions per crossing $\langle\mu\rangle$ [a.u.]",
        logy=False,
        atlas_first_tag="Simulation Internal",
        atlas_second_tag="Example for a comparison plot",
        figsize=(6, 5),
        n_ratio_panels=1,
    )

    # Add histograms and plot
    plot_histo.add(run_2, reference=True)
    plot_histo.add(run_3)
    plot_histo.draw()

    plot_histo.savefig("histogram_pileup.png", transparent=False)

    ```

### Compare the "Flipped Taggers" to the Regular Flavour Tagging Algorithms

This task further extends over the simple histogram plotting you already encountered in Task 1. You
are asked to compare distributions from a regular flavour tagging algorithm and a so-called "flipped
tagger", which is a modified version of the flavour tagging algorithm used for light-jet mistag
calibration. For this version, the sign of $d_0$/$z_0$ signed impact parameter is flipped, resulting
in a selection of jets with "negative lifetime".

Consequently, the flipped tagger's $b$-tagging efficiency is reduced while its light-jet mistag rate
is left unchanged.

For this task, you will:

1. Write a plotting script to compare the scores $p_b$, $p_c$, and $p_u$, for the RNNIP tagger to
   the flipped version. You should produce three plots, one for each score (such as $p_b$), which
   show the distributions of the RNNIP tagger and the flipped RNNIP tagger overlaid for the three
   different jet flavours $b$-jets, $c$-jets and light-flavour jets.
2. Next, extend the script to compare also the flavour tagging discriminant based on the flipped
   tagger and the regular RNNIP tagger. You should produce one plot which compares the distributions
   of the RNNIP tagger and the flipped RNNIP tagger overlaid for the three different jet flavours
   $b$-jets, $c$-jets and light-flavour jets.

??? info "Hint: Names of the RNNIP tagger and the flipped tagger scores and the corresponding $b$-tagging discriminant"

    The names of the RNNIP tagger scores are

    - `rnnip_pu`
    - `rnnip_pc`
    - `rnnip_pb`

    The names of the flipped version are

    - `rnnipflip_pu`
    - `rnnipflip_pc`
    - `rnnipflip_pb`

    An example how the discriminant can be calculated is provided in [one of the puma example scripts](https://github.com/umami-hep/puma/blob/main/examples/plot_discriminant_scores.py).

??? warning "Solution"

    You should provide the path for the dummy `</path/to/tutorial/data/>` in the python example below:


    ```python
    import numpy as np
    import pandas as pd
    from puma import Histogram, HistogramPlot
    from ftag.utils import get_discriminant
    from ftag import Flavours
    from ftag.hdf5 import H5Reader

    # load the "jets" dataset from the h5 file
    filepath = "/path/to/tutorial/data/ttbar.h5"
    reader = H5Reader(filepath, batch_size=100)
    data = reader.load()
    jets = data["jets"]
    jets_df = pd.DataFrame(jets)


    # defining boolean arrays to select the different flavour classes
    is_light = jets["HadronConeExclTruthLabelID"] == 0
    is_c = jets["HadronConeExclTruthLabelID"] == 4
    is_b = jets["HadronConeExclTruthLabelID"] == 5


    # Calculate discriminant scores for RNNIP and flipped tagger, and add them to the dataframe
    
    jets_df["disc_rnnip"] = get_discriminant(
        jets=jets,
        tagger="rnnip",
        signal=Flavours["bjets"],
        flavours=Flavours.by_category("single-btag"),
        fraction_values={
            "fc": 0.07,
            "fu": 0.93,
            "ftau": 0,
        },
    )

    jets_df["disc_rnnipflip"] = get_discriminant(
        jets=jets,
        tagger="rnnipflip",
        signal=Flavours["bjets"],
        flavours=Flavours.by_category("single-btag"),
        fraction_values={
            "fc": 0.07,
            "fu": 0.93,
            "ftau": 0,
        },
    )

    variables = [
        ('rnnip_pu', 'rnnipflip_pu'),
        ('rnnip_pc', 'rnnipflip_pc'),
        ('rnnip_pb', 'rnnipflip_pb'),
        ('disc_rnnip', 'disc_rnnipflip'),
    ]

    axis_labels = {
        'rnnip_pu': 'RNNIP $p_\\mathrm{light}$',
        'rnnip_pc': 'RNNIP $p_{c}$',
        'rnnip_pb': 'RNNIP $p_{b}$',
        'disc_rnnip': 'RNNIP $b$-tagging discriminant',
    }

    # plot score and discriminant distributions
    for v in variables:
        rnnip_light = Histogram(
            jets_df[is_light][v[0]],
            flavour=Flavours["ujets"],
            label="RNNIP",
            ratio_group="ujet",
            bins=np.linspace(-10, 10, 40) if v[0] == "disc_rnnip" else np.linspace(0, 1, 20),
            norm=False,
        )
        rnnip_c = Histogram(
            jets_df[is_c][v[0]],
            flavour=Flavours["cjets"],
            label="RNNIP",
            ratio_group="cjet",
            bins=np.linspace(-10, 10, 40) if v[0] == "disc_rnnip" else np.linspace(0, 1, 20),
            norm=False,
        )
        rnnip_b = Histogram(
            jets_df[is_b][v[0]],
            flavour=Flavours["bjets"],
            label="RNNIP",
            ratio_group="bjet",
            bins=np.linspace(-10, 10, 40) if v[0] == "disc_rnnip" else np.linspace(0, 1, 20),
            norm=False,
        )

        rnnip_light_flip = Histogram(
            jets_df[is_light][v[1]],
            linestyle="dashed",
            flavour=Flavours["ujets"],
            label="RNNIP (flip)",
            ratio_group="ujet",
            bins=np.linspace(-10, 10, 40) if v[0] == "disc_rnnip" else np.linspace(0, 1, 20),
            norm=False,
        )
        rnnip_c_flip = Histogram(
            jets_df[is_c][v[1]],
            linestyle="dashed",
            flavour=Flavours["cjets"],
            label="RNNIP (flip)",
            ratio_group="cjet",
            bins=np.linspace(-10, 10, 40) if v[0] == "disc_rnnip" else np.linspace(0, 1, 20),
            norm=False,
        )
        rnnip_b_flip = Histogram(
            jets_df[is_b][v[1]],
            linestyle="dashed",
            flavour=Flavours["bjets"],
            label="RNNIP (flip)",
            ratio_group="bjet",
            bins=np.linspace(-10, 10, 40) if v[0] == "disc_rnnip" else np.linspace(0, 1, 20),
            norm=False,
        )

        # Initialise histogram plot
        plot_histo = HistogramPlot(
            ylabel="Number of events",
            xlabel=axis_labels[v[0]],
            logy=True,
            atlas_first_tag="Simulation Internal",
            figsize=(6, 5),
            n_ratio_panels=1,
        )

        # Add histograms and plot
        plot_histo.add(rnnip_light, reference=True)
        plot_histo.add(rnnip_c, reference=True)
        plot_histo.add(rnnip_b, reference=True)
        plot_histo.add(rnnip_light_flip)
        plot_histo.add(rnnip_c_flip)
        plot_histo.add(rnnip_b_flip)
        plot_histo.draw()

        plot_histo.savefig(f"histogram_flip_{v[0]}.png", transparent=False)
    ```
