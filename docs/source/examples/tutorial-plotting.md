# Umami plotting interface tutorial

## Introduction

In this tutorial you will learn how to use `puma`, the _Plotting UMami Api_.
The idea behind `puma` is to provide a plotting API that is easy to use but at the same
time highly configurable.
This means that the user has full control over the plot while things like uncertainties,
labels, ratio panels can be added easily.

You can find the `puma` documentation [here](https://umami-hep.github.io/puma/).

`puma` is based on `matplotlib` and helps you to produce most of the types of
plots that are commonly used in flavour tagging like the ones shown below:

|                                     ROC curves                                      |                                            Histogram plots                                             |                                    Variable vs efficiency                                    |
| :---------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: |
| <img src=https://github.com/umami-hep/puma/raw/examples-material/roc.png width=200> | <img src=https://github.com/umami-hep/puma/raw/examples-material/histogram_discriminant.png width=220> | <img src=https://github.com/umami-hep/puma/raw/examples-material/pt_light_rej.png width=220> |

In this tutorial you will learn how to:

0. A small introduction to the different metrics used in flavour tagging.
1. Plot histograms with `puma`, where you will produce plots of both jet- and track-variables.
2. Plot ROC curves of two taggers, with ratio panels that compare the curves.
3. Plot the efficiency/rejection of a tagger as a function of $p_T$.
4. Plot the input variables of the files using Umami.

The tutorial is meant to be followed in a self-guided manner. You will be prompted to
do certain tasks by telling you what the desired outcome will be, without telling you
how to do it. Using the documentation of the training-dataset-dumper, you can find out
how to achieve your goal. In case you are stuck, you can click on the "hint" toggle
box to get a hint. If you tried for more than 10 min at a problem, feel free to toggle
also the solution with a worked example.

## Setting Up Everything You Need

For the tutorial, you will need both `puma` and (for the Task 4) `umami`. While `puma`
can be easily installed via the command

```bash
pip install puma-hep
```

installing `umami` is not so easy. Due to this fact, we encourage you to use docker or
singularity images from `umami` (`puma` is already installed in there). An explanation
how to run singularity images (also on lxplus) can be found [here](https://umami-docs.web.cern.ch/setup/installation/#docker-container).
If you have access to the `cvmfs`, you can easily run this from there without building the
container locally again (also explained [here](https://umami-docs.web.cern.ch/setup/installation/#docker-container)).

For this tutorial, two `.h5` files were prepared, one $t\bar{t}$ and one $Z'$ file. You
can download them from the `eos` directly. The path to the directory where the files are
stored is `/eos/user/u/umamibot/tutorials/`.

## Tutorial tasks

The tasks are divided in several sub-tasks. You don't have to do all of them in case you
are more interested in the other tasks.
However, the sub-tasks depend on each other, so you should finish a subtask before
proceeding to the next one (also the solutions assume that you already have the previous
subtasks completed).

### Task 0: Flavour tagging metrics

To get started with the plotting, we will first have a quick look into different metrics
used in flavour tagging to evaluate the performance.

#### Task 0.1: Generating dummy data

The typical output of our ML-based taggers, like DIPS or DL1d, are 3 scores indicating
the probabilities of being a light-flavour jet, a $c$-jet and a $b$-jet. For example the scores
`[0.1, 0.3, 0.6]` would indicate that we have most probably a $b$-jet while `[0.7, 0.2, 0.1]`
would indicate a light-flavour jet.

Even though we always have MC simulation, it is sometimes useful to have dummy data,
for instance if we want to test certain plotting features etc. (and to understand the
actual underlying distributions).

Now, it is up to you. **Generate a dummy multi-class output of a neural network.**

??? info "Hint: Where can I find such a function?"
    You can have a look at the [puma documentation](https://umami-hep.github.io/puma/main/index.html)
    and search in the API reference.

??? info "Hint: Which exact function?"
    the [`puma.utils.generate`](https://umami-hep.github.io/puma/main/autoapi/puma/utils/generate/index.html)
    module contains the desired functions

??? warning "Solution"

    ```py
    from puma.utils.generate import get_dummy_multiclass_scores

    # Generating 10_000 jets
    dummy_jets, dummy_labels = get_dummy_multiclass_scores()
    print(dummy_jets)
    print(dummy_labels)
    ```

#### Task 0.2: Defining Working points - Likelihood ratio

Since we are not (yet) able to calibrate the entire spectrum of the different multi-class outputs,
we need to define so-called working points (or operating points).
In the case of ATLAS, we have four different b-tagging working points (WPs) which are defined
covering various needs of the physics analyses.
The efficiency of a specific flavour j (b, c or light) is defined as

$$
\varepsilon^j = \frac{N_\text{pass}^j(\mathcal{D}>T_f)}{N_\text{total}^j},
$$

where $N_\text{pass}^j(\mathcal{D}>T_f)$ are the number of jets of flavour $j$ passing the cut $T_f$ on the tagger discriminant $\mathcal{D}$ and $N_\text{total}^j$ are all jets of flavour $j$ before the cut.

The final output score is calculated from the multi-class output and results for the $b$-tagging discriminant into the log-likelihood

$$
\mathcal{D}_\text{b}(f_c) = \log \left(  \frac{p_b}{f_c\cdot p_c+(1-f_c)\cdot p_\text{l}} \right),
$$

with $p_b$, $p_c$ and $p_l$ being the probabilities for the jet to be a $b$-jet, $c$-jet or light-flavour jet, respectively.
The $c$-jet fraction $f_c$ allows to tune how much emphasis is given to the $c$-jet or to the light-flavour performance.
While the $c$-jet rejection increases as a function of $f_c$, the light-flavour jet rejection decreases. This parameter has
to be tuned separately for each tagger and depends on the needs of the physics analyses.

The advantage of the multi-class output is that this tuning is possible after the training and the $c$-jet fraction in the training sample does not have to be adapted. Another advantage of the multi-class output is that one can by changing the log-likelihood to

$$
\mathcal{D}_\text{c}(f_b) = \log \left(  \frac{p_c}{f_b\cdot p_b+(1-f_b)\cdot p_l} \right),
$$

perform $c$-tagging without the need of retraining the tagger. Here $f_b$ is now the $b$-jet fraction.

**Define a function which calculates the log-likelihood, when giving it the 3 scores and the $f_c$ value as input.**

??? info "Hint 1"
    You can either use a python function `def` or a `lambda` function

??? info "Hint 2"
    In the puma examples you might find something similar.

??? warning "Solution"
    You can find the solution in the [histogram example](https://umami-hep.github.io/puma/main/examples/rocs.html)
    of the puma documentation.

    ```py
    import numpy as np
    def disc_fct(arr: np.ndarray, f_c: float = 0.018) -> np.ndarray:
        """Tagger discriminant

        Parameters
        ----------
        arr : numpy.ndarray
            array with with shape (, 3)
        f_c : float, optional
            f_c value in the discriminant (weight for c-jets rejection)

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

??? info "Hint: Where to get the labels from?"
    the `labels` from task 0.1 have the same values as the `HadronConeExclTruthLabelID`
    described in the [FTAG algo docs](https://ftag.docs.cern.ch/algorithms/labelling/jet_labels/#delta-r-matching-default-scheme-in-ftag).

??? info "Hint: Which function to use?"
    You can have a look at the `percentile` function from `numpy`.
    Be aware from which site we need to integrate!
    And the `apply_along_axis` function to evaluate an entire array.

??? warning "Solution"
    ```py
    import numpy as np

    bjets = dummy_labels == 5
    scores = np.apply_along_axis(disc_fct, axis=1, arr=dummy_jets)
    target_eff = 0.7
    cutvalue = np.percentile(scores[bjets], 100.0 * (1.0 - target_eff))
    print("cut value for 70% b-jet efficiency:", cutvalue)
    ```
    You can also have a look at the [metrics function in puma](https://github.com/umami-hep/puma/blob/main/puma/metrics.py#L41-L42),
    where this code is also being used.

#### Task 0.3: Performance metrics - efficiency and rejection

To quantify the performance of a tagger at a given working point, the background
rejection is a good measure. The rejection is simply the inverse of the efficiency
$\frac{1}{\varepsilon^j}$.

**Calculate the light-flavour jet and $c$-jet rejection for the 70% working point from
task 0.2.**

??? warning "Solution"
    ```py
    ljets = dummy_labels == 0
    cjets = dummy_labels == 4

    ljets_eff = ((scores > cutvalue) & ljets).sum() / ljets.sum()
    print("light-flavour jets efficiency:", ljets_eff)
    print("light-flavour jets rejection:", 1 / ljets_eff)

    cjets_eff = ((scores > cutvalue) & cjets).sum() / cjets.sum()
    print("c-flavour jets efficiency:", cjets_eff)
    print("c-flavour jets rejection:", 1 / cjets_eff)
    ```

    Alternatively, all this functionality is also provided by `puma.metrics`.
    In that case this would simplify to
    ```py
    from puma.metrics import calc_rej
    ljets = dummy_labels == 0
    cjets = dummy_labels == 4
    rej = calc_rej(scores[bjets], scores[ljets], target_eff=0.7)
    print("light-flavour jets rejection:", rej)
    ```

Starting from these metrics, we can plot for instance:

- **ROC curves**: which show the background rejection as function of the $b$-jet efficency
- **Efficiency vs $p_T$**: where one fixes a working point and calculates the background rejection in bins of $p_T$

### Task 1: Histogram plots

#### Task 1.1: Loading the h5 file

Before starting with the different plotting exercises, you have to load the h5 file
that was prepared for this tutorial.
The expected outcome of this is that you have access to the jet variables as well as
to the track variables. You can put the jet variables in a `pandas.DataFrame` in case
you feel more comfortable with that, but this will not be possible for the tracks,
since the `tracks_loose` dataset in the h5 file has an extra dimension for the tracks
(for each jet we store the information of up to 40 tracks).

1. Write a little python script that loads the jet and track variables.
2. Have a look at how large the dataset is, and what shape the loaded arrays have.

For the following tasks you can re-use the code that loads the h5 file or just extend
your python script from this task.

??? info "Hint: how can I load a h5 file with python?"

    You can find the documentation of `h5py` [here](https://docs.h5py.org/en/stable/quick.html).

??? warning "Solution"

    ```py
    import h5py

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

#### Task 1.2: Plotting the $p_T$ distribution for jets of different flavours

As a next step, you will produce a histogram plot that shows the $p_T$ distribution of
light-flavour jets, $b$-jets and $b$-jets.

??? info "Hint: How do I create a histogram plot with `puma`?"

    You can find the examples of histogram plots
    [here](https://umami-hep.github.io/puma/main/examples/histograms.html) and the
    documentation for histogram plots with `puma`
    [here](https://umami-hep.github.io/puma/main/autoapi/puma/histogram/index.html).

??? warning "Solution"

    ```py
    import h5py
    from puma import Histogram, HistogramPlot

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
        bins_range=(0, 250_000),
        xlabel="$p_T$ [MeV]",
        ylabel="Normalised number of jets",
    )

    # add the histograms
    pt_plot.add(Histogram(jets[is_light]["pt_btagJes"], flavour="ujets"))
    pt_plot.add(Histogram(jets[is_c]["pt_btagJes"], flavour="cjets"))
    pt_plot.add(Histogram(jets[is_b]["pt_btagJes"], flavour="bjets"))

    pt_plot.draw()
    pt_plot.savefig("tutorial_histogram_pT.png")
    ```

#### Task 1.3: Plot the $b$-jets probability output of two different taggers

In this task you will plot the $b$-jets probability of two different taggers - RNNIP
and DIPS.

1.  Create the histogram plot (similar to the one from the previous task) and the different
    histograms. If you plot this for light-flavour jets, $c$-jets and $b$-jets, you should
    have 6 histograms.
2.  Make sure that you use a different linestyle for the histgrams of you second tagger.
3.  Add a ratio panel to the plot
4.  Make your plot look pretty. Have a look at the arguments that are supported by
    [`puma.PlotObject`](https://umami-hep.github.io/puma/main/autoapi/puma/plot_base/index.html#puma.plot_base.PlotObject).

??? info "Hint 1: Histogram and HistogramPlot objects"

    After you defined your HistogramPlot object, you can start adding lines to it.
    This lines are the Histogram objects you need to define.

??? info "Hint 2: Linestyle"

    The `linestyle` can be set when the different `Histogram` lines are initalised.

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
        Histogram(jets[is_light]["rnnip_pb"], ratio_group="ujets", flavour="ujets"),
        reference=True,
    )
    tagger_output_plot.add(
        Histogram(jets[is_c]["rnnip_pb"], ratio_group="cjets", flavour="cjets"),
        reference=True,
    )
    tagger_output_plot.add(
        Histogram(jets[is_b]["rnnip_pb"], ratio_group="bjets", flavour="bjets"),
        reference=True,
    )

    # add the histograms
    tagger_output_plot.add(
        Histogram(
            jets[is_light]["dipsLoose20220314u2_pb"],
            ratio_group="ujets",
            flavour="ujets",
            linestyle="--"
        )
    )
    tagger_output_plot.add(
        Histogram(
            jets[is_c]["dipsLoose20220314v2_pb"],
            ratio_group="cjets",
            flavour="cjets",
            linestyle="--"
        )
    )
    tagger_output_plot.add(
        Histogram(
            jets[is_b]["dipsLoose20220314v2_pb"],
            ratio_group="bjets",
            flavour="bjets",
            linestyle="--"
        )
    )

    tagger_output_plot.draw()
    tagger_output_plot.savefig("tutorial_histogram_tagger_pb_comparison.png")
    ```

#### Task 1.4: Plot a track variable of your choice

In this task you are asked to make a histogram plot of a _track variable_.
This is slightly more tricky, since the array that you load from the h5 file has a
different shape compared to the array storing the jet information. In addition to that,
many entries might be filled with `nan` values, which is challenging here and there.

1.  Choose a track variable that you want to plot.
2.  Create a histogram plot (maybe again for multiple flavours, but that is up to you).

??? info "Hint 1: NaN-Values in binning"

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
        bins_range=(-3, 3),
        xlabel="$d_0$ significance",
        ylabel="Normalised number of tracks",
        figsize=(6, 4.5),
    )

    d0_plot.add(
        Histogram(
            tracks["IP3D_signed_d0_significance"][is_light, :].flatten(), flavour="ujets"
        )
    )
    d0_plot.add(
        Histogram(tracks["IP3D_signed_d0_significance"][is_c, :].flatten(), flavour="cjets")
    )
    d0_plot.add(
        Histogram(tracks["IP3D_signed_d0_significance"][is_b, :].flatten(), flavour="bjets")
    )

    d0_plot.draw()
    d0_plot.savefig("tutorial_histogram_track_variable.png")
    ```

### Task 2: ROC plots

In this task, you will plot a ROC comparison for the two taggers _RNNIP_ and _DIPS_.

#### Task 2.1: Calculate the rejections as a function of the $b$-jets efficiency

Before you can actually plot the ROC curves, you have to calculate the light-flavour
and $c$-jets rejection for a range of $b$-jets efficiencies.

1.  Define a function that calculates the $b$-jets discriminant from the tagger output.
2.  Calculate the light-flavour jets rejection as a function of the $b$-jets efficiency.

??? warning "Solution"

    ```py

    import numpy as np
    import pandas as pd
    import h5py

    from puma import Roc, RocPlot
    from puma.metrics import calc_rej

    ttbar_filepath = "/eos/user/u/umamibot/tutorials/ttbar.h5"

    # load the jets dataset from the h5 file

    with h5py.File(ttbar_filepath, "r") as h5file:
        jets = pd.DataFrame(h5file["jets"][:])

    # defining boolean arrays to select the different flavour classes
    is_light = jets["HadronConeExclTruthLabelID"] == 0
    is_c = jets["HadronConeExclTruthLabelID"] == 4
    is_b = jets["HadronConeExclTruthLabelID"] == 5

    # define a small function to calculate discriminant
    def disc_fct(arr: np.ndarray, f_c: float = 0.018) -> np.ndarray:
        """Tagger discriminant

        Parameters
        ----------
        arr : numpy.ndarray
            array with with shape (, 3)
        f_c : float, optional
            f_c value in the discriminant (weight for c-jets rejection)

        Returns
        -------
        np.ndarray
            Array with the discriminant values inside.
        """
        # you can adapt this for your needs
        return np.log(arr[2] / (f_c * arr[1] + (1 - f_c) * arr[0]))


    # calculate discriminant
    discs_rnnip = np.apply_along_axis(
        disc_fct, 1, jets[["rnnip_pu", "rnnip_pc", "rnnip_pb"]].values
    )
    discs_dips = np.apply_along_axis(
        disc_fct,
        1,
        jets[
            ["dipsLoose20220314v2_pu", "dipsLoose20220314v2_pc", "dipsLoose20220314v2_pb"]
        ].values,
    )
    # defining target efficiency
    sig_eff = np.linspace(0.49, 1, 20)
    # defining boolean arrays to select the different flavour classes
    is_light = jets["HadronConeExclTruthLabelID"] == 0
    is_c = jets["HadronConeExclTruthLabelID"] == 4
    is_b = jets["HadronConeExclTruthLabelID"] == 5

    n_jets_light = sum(is_light)
    n_jets_c = sum(is_c)

    rnnip_ujets_rej = calc_rej(discs_rnnip[is_b], discs_rnnip[is_light], sig_eff)
    dips_ujets_rej = calc_rej(discs_dips[is_b], discs_dips[is_light], sig_eff)
    ```

#### Task 2.2:

1.  Plot the light-flavour jets rejection as a function of the $b$-jets efficiency. Use
    `n_ratio_panels=1` to also get the ratio of the two rejection curves.

??? info "Hint 1: How do I initialise a ROC curve plot?"

    Plotting ROC curves with `puma` is similar to plotting histograms. The main difference
    is that you are using the `puma.RocPlot` and `puma.Roc` classes. Search the
    [puma docs] for "roc" to have a look at an example and the API reference.

??? info "Hint 2: I initialised the plot and added the ROC curves - is there anything else to do?"

    For ROC curves you also have to define the class which is drawn in the ratio panel.
    The method you need to use here is `RocPlot.set_ratio_class()`.

??? warning "Solution"

    ```py
    # here the plotting of the roc starts
    roc_plot = RocPlot(
        n_ratio_panels=1,
        ylabel="Background rejection",
        xlabel="$b$-jets efficiency",
        atlas_second_tag="$\\sqrt{s}=13$ TeV, $t\\bar{t}$ Release 22, \n$f_c=0.018$",
    )
    roc_plot.add_roc(
        Roc(
            sig_eff,
            rnnip_ujets_rej,
            n_test=n_jets_light,
            rej_class="ujets",
            signal_class="bjets",
            label="RNNIP",
        ),
        reference=True,
    )
    roc_plot.add_roc(
        Roc(
            sig_eff,
            dips_ujets_rej,
            n_test=n_jets_light,
            rej_class="ujets",
            signal_class="bjets",
            label="DIPS",
        ),
    )
    roc_plot.set_ratio_class(1, "ujets", label="Light-flavour jets ratio")
    roc_plot.set_leg_rej_labels("ujets", "Light-flavour jets rejection")
    roc_plot.draw()
    roc_plot.savefig("tutorial_roc.png", transparent=False)
    ```

#### Task 2.3: Add the $c$-rejection to your plot

1.  Repeat the calculation of the rejection for $c$-jets
2.  Add the corresponding ROC curves to the plot. Don't forget to increase `n_ratio_panels`
    of your `puma.RocPlot`.

??? info "Hint 1: How can I modify the legend that states which linestyle corresponds to the different rejection classes"

    You can add labels for the different rejection classes using the `RocPlot.set_leg_rej_labels()` method.

??? warning "Solution"

    ```py
    rnnip_cjets_rej = calc_rej(discs_rnnip[is_b], discs_rnnip[is_c], sig_eff)
    dips_cjets_rej = calc_rej(discs_dips[is_b], discs_dips[is_c], sig_eff)

    # add this to the code from the previous task (has to be before the RocPlot.draw()
    # method is called)
    roc_plot.add_roc(
        Roc(
            sig_eff,
            rnnip_cjets_rej,
            n_test=n_jets_c,
            rej_class="cjets",
            signal_class="bjets",
            label="RNNIP",
        ),
        reference=True,
    )
    roc_plot.add_roc(
        Roc(
            sig_eff,
            dips_cjets_rej,
            n_test=n_jets_c,
            rej_class="cjets",
            signal_class="bjets",
            label="DIPS",
        ),
    )
    roc_plot.set_ratio_class(2, "cjets", label="$c$-jets ratio")
    roc_plot.set_leg_rej_labels("cjets", "$c$-jets rejection")
    ```

### Task 3: $p_T$ vs. efficiency

In this task, you will plot both the $b$-jets efficiency and the light-flavour jets
rejection for specific bins of $p_T$.

#### Task 3.1: Calculate the discriminant values

Just like you did in Task 2.1, calculate the discriminant scores for _RNNIP_ and
_DIPS_. You can reuse the code from task 2.1. If you are putting everything in one
python script you can just reuse the values that are already calculated.

#### Task 3.2: Create a $p_T$ vs. $b$-efficiency plot

For a fixed inclusive $b$-efficiency, you plot the $b$-efficiency for different bins
of $p_T$.

??? warning "Solution"

    ```py
    import numpy as np
    import pandas as pd
    import h5py

    from puma import VarVsEff, VarVsEffPlot

    ttbar_filepath = "/eos/user/u/umamibot/tutorials/ttbar.h5"

    # load the jets dataset from the h5 file
    with h5py.File(ttbar_filepath, "r") as h5file:
        jets = pd.DataFrame(h5file["jets"][:])

    # define a small function to calculate discriminant
    def disc_fct(arr: np.ndarray, f_c: float = 0.018) -> np.ndarray:
        """Tagger discriminant

        Parameters
        ----------
        arr : numpy.ndarray
            array with with shape (, 3)
        f_c : float, optional
            f_c value in the discriminant (weight for c-jets rejection)

        Returns
        -------
        np.ndarray
            Array with the discriminant values inside.
        """
        # you can adapt this for your needs
        return np.log(arr[2] / (f_c * arr[1] + (1 - f_c) * arr[0]))


    # calculate discriminant
    discs_rnnip = np.apply_along_axis(
        disc_fct, 1, jets[["rnnip_pu", "rnnip_pc", "rnnip_pb"]].values
    )
    discs_dips = np.apply_along_axis(
        disc_fct,
        1,
        jets[
            ["dipsLoose20220314v2_pu", "dipsLoose20220314v2_pc", "dipsLoose20220314v2_pb"]
        ].values,
    )

    # Getting jet pt in GeV
    pt = jets["pt"].values / 1e3
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
        fixed_eff_bin=False,
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
        fixed_eff_bin=False,
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

    plot_sig_eff.atlas_second_tag += "\nInclusive $\\epsilon_b=70%%$"

    # If you want to inverse the discriminant cut you can enable it via
    # plot_sig_eff.set_inverse_cut()
    plot_sig_eff.draw()
    # Drawing a hline indicating inclusive efficiency
    plot_sig_eff.draw_hline(0.7)
    plot_sig_eff.savefig("tutorial_pt_b_eff.png", transparent=False)
    ```

#### Task 3.3: Create a $p_T$ vs. light-flavour jets rejection plot

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
    plot_sig_eff.atlas_second_tag += "\nInclusive $\\epsilon_b=70%%$"
    plot_bkg_rej.add(rnnip_light, reference=True)
    plot_bkg_rej.add(dips_light)

    plot_bkg_rej.draw()
    plot_bkg_rej.savefig("tutorial_pt_light_rej.png")
    ```

### Task 4: Plotting inside of Umami

Although `puma` gives you the opportunity to plot all the histograms by yourself in a very easy way, most of the plots you did up till now are also producable via `umami`. We will start in this tutorial with the plotting of input variables with some specific settings. To do so, you need to fork and clone the `umami` repository from [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami).

#### Task 4.1: Setup umami

To use all the functionalities of `umami`, you need to properly setup and load `umami`. Try to access the `umami` singularity image and source the `run_setup.sh`.

??? info "Hint: Setup umami"

    You can find the documentation of `umami` [here](https://umami-docs.web.cern.ch/setup/installation/).

??? warning "Solution: Setup umami"

    After you have forked and downloaded it, you can (and should be able to) switch to the umami folder and start the `umami` singularity image (the same one you are already using). Now you just need to run the setup for `umami` via

    ```bash
    source run_setup.sh
    ```

    which should load all needed settings for `umami`.

#### Task 4.2: Create a new Plotting Config File

`umami` uses yaml files as config files for most of it's functions. One of the most basic functionalities is the plotting of variables inside a `.h5` file which was produced by the [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper). To do so, you need to create a new yaml config file. This config file should contain for now just one dict named `Eval_parameters` with two entries: `nJets` and `var_dict`. `nJets` is an `int` for how many jets will be used for plotting and `var_dict` is a string with the path to the variable config which will be used. For this tutorial, you can use one of the configs provided inside of `umami`. The path is `umami/umami/configs/Umami_Variables_R22.yaml`.

??? info "Hint: Create a new plotting config file"

    A little explanation how python dicts are written in yaml files is given [here](https://stackoverflow.com/a/13020322)

??? warning "Solution: Create a new plotting config file"

    ```yaml
    Eval_parameters:
      nJets: 3e4
      var_dict: <path_palce_holder>/umami/umami/configs/Dips_Variables.yaml
    ```

#### Task 4.3: Plot Jet Input Variables

The most basic to plot are the jet-level variables. To plot those, you need to add a new entry to the dict in the yaml file. The name of the entry is not relevant but this dict entry needs very specific keys. To guide you trough the process, small sub-tasks are given here.
To run the plotting of the jet variables, you need to run the following command in the `umami/umami/` directory

```bash
plot_input_vars.py -c <path/to/config> --jets
````

##### Task 4.3.1: Plot the $p_T$ and |η| distribution

Try to plot the `pt_btagJes` and the `absEta_btagJes` distributions for the ttbar and the Z' sample in the same and in two different plots. Plot this for
the _b_-, _c_- and light jets (all in one plot).

??? info "Hint: Plot the $p_T$ and |η| distribution"

    The explanation of the different options are given [here](https://umami-docs.web.cern.ch/plotting/plotting_inputs/#input-variables-jets). The label values of the different flavours can be found [here](https://umami-docs.web.cern.ch/preprocessing/truth_labels/).

??? warning "Solution: Plot the $p_T$ and |η| distribution"

    ```yaml
    ttbar_only:
      variables: "jets"
      folder_to_save: jets_input_vars
      Datasets_to_plot:
        ttbar:
          files: <path_palce_holder>/ttbar.h5
          label: "ttbar"
      class_labels: ["bjets", "cjets", "ujets"]
      binning:
        pt_btagJes: 100
        absEta_btagJes: 100
      flavours:
        b: 5
        c: 4
        u: 0
      plot_settings:
        logy: True
        use_atlas_tag: True
        atlas_first_tag: "Simulation Internal"
        atlas_second_tag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow jets \n30000 jets"
        y_scale: 2
        figsize: [7, 5]
    ```

    ```yaml
    zpext_only:
      variables: "jets"
      folder_to_save: jets_input_vars
      Datasets_to_plot:
        zpext:
          files: <path_palce_holder>/zpext.h5
          label: "zpext"
      class_labels: ["bjets", "cjets", "ujets"]
      binning:
        pt_btagJes: 100
        absEta_btagJes: 100
      flavours:
        b: 5
        c: 4
        u: 0
      plot_settings:
        logy: True
        use_atlas_tag: True
        atlas_first_tag: "Simulation Internal"
        atlas_second_tag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow jets \n30000 jets"
        y_scale: 2
        figsize: [7, 5]
    ```

    ```yaml
    ttbar_zpext:
      variables: "jets"
      folder_to_save: jets_input_vars
      Datasets_to_plot:
        ttbar:
          files: <path_palce_holder>/ttbar.h5
          label: "ttbar"
        zpext:
          files: <path_palce_holder>/zpext.h5
          label: "zpext"
      class_labels: ["bjets", "cjets", "ujets"]
      binning:
        pt_btagJes: 100
        absEta_btagJes: 100
      flavours:
        b: 5
        c: 4
        u: 0
      plot_settings:
        logy: True
        use_atlas_tag: True
        atlas_first_tag: "Simulation Internal"
        atlas_second_tag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow jets \n30000 jets"
        y_scale: 2
        figsize: [7, 5]
    ```

##### Task 4.3.2: Adding new Variables to the Variable Config File

Try to add the output probabilities of the `dipsLoose20220314v2` tagger to the `var_dict` file in a new sublist called `DIPS`. The probabilites are named using the tagger name and the probability in short form, i.e. `dipsLoose20220314v2_pb`.

??? info "Hint: Adding new Variables to the Variable Config File"

    The `var_dict` file can be found in the umami folder under the path `umami/umami/configs/Umami_Variables_R22.yaml`.
    The variable names you need to add are `dipsLoose20220314v2_pb`, `dipsLoose20220314v2_pc`, `dipsLoose20220314v2_pu`.
    To plot the new variables, you need to also add them to the `binning` of the already existing config entries.

??? warning "Solution: Adding new Variables to the Variable Config File"

    ```yaml
    ttbar_only:
      variables: "jets"
      folder_to_save: jets_input_vars
      Datasets_to_plot:
        ttbar:
          files: <path_palce_holder>/ttbar.h5
          label: "ttbar"
      class_labels: ["bjets", "cjets", "ujets"]
      binning:
        pt_btagJes: 100
        absEta_btagJes: 100
        dipsLoose20220314v2_pb: 100
        dipsLoose20220314v2_pc: 100
        dipsLoose20220314v2_pu: 100
      flavours:
        b: 5
        c: 4
        u: 0
      plot_settings:
        logy: True
        use_atlas_tag: True
        atlas_first_tag: "Simulation Internal"
        atlas_second_tag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow jets \n30000 jets"
        y_scale: 2
        figsize: [7, 5]
    ```

    The other two can be adapted similary.

##### Task 4.3.3: Adding specific lower and upper x-limits

Try to plot the `JetFitterSecondaryVertex_nTracks` variable from `0` to `17` with 17 bins.

??? info "Hint: Adding specific lower and upper x-limits"

    The entry, like `binning` and `flavours`, you are looking for is called `special_param_jets`. You
    can look this up [here](https://umami-docs.web.cern.ch/plotting/plotting_inputs/#input-variables-jets).

??? warning "Solution: Adding specific lower and upper x-limits"

    ```yaml
    ttbar_only:
      variables: "jets"
      folder_to_save: jets_input_vars
      Datasets_to_plot:
        ttbar:
          files: <path_palce_holder>/ttbar.h5
          label: "ttbar"
      class_labels: ["bjets", "cjets", "ujets"]
      binning:
        pt_btagJes: 100
        absEta_btagJes: 100
        dipsLoose20220314v2_pb: 100
        dipsLoose20220314v2_pc: 100
        dipsLoose20220314v2_pu: 100
        JetFitterSecondaryVertex_nTracks: 17
      flavours:
        b: 5
        c: 4
        u: 0
      special_param_jets:
        JetFitterSecondaryVertex_nTracks:
          lim_left: 0
          lim_right: 17
      plot_settings:
        logy: True
        use_atlas_tag: True
        atlas_first_tag: "Simulation Internal"
        atlas_second_tag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow jets \n30000 jets"
        y_scale: 2
        figsize: [7, 5]
    ```

    The other two can be adapted similary.

#### Task 4.4: Plot Track Input Variables

Due to the different dimensionality of the `tracks`, the track input variables have their own plotting function. The setup for the plots is pretty similar to the ones of the jets. To guide you through the different possibilities you have when plotting, there are some sub-tasks.
To run the plotting of the track variables, you need to run the following command in the `umami/umami/` directory

```bash
plot_input_vars.py -c <path/to/config> --tracks
```

##### Task 4.4.1: Plot all Tracks

Try to plot the `IP3D_signed_d0_significance` and the `IP3D_signed_z0_significance` of the ttbar and Z' test files (ttbar only, Z' only, both in the same plots). You need to figure out how the `tracks` are called inside the files to properly load them and plot them.

??? info "Hint: Plot all Tracks"

    The name of the `tracks` in the files can be found out by using the `h5ls` command. For the entry in the config, you can look into the documentation for the options you need to define. This can be done [here](https://umami-docs.web.cern.ch/plotting/plotting_inputs/#input-variables-tracks).

??? warning "Solution: Plot all Tracks"

    ```yaml
    ttbar_only:
      variables: "tracks"
      folder_to_save: ttbar_only
      Datasets_to_plot:
        ttbar:
          files: <path_palce_holder>/ttbar.h5
          label: "ttbar"
          tracks_name: "tracks_loose"
      plot_settings:
        logy: True
        use_atlas_tag: True
        atlas_first_tag: "Simulation Internal"
        atlas_second_tag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow jets \n30000 jets"
        y_scale: 2
        figsize: [7, 5]
      binning:
        IP3D_signed_d0_significance: 100
        IP3D_signed_z0_significance: 100
      class_labels: ["bjets", "cjets", "ujets"]
    ```

    ```yaml
    zpext_only:
      variables: "tracks"
      folder_to_save: zpext_only
      Datasets_to_plot:
        zpext:
          files: <path_palce_holder>/zpext.h5
          label: "zpext"
          tracks_name: "tracks_loose"
      plot_settings:
        logy: True
        use_atlas_tag: True
        atlas_first_tag: "Simulation Internal"
        atlas_second_tag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow jets \n30000 jets"
        y_scale: 2
        figsize: [7, 5]
      binning:
        IP3D_signed_d0_significance: 100
        IP3D_signed_z0_significance: 100
      class_labels: ["bjets", "cjets", "ujets"]
    ```

    ```yaml
    ttbar_zpext:
      variables: "tracks"
      folder_to_save: ttbar_zpext
      Datasets_to_plot:
        ttbar:
          files: <path_palce_holder>/ttbar.h5
          label: "ttbar"
          tracks_name: "tracks_loose"
        zpext:
          files: <path_palce_holder>/zpext.h5
          label: "zpext"
          tracks_name: "tracks_loose"
      plot_settings:
        logy: True
        use_atlas_tag: True
        atlas_first_tag: "Simulation Internal"
        atlas_second_tag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow jets \n30000 jets"
        y_scale: 2
        figsize: [7, 5]
      binning:
        IP3D_signed_d0_significance: 100
        IP3D_signed_z0_significance: 100
      class_labels: ["bjets", "cjets", "ujets"]
    ```

##### Task 4.4.2: Plot only Second-leading Tracks in `ptfrac`

One specific thing about the tracks is the ordering. To order them properly, you can give the `sorting_variable` option which tells the function, after which variable the tracks are sorted (in decreasing order, starting with the highest value). Also, you can tell the function which n-leading tracks from each jet should be plotted.
Try to plot only the second leading tracks (ordered by `ptfrac`).

??? info "Hint: Plot only Second-leading Tracks in `ptfrac`"

    You can adapt again the plot config of the last sub-task. The options you need here to add can be found [here](https://umami-docs.web.cern.ch/plotting/plotting_inputs/#input-variables-tracks).

??? warning "Solution: Plot only Second-leading Tracks in `ptfrac`"

    ```yaml
    ttbar_only:
      variables: "tracks"
      folder_to_save: ttbar_only
      Datasets_to_plot:
        ttbar:
          files: <path_palce_holder>/ttbar.h5
          label: "ttbar"
          tracks_name: "tracks_loose"
      plot_settings:
        logy: True
        use_atlas_tag: True
        atlas_first_tag: "Simulation Internal"
        atlas_second_tag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow jets \n30000 jets"
        y_scale: 2
        figsize: [7, 5]
        sorting_variable: "ptfrac"
        n_leading: [2]
      binning:
        IP3D_signed_d0_significance: 100
        IP3D_signed_z0_significance: 100
      class_labels: ["bjets", "cjets", "ujets"]
    ```

    The other two can be adapted similary.

#### Task 4.5: Plot Number of Tracks per Flavour

Another important factor of the tracks is the track multiplicity for the given jets. This can also be plotted using the umami. To do so, you need to define another entry in your config file. In theory, you can keep the basic structure from the `tracks` plotting, but you can get rid of the `binning` part and set the `nTracks` argument to `True`.
Try to plot the number of tracks for the files (ttbar, Z', both) for the three basic flavours.

??? info "Hint: Plot Number of Tracks per Flavour"

    All options available (and which are necessary and optional) can be found [here](https://umami-docs.web.cern.ch/plotting/plotting_inputs/#number-of-tracks-per-jet).

??? warning "Solution: Plot Number of Tracks per Flavour"

    ```yaml
    nTracks_ttbar:
      variables: "tracks"
      folder_to_save: nTracks_ttbar
      nTracks: True
      Datasets_to_plot:
        ttbar:
          files: <path_palce_holder>/ttbar.h5
          label: "ttbar"
          tracks_name: "tracks_loose"
      plot_settings:
          logy: True
          use_atlas_tag: True
          atlas_first_tag: "Simulation Internal"
          atlas_second_tag: "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$ PFlow jets \n30000 jets"
          y_scale: 2
          figsize: [7, 5]
      class_labels: ["bjets", "cjets", "ujets"]
    ```

    The other two can be adapted similary.

## Bonus tasks

### Run over a Run-3 MC sample and compare the pileup distributions

This task will extend over the simple histogram plotting you already encountered in Task 1.
You are asked to compare distributions from two different files: the Run-2 MC for the Z' sample and the Run-3 MC for the Z' sample.

For this task, you will:

1. Download the Z' sample for the Run-3 MC `zpext_run3.h5` from `/eos/user/u/umamibot/tutorials/`.
2. Write a plotting script to compare the `averageInteractionsPerCrossing` between the two samples.

??? warning "Solution"

    Copy the Run-3 MC file (assuming you work on lxplus):

    ```bash

    cp /eos/user/u/umamibot/tutorials/zpext_run3.h5 </path/to/tutorial/data/>
    ```

    You should provide a path for the dummy `</path/to/tutorial/data/>` in the command above and in the
    python example below:


    ```python
    import numpy as np
    import h5py
    from puma import Histogram, HistogramPlot

    # load the "jets" datasets from the h5 files
    filepath_run2 = "/path/to/tutorial/data/zpext.h5"
    with h5py.File(filepath_run2, "r") as h5file:
        jets_run2 = h5file["jets"][:]

    filepath_run3 = "/path/to/tutorial/data/zpext_run3.h5"
    with h5py.File(filepath_run3, "r") as h5file:
        jets_run3 = h5file["jets"][:]

    variable = "averageInteractionsPerCrossing"
    run_2 = Histogram(jets_run2[variable], label="Run 2 MC")
    run_3 = Histogram(jets_run3[variable], label="Run 3 MC")

    # Initialise histogram plot
    plot_histo = HistogramPlot(
        ylabel="Number of events",
        xlabel=r"average interactions per crossing $\langle\mu\rangle$ [a.u.]",
        logy=False,
        bins=60,
        bins_range=(10, 70),
        norm=True,
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

### Compare the "flipped taggers" to the regular flavour tagging algorithms

This task further extends over the simple histogram plotting you already encountered in Task 1.
You are asked to compare distributions from a regular flavour tagging algorithm and a so-called "flipped tagger", which is a modified version of the flavour tagging algorithm used for light-jet mistag calibration.
For this version, the sign of d0/z0 signed impact parameter is flipped, resulting in a selection of jets with “negative lifetime”.

Consequently, the flipped tagger's b-tagging efficiency is reduced while its light-jet mistag rate is left unchanged.

For this task, you will:

1. Write a plotting script to compare the scores $p_b$, $p_c$, and $p_u$, for the RNNIP tagger to the flipped version. You should produce three plots, one for each score (such as $p_b$), which show the distributions of the RNNIP tagger and the flipped RNNIP tagger overlaid for the three different jet flavours b-jets, c-jets and light-flavour jets.
2. Next, extend the script to compare also the flavour tagging discriminant based on the flipped tagger and the regular RNNIP tagger. You should produce one plot which compares the distributions of the RNNIP tagger and the flipped RNNIP tagger overlaid for the three different jet flavours b-jets, c-jets and light-flavour jets.

??? info "Hint: Names of the RNNIP tagger and the flipped tagger scores and the corresponding b-tagging discriminant"

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
    import h5py
    import pandas as pd
    from puma import Histogram, HistogramPlot

    # load the "jets" dataset from the h5 file
    filepath = "/path/to/tutorial/data/ttbar.h5"
    with h5py.File(filepath, "r") as h5file:
        jets = h5file["jets"][:]
        jets = pd.DataFrame(jets)


    # defining boolean arrays to select the different flavour classes
    is_light = jets["HadronConeExclTruthLabelID"] == 0
    is_c = jets["HadronConeExclTruthLabelID"] == 4
    is_b = jets["HadronConeExclTruthLabelID"] == 5


    # Calculate discriminant scores for RNNIP and flipped tagger, and add them to the dataframe
    FRAC_C = 0.07
    jets["disc_rnnip"] = np.log(
        jets["rnnip_pb"] / (FRAC_C * jets["rnnip_pc"] + (1 - FRAC_C) * jets["rnnip_pu"])
    )
    jets["disc_rnnipflip"] = np.log(
        jets["rnnipflip_pb"] / (FRAC_C * jets["rnnipflip_pc"] + (1 - FRAC_C) * jets["rnnipflip_pu"])
    )

    variables = [
        ('rnnip_pu', 'rnnipflip_pu'),
        ('rnnip_pc', 'rnnipflip_pc'),
        ('rnnip_pb', 'rnnipflip_pb'),
        ('disc_rnnip', 'disc_rnnipflip'),
    ]

    axis_labels = {
        'rnnip_pu': 'RNNIP $p_{light}$',
        'rnnip_pc': 'RNNIP $p_{c}$',
        'rnnip_pb': 'RNNIP $p_{b}$',
        'disc_rnnip': 'RNNIP b-tagging discriminant',
    }


    # plot score and discriminantdistributions
    for v in variables:
        rnnip_light = Histogram(jets[is_light][v[0]], flavour="ujets", label="RNNIP")
        rnnip_c = Histogram(jets[is_c][v[0]], flavour="cjets", label="RNNIP")
        rnnip_b = Histogram(jets[is_b][v[0]], flavour="bjets", label="RNNIP")

        rnnip_light_flip = Histogram(jets[is_light][v[1]], linestyle="dashed", flavour="ujets", label="RNNIP (flip)")
        rnnip_c_flip = Histogram(jets[is_c][v[1]], linestyle="dashed", flavour="cjets", label="RNNIP (flip)")
        rnnip_b_flip = Histogram(jets[is_b][v[1]], linestyle="dashed", flavour="bjets", label="RNNIP (flip)")

        # Initialise histogram plot
        plot_histo = HistogramPlot(
            ylabel="Number of events",
            xlabel=axis_labels[v[0]],
            logy=True,
            bins=np.linspace(-10, 10, 40) if v[0] == 'disc_rnnip' else np.linspace(0, 1, 20),
            norm=False,
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
