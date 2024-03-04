## Some basics before plotting

In the following a small example how to read in h5 and calculate the _b_-tagging
discriminant.

First you need to import some packages

```py
import h5py
import numpy as np
import pandas as pd


from puma.metrics import calc_rej
```

Then you can read a h5 file that contains the data you want to plot:

```py
# this is just an example to read in your h5 file
# if you have tagger predictions you can plug them in directly in the `disc_fct` as well
# taking one random ttbar file
ttbar_file = (
    "user.alfroch.410470.btagTraining.e6337_s3681_r13144_p4931.EMPFlowAll."
    "2022-02-07-T174158_output.h5/user.alfroch.28040424._001207.output.h5"
)
with h5py.File(ttbar_file, "r") as f:
    df = pd.DataFrame(
        f["jets"].fields(
            [
                "rnnip_pu",
                "rnnip_pc",
                "rnnip_pb",
                "dipsLoose20210729_pu",
                "dipsLoose20210729_pc",
                "dipsLoose20210729_pb",
                "HadronConeExclTruthLabelID",
            ]
        )[:300000]
    )
    n_test = len(df)
```

To run the next steps you need to modify the format of the dataframe:
```
df = df.to_records()
```

In the example below you can find an example on how you can calculate the tagger
discriminant using the raw output (i.e. `p_u`, `p_c` and `p_b`) of the tagger,
and a function from [atlas-ftag-tools](https://github.com/umami-hep/atlas-ftag-tools/).

```py
from ftag import get_discriminant
discs_rnnip = get_discriminant(df, "rnnip", signal="bjets", fc=0.018)
discs_dips = get_discriminant(df, "dips", signal="bjets", fc=0.018)
```

To calculate the rejection values you can do the following or using a results file
from a previous evaluation.

```py
# defining target efficiency
sig_eff = np.linspace(0.49, 1, 20)
# defining boolean arrays to select the different flavour classes
is_light = df["HadronConeExclTruthLabelID"] == 0
is_c = df["HadronConeExclTruthLabelID"] == 4
is_b = df["HadronConeExclTruthLabelID"] == 5

rnnip_ujets_rej = calc_rej(discs_rnnip[is_b], discs_rnnip[is_light], sig_eff)
rnnip_cjets_rej = calc_rej(discs_rnnip[is_b], discs_rnnip[is_c], sig_eff)
dips_ujets_rej = calc_rej(discs_dips[is_b], discs_dips[is_light], sig_eff)
dips_cjets_rej = calc_rej(discs_dips[is_b], discs_dips[is_c], sig_eff)
```

```py
# Alternatively you can simply use a results file with the rejection values
df = pd.read_hdf("results-rej_per_eff-1_new.h5", "ttbar")
print(df.columns.values)
sig_eff = df["effs"]
rnnip_ujets_rej = df["rnnip_ujets_rej"]
rnnip_cjets_rej = df["rnnip_cjets_rej"]
dips_ujets_rej = df["dips_ujets_rej"]
dips_cjets_rej = df["dips_cjets_rej"]
n_test = 10_000
# it is also possible to extract it from the h5 attributes
with h5py.File("results-rej_per_eff-1_new.h5", "r") as h5_file:
    n_test = h5_file.attrs["N_test"]
```

## Showing the plot in pop-up window

`puma` does *not* use the `matplotlib.pyplot` module, but instead uses the `matplotlib.figure`
API in order to avoid global states within `matplotlib` that could affect subsequent plots.

It is recommended to just save the plot via the `savefig` method and then view it with
a local pdf/png viewer. If you are working on a cluster, you can also mount the directory
you are working on to your local machine, which allows you to use your locally installed pdf viewer.

However, if you want to have a pop-up window when running a python script with a `puma`
plot, you can try the following workaround using the `PIL` package.

```py
from puma import Histogram, HistogramPlot
import numpy as np  # used here for random numbers
from PIL import Image  # used for the pop-up window

my_hist = HistogramPlot()
my_hist.add(Histogram(np.random.normal(size=10_000)))
my_hist.draw()
my_hist.savefig("my_hist.png")

# show the image
img = Image.open("my_hist.png")
img.show()
```