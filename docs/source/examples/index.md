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

In the example below you can find an example on how you can calculate the tagger
discriminant using the raw output (i.e. `p_u`, `p_c` and `p_b`) of the tagger.

```py
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


# you can also use a lambda function
# discs_rnnip = np.apply_along_axis(
#     lambda a: np.log(a[2] / (0.018 * a[1] + (1 - 0.018) * a[0])),
#     1,
#     df[["rnnip_pu", "rnnip_pc", "rnnip_pb"]].values,
# )

# calculate discriminant
discs_rnnip = np.apply_along_axis(
    disc_fct, 1, df[["rnnip_pu", "rnnip_pc", "rnnip_pb"]].values
)
discs_dips = np.apply_along_axis(
    disc_fct,
    1,
    df[["dipsLoose20210729_pu", "dipsLoose20210729_pc", "dipsLoose20210729_pb"]].values,
)
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
