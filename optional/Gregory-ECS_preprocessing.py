#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate quick Gregory ECS diagnostics and overview plots
============================================================

This script provides a simplified and computationally inexpensive
calculation of equilibrium climate sensitivity (ECS), effective
radiative forcing (ERF), and the net feedback parameter (lambda)
from CMIP5 or CMIP6 abrupt-4xCO2 experiments.

Unlike the full climate-feedback workflow, this script does not
calculate individual radiative-kernel feedbacks. It is intended as
a quick diagnostic for inspecting the Gregory regressions and
comparing ECS across models.

For each model, global annual-mean surface temperature change and
TOA radiative imbalance are calculated relative to the piControl
simulation. The piControl reference is removed using a 21-year
running mean, following Caldwell et al. (2016) and Zelinka et al.
(2020).

ECS and ERF are derived from the linear Gregory regression:

    N = F + lambda * Delta T

where F is the effective radiative forcing for a 2xCO2-equivalent
forcing, lambda is the net feedback parameter, and ECS is derived
from the zero-crossing of the regression line.

The script additionally produces an overview figure showing the
Gregory regression for all models.

This script is intended as a fast diagnostic and does not replace
the full feedback calculation used for the main analysis.
"""


#%%
# =================================
# Imports
# =================================
import glob
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")


# =================================
# Paths and settings
# =================================

# Directory containing the preprocessed CMIP data.
DIR = "/path/to/data_output"
# Directory where the calculated diagnostics are saved.
SAVE = os.path.join(DIR, "feedbacks")

# Variables required for the Gregory regression.
VARIABLES = [
    "tas",
    "rsut",
    "rsdt",
    "rlut",
]

# Experiments.
PI_CONTROL = "piControl"
ABRUPT_4XCO2 = "abrupt-4xCO2"

EXPERIMENTS = [
    PI_CONTROL,
    ABRUPT_4XCO2,
]

# Number of years used for the piControl running mean.
RUNNING_MEAN = 21


# =================================
# Utility functions
# =================================

def common_model_list_experiments(
    directory,
    variables,
    experiments,
):
    """
    Return models with all requested variables and experiments.
    """

    common = None

    for variable in variables:

        models = None

        for experiment in experiments:

            experiment_models = set()

            for filename in os.listdir(directory):

                if (
                    filename.startswith(variable + "_")
                    and experiment in filename
                ):
                    experiment_models.add(
                        filename.split("_")[2]
                    )

            models = (
                experiment_models
                if models is None
                else models & experiment_models
            )

        common = (
            models
            if common is None
            else common & models
        )

    common = sorted(common)

    print(
        f"Number of models with {variables} and "
        f"{experiments}: {len(common)}"
    )

    return common


def gam(data):
    """
    Calculate the global annual-mean time series.

    The global mean is weighted by the cosine of latitude and
    monthly values are subsequently averaged to annual means.

    This function assumes a regular latitude-longitude grid.
    """

    weights = np.cos(
        np.deg2rad(data.lat)
    )

    return (
        data
        .weighted(weights)
        .mean(dim={"lat","lon","month"})
    )

def read(var, model, experiment):
    """
    Read a preprocessed CMIP variable for one model and experiment.
    """

    filename = glob.glob(
        f"{DIR}/*{var}_Amon_{model}_{experiment}_*.nc"
    )

    if not filename:
        return None

    return xr.open_dataset(
        filename[0]
    )[var]


def gregory_regression(d_tas, d_r):
    """
    Calculate Gregory regression parameters.

    Returns:
        lambda  : net feedback parameter
        erf2x   : effective radiative forcing for 2xCO2
        ecs     : equilibrium climate sensitivity
    """

    x = d_tas.values
    y = d_r.values

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    slope, intercept = np.polyfit(
        x[valid],
        y[valid],
        1,
    )

    # The intercept corresponds to the 4xCO2 ERF.
    # Divide by two to obtain the 2xCO2-equivalent ERF.
    erf2x = intercept / 2.0

    # ECS corresponds to the temperature at which N = 0.
    # The factor 0.5 accounts for the 4xCO2 experiment.
    ecs = -intercept / slope * 0.5

    return (
        xr.DataArray(
            slope,
            name="lambda",
        ),
        xr.DataArray(
            erf2x,
            name="erf2x",
        ),
        xr.DataArray(
            ecs,
            name="ecs",
        ),
    )


# =================================
# Calculate Gregory diagnostics
# =================================

models = common_model_list_experiments(
    DIR,
    VARIABLES,
    EXPERIMENTS,
)

datasets = []

for i, model in enumerate(models):

    print(
        f"[{i + 1}/{len(models)}] {model}"
    )

    # ---------------------------------
    # Read input data
    # ---------------------------------

    piC = {
        var: read(
            var,
            model,
            PI_CONTROL,
        )
        for var in VARIABLES
    }

    co2 = {
        var: read(
            var,
            model,
            ABRUPT_4XCO2,
        )
        for var in VARIABLES
    }

    if any(
        value is None
        for value in piC.values()
    ) or any(
        value is None
        for value in co2.values()
    ):
        print(
            f"  [{model}] missing input data"
        )
        continue

    # ---------------------------------
    # Global annual-mean anomalies
    # ---------------------------------

    tas_piC = gam(piC["tas"])
    tas_co2 = gam(co2["tas"])

    toa_piC = gam(
        -piC["rsut"]
        - piC["rlut"]
    )

    toa_co2 = gam(
        -co2["rsut"]
        - co2["rlut"]
    )

    # Remove the piControl mean state using a
    # 21-year centred running mean.
    piC_tas_reference = (
        tas_piC
        .rolling(
            year=RUNNING_MEAN,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    piC_toa_reference = (
        toa_piC
        .rolling(
            year=RUNNING_MEAN,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    d_tas = (
        tas_co2
        - piC_tas_reference
    )

    d_r = (
        toa_co2
        - piC_toa_reference
    )

    # CSIRO-Mk3-6-0 requires an additional two-year
    # running mean, as in the original analysis.
    if model == "CSIRO-Mk3-6-0":

        d_tas = (
            d_tas
            .rolling(
                year=2,
                center=True,
                min_periods=1,
            )
            .mean()
        )

        d_r = (
            d_r
            .rolling(
                year=2,
                center=True,
                min_periods=1,
            )
            .mean()
        )

    # MPI-ESM-LR is treated without the piControl
    # running-mean correction, as in the original analysis.
    elif model == "MPI-ESM-LR":

        d_tas = (
            tas_co2
            - tas_piC
        )

        d_r = (
            toa_co2
            - toa_piC
        )

    # ---------------------------------
    # Gregory regression
    # ---------------------------------

    d_tas.name = "d_tas"
    d_r.name = "d_r"

    lambda_, erf2x, ecs = gregory_regression(
        d_tas,
        d_r,
    )

    print(
        f"  ECS = {ecs.item():.2f} K | "
        f"ERF2x = {erf2x.item():.2f} W m-2 | "
        f"lambda = {lambda_.item():.2f} W m-2 K-1"
    )

    # ---------------------------------
    # Store model diagnostics
    # ---------------------------------

    ds_model = xr.merge(
        [
            d_tas,
            d_r,
            ecs,
            erf2x,
            lambda_,
        ]
    )

    ds_model = ds_model.assign_coords(
        model=model
    )

    if "height" in ds_model.coords:
        ds_model = ds_model.drop_vars(
            "height"
        )

    datasets.append(ds_model)


print("Gregory calculations completed.")


# =================================
# Combine model datasets
# =================================

ds_concat = xr.concat(
    datasets,
    dim="model",
)

ds_concat.attrs["description"] = (
    "Global annual-mean tas and TOA radiative imbalance "
    "anomalies for abrupt-4xCO2 experiments relative to "
    "piControl. PiControl anomalies are calculated using "
    "a 21-year running mean following Caldwell et al. "
    "(2016) and Zelinka et al. (2020). ECS, 2xCO2 ERF, "
    "and the net feedback parameter are derived from "
    "Gregory regression."
)

ds_concat.attrs["author"] = "O. Linke"
ds_concat.attrs["date"] = datetime.today().strftime(
    "%Y-%m-%d"
)

# =================================
# Save diagnostics
# =================================

os.makedirs(
    SAVE,
    exist_ok=True,
)

output_file = os.path.join(
    SAVE,
    f"ecs_feedbacks_erf_allmodels.nc",
)

ds_concat.to_netcdf(output_file)

print("Saved:")
print(f"  {output_file}")


#%%
# =================================
# Gregory overview plot
# =================================

ds = xr.open_dataset(
    output_file
).sortby("ecs")

fs = 20

n_models = len(ds.model)

# Six columns with enough rows for all models.
ncols = 6
nrows = int(np.ceil(n_models / ncols))

fig, axs = plt.subplots(
    nrows,
    ncols,
    figsize=(25, 4 * nrows),
    sharex=True,
    sharey=True,
)

axs = np.atleast_1d(axs).flatten()


for i, model in enumerate(ds.model.values):

    data = ds.sel(
        model=model
    )

    x = data.d_tas
    y = data.d_r

    ecs = data.ecs
    erf2x = data.erf2x
    lambda_ = data["lambda"]

    # Gregory regression line.
    x_fit = np.linspace(
        0,
        15,
        100,
    )

    y_fit = (
        lambda_.item() * x_fit
        + erf2x.item() * 2
    )

    axs[i].scatter(
        x,
        y,
        color="k",
        s=50,
    )

    axs[i].plot(
        x_fit,
        y_fit,
        color="C1",
    )

    text_lines = [
        f"$\\bf{{{model}}}$",
        (
            r"$F_{\mathrm{2xCO2}}$"
            f" = {erf2x.item():.2f} "
            r"$\mathrm{W\,m^{-2}}$"
        ),
        (
            rf"$\lambda = {lambda_.item():.2f}$ "
            r"$\mathrm{W\,m^{-2}\,K^{-1}}$"
        ),
        f"ECS = {ecs.item():.2f} K",
    ]

    axs[i].text(
        1.0,
        1.0,
        "\n".join(text_lines),
        transform=axs[i].transAxes,
        fontsize=12,
        ha="right",
        va="top",
        linespacing=1.0,
    )

    if i % ncols == 0:
        axs[i].set_ylabel(
            r"$N$ ($\mathrm{W\,m^{-2}}$)"
        )

    if i >= (nrows - 1) * ncols:
        axs[i].set_xlabel(
            r"$\Delta \overline{T}$ (K)"
        )


# Hide unused axes.
for ax in axs[n_models:]:
    ax.axis("off")


plt.xlim(
    0,
    12,
)

plt.ylim(
    0,
    8,
)

plt.tight_layout()
plt.show()
# %%
