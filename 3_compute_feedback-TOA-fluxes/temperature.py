#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate temperature feedbacks using radiative kernels
========================================================

This script calculates Planck and lapse-rate feedbacks from CMIP6
piControl and abrupt-4xCO2 atmospheric temperature and surface
temperature fields using multiple radiative kernels.

For each model and kernel, both all-sky and clear-sky Planck and
lapse-rate feedbacks are calculated.

Input files are preprocessed CMIP6 monthly data with separate `year`
and `month` dimensions.

The script is configured for the DKRZ file system. Users reproducing
the analysis on another system should adapt the input and output paths.

The calculations are run one model-kernel combination at a time in a
separate process to limit memory usage.
"""
#%%
# =================================
# Imports
# =================================
import gc
import glob
import os
import warnings
from multiprocessing import get_context

import climkern as ck
import numpy as np
import psutil
import xarray as xr

warnings.filterwarnings("ignore")

# =================================
# Paths and settings
# =================================

# Directory containing the preprocessed CMIP data.
DIR = "/path/to/data_output"
# Directory where the calculated feedback fields are saved.
SAVE = os.path.join(DIR, "feedback_toa_flux")

os.makedirs(SAVE, exist_ok=True)

# Radiative kernels used for the temperature-feedback calculation.
kernels = [
    "BMRC",
    # "CAM3",
    # "CAM5",
    # "CERES",
    # "CloudSat",
    # "ECHAM6",
    # "ECMWF-RRTM",
    # "ERA5",
    # "GFDL",
    # "HadGEM2",
    # "HadGEM3-GA7.1",
]


# Required variables for the temperature-feedback calculation.
REQ_TEMP = [
    ("ta", "piControl"),
    ("ta", "abrupt-4xCO2"),
    ("tas", "piControl"),
    ("tas", "abrupt-4xCO2"),
    ("ps", "piControl"),
    ("ps", "abrupt-4xCO2"),
]

# =================================
# Utility functions
# =================================
def all_inputs_present(data, keys):
    """
    Check whether all requested input variables are available as DataArrays.
    """
    return all(
        isinstance(data.get(key), xr.DataArray)
        for key in keys
    )

def read(var, model, exp):
    """
    Read a preprocessed CMIP6 variable for a given model and experiment.

    Parameters
    ----------
    var : str
        CMIP6 variable name.
    model : str
        CMIP6 model name.
    exp : str
        Experiment name.

    Returns
    -------
    xarray.DataArray or float
        Requested variable, or NaN if no matching file is found.
    """

    filename = glob.glob(
        f"{DIR}/{var}_Amon_{model}_{exp}_*.nc"
    )

    if not filename:
        return np.nan

    ds = xr.open_dataset(
        filename[0],
        chunks={
            "year": 150,
            "month": 12,
            "plev": 19,
            "lat": 60,
            "lon": 90,
        },
    )

    return ds[var]

def common_model_list(directory, var="tas"):
    """
    Return the models available for a given preprocessed variable.
    """

    files = glob.glob(
        f"{directory}/{var}_*.nc"
    )

    models = sorted(
        set(
            os.path.basename(file).split("_")[2]
            for file in files
        )
    )

    return models

# =================================
# Memory reporting
# =================================
_PROC = psutil.Process(os.getpid())


def print_mem(tag=""):
    """
    Print the current process memory usage.
    """

    rss = _PROC.memory_info().rss / (1024**3)

    label = f" {tag}" if tag else ""

    print(
        f"  MEM{label}: {rss:.1f} GB"
    )

# =================================
# Temperature-feedback calculation
# =================================
def compute_temperature_feedbacks(model, kernel):
    """
    Calculate all-sky and clear-sky Planck and lapse-rate feedbacks.

    Parameters
    ----------
    model : str
        CMIP6 model name.
    kernel : str
        Radiative kernel name.

    Notes
    -----
    The calculation uses atmospheric temperature (ta), surface air
    temperature (tas), and surface pressure (ps) from piControl and
    abrupt-4xCO2 simulations.
    """

    fn_planck = f"Planck_{model}_{kernel}.nc"
    fn_lr = f"LR_{model}_{kernel}.nc"
    fn_planck_cs = f"Planck_cs_{model}_{kernel}.nc"
    fn_lr_cs = f"LR_cs_{model}_{kernel}.nc"

    output_files = {
        os.path.basename(file)
        for file in glob.glob(f"{SAVE}/*.nc")
    }

    # Skip the calculation if all outputs already exist.
    if all(
        file in output_files
        for file in [
            fn_planck,
            fn_lr,
            fn_planck_cs,
            fn_lr_cs,
        ]
    ):
        print(
            f"[{model} {kernel}] already done"
        )
        return

    # --------------------------
    # Read required input fields
    # --------------------------

    data = {
        key: read(var, model, exp)
        for key in REQ_TEMP
        for var, exp in [key]
    }

    if not all_inputs_present(data, REQ_TEMP):
        print(
            f"[{model} {kernel}] missing input data"
        )
        return

    print(
        f"[{model} {kernel}] START"
    )
    print_mem("after data read")

    # --------------------------
    # All-sky temperature feedbacks
    # --------------------------

    if fn_planck not in output_files or fn_lr not in output_files:

        print(
            "  calculating all-sky temperature feedbacks"
        )

        LR, Planck = ck.calc_T_feedbacks(
            data[("ta", "piControl")],
            data[("tas", "piControl")],
            data[("ps", "piControl")],
            data[("ta", "abrupt-4xCO2")],
            data[("tas", "abrupt-4xCO2")],
            data[("ps", "abrupt-4xCO2")],
            kern=kernel,
        )

        Planck.name = "Planck"
        Planck = Planck.rename({"time": "month"})
        Planck = Planck.astype("float32")

        Planck.to_netcdf(
            f"{SAVE}/{fn_planck}"
        )

        LR.name = "LR"
        LR = LR.rename({"time": "month"})
        LR = LR.astype("float32")

        LR.to_netcdf(
            f"{SAVE}/{fn_lr}"
        )

        del LR, Planck
        gc.collect()

        print_mem("after all-sky")

    # --------------------------
    # Clear-sky temperature feedbacks
    # --------------------------

    if fn_planck_cs not in output_files or fn_lr_cs not in output_files:

        print(
            "  calculating clear-sky temperature feedbacks"
        )

        LR_cs, Planck_cs = ck.calc_T_feedbacks(
            data[("ta", "piControl")],
            data[("tas", "piControl")],
            data[("ps", "piControl")],
            data[("ta", "abrupt-4xCO2")],
            data[("tas", "abrupt-4xCO2")],
            data[("ps", "abrupt-4xCO2")],
            kern=kernel,
            sky="clear-sky",
        )

        Planck_cs.name = "Planck_cs"
        Planck_cs = Planck_cs.rename({"time": "month"})
        Planck_cs = Planck_cs.astype("float32")

        Planck_cs.to_netcdf(
            f"{SAVE}/{fn_planck_cs}"
        )

        LR_cs.name = "LR_cs"
        LR_cs = LR_cs.rename({"time": "month"})
        LR_cs = LR_cs.astype("float32")

        LR_cs.to_netcdf(
            f"{SAVE}/{fn_lr_cs}"
        )

        del LR_cs, Planck_cs
        gc.collect()

        print_mem("after clear-sky")

    # --------------------------
    # Clean up
    # --------------------------

    del data
    gc.collect()

    print_mem("after kernel")

    print(
        f"[{model} {kernel}] done"
    )

#%%
# =================================
# Run calculations
# =================================
if __name__ == "__main__":

    # Determine the models available in the preprocessed dataset.
    models = common_model_list(
        DIR,
        var="tas",
    )
    # Create all model-kernel combinations.
    tasks = [
        (model, kernel)
        for model in models
        for kernel in kernels
    ]
    # Run one model-kernel combination at a time.
    #
    # Each calculation is executed in a separate process to limit memory
    # accumulation and ensure that memory is released when the process ends.
    ctx = get_context("spawn")

    for model, kernel in tasks:

        process = ctx.Process(
            target=compute_temperature_feedbacks,
            args=(model, kernel),
        )

        process.start()

        # Allow a maximum of 30 minutes for each model-kernel calculation.
        process.join(
            timeout=1800
        )

        if process.is_alive():

            print(
                f"WARNING: Process {model}-{kernel} exceeded "
                "the 30-minute limit and will be terminated."
            )

            process.terminate()
            process.join()

    print(
        "All temperature-feedback calculations completed."
    )
# %%
