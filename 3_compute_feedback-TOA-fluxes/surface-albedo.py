#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate surface-albedo feedback using radiative kernels
==========================================================

This script calculates surface-albedo feedback from CMIP6 piControl and
abrupt-4xCO2 surface radiative fluxes using multiple radiative kernels.

For each model and kernel, both all-sky and clear-sky surface-albedo
feedbacks are calculated from surface upwelling and downwelling shortwave
radiative fluxes (rsus and rsds).

Input files are preprocessed CMIP6 monthly data with separate `year` and
`month` dimensions.

The script is configured for the DKRZ file system. Users reproducing the
analysis on another system should adapt the input and output paths.

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

# Directory containing the preprocessed CMIP6 radiative fluxes.
DIR = "/work/bb1036/b381106/data_output"

# Directory where the calculated feedback fields are saved.
SAVE = os.path.join(DIR, "feedback_toa_flux")

os.makedirs(SAVE, exist_ok=True)


# Radiative kernels used for the surface-albedo feedback calculation.
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


# Required surface shortwave radiative fluxes.
REQ_SA = [
    ("rsus", "piControl"),
    ("rsus", "abrupt-4xCO2"),
    ("rsds", "piControl"),
    ("rsds", "abrupt-4xCO2"),
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
# Surface-albedo feedback calculation
# =================================

def compute_surface_albedo_feedback(model, kernel):
    """
    Calculate all-sky and clear-sky surface-albedo feedback.

    Parameters
    ----------
    model : str
        CMIP6 model name.
    kernel : str
        Radiative kernel name.

    Notes
    -----
    The calculation uses surface upwelling (rsus) and downwelling (rsds)
    shortwave radiative fluxes from piControl and abrupt-4xCO2 simulations.
    """

    fn_sa = f"SA_{model}_{kernel}.nc"
    fn_sa_cs = f"SA_cs_{model}_{kernel}.nc"

    output_files = {
        os.path.basename(file)
        for file in glob.glob(f"{SAVE}/*.nc")
    }

    # Skip the calculation if both outputs already exist.
    if fn_sa in output_files and fn_sa_cs in output_files:
        print(
            f"[{model} {kernel}] already done"
        )
        return

    # --------------------------
    # Read required radiative fluxes
    # --------------------------

    data = {
        key: read(var, model, exp)
        for var, exp in REQ_SA
        for key in [(var, exp)]
    }

    if not all_inputs_present(data, REQ_SA):
        print(
            f"[{model} {kernel}] missing input data"
        )
        return

    print(
        f"[{model} {kernel}] START"
    )
    print_mem("after data read")

    # --------------------------
    # All-sky surface-albedo feedback
    # --------------------------

    if fn_sa not in output_files:

        print(
            "  calculating all-sky surface-albedo feedback"
        )

        SA = ck.calc_alb_feedback(
            data[("rsus", "piControl")],
            data[("rsds", "piControl")],
            data[("rsus", "abrupt-4xCO2")],
            data[("rsds", "abrupt-4xCO2")],
            kern=kernel,
        )

        SA.name = "SA"

        # The kernel returns monthly feedback values. Store the temporal
        # coordinate as month in the output file.
        SA = SA.rename({"time": "month"})
        SA = SA.astype("float32")

        SA.to_netcdf(
            f"{SAVE}/{fn_sa}"
        )

        del SA
        gc.collect()

        print_mem("after all-sky")

    # --------------------------
    # Clear-sky surface-albedo feedback
    # --------------------------

    if fn_sa_cs not in output_files:

        print(
            "  calculating clear-sky surface-albedo feedback"
        )

        SA_cs = ck.calc_alb_feedback(
            data[("rsus", "piControl")],
            data[("rsds", "piControl")],
            data[("rsus", "abrupt-4xCO2")],
            data[("rsds", "abrupt-4xCO2")],
            kern=kernel,
            sky="clear-sky",
        )

        SA_cs.name = "SA_cs"

        # The kernel returns monthly feedback values. Store the temporal
        # coordinate as month in the output file.
        SA_cs = SA_cs.rename({"time": "month"})
        SA_cs = SA_cs.astype("float32")

        SA_cs.to_netcdf(
            f"{SAVE}/{fn_sa_cs}"
        )

        del SA_cs
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
            target=compute_surface_albedo_feedback,
            args=(model, kernel),
        )

        process.start()

        # Allow a maximum of 10 minutes for each model-kernel calculation.
        process.join(
            timeout=600
        )

        if process.is_alive():

            print(
                f"WARNING: Process {model}-{kernel} exceeded "
                "the 10-minute limit and will be terminated."
            )

            process.terminate()
            process.join()

    print(
        "All surface-albedo feedback calculations completed."
    )
# %%
