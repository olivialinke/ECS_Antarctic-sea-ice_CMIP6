#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMIP5/CMIP6 preprocessing
=========================

This script preprocesses raw CMIP5/CMIP6 model output accessed from the
DKRZ CMIP data pool into standardized NetCDF files for subsequent feedback
and ECS analyses.

For a specified experiment, variable, and CMIP table, the script:

1. Searches the DKRZ CMIP data pool for all available model output.
2. Identifies the available models.
3. Selects the first ensemble member (lowest realization number) for each model.
4. Merges split NetCDF files into a continuous time series using CDO.
5. Retains the first 150 years (1800 monthly time steps).
6. Removes models with simulations shorter than 150 years.
7. Replaces the original model calendar with a standardized 365-day
   no-leap calendar covering years 1–150.
8. Saves one processed NetCDF file per model.

The script is configured for the DKRZ file system and therefore requires
access to the corresponding DKRZ CMIP data pool, or adaptation of the input
paths to locally downloaded CMIP data.

Output files follow the naming convention:

    <variable>_<table>_<model>_<experiment>_<variant>_<grid>.nc

Author: O. Linke
Created: January 2025
"""
#%%
# ===============================================
# Imports
# ===============================================
import glob
import os

import cftime
import numpy as np
import xarray as xr
from cdo import Cdo
cdo = Cdo() # Initialize CDO

import warnings
warnings.filterwarnings("ignore")

# ===============================================
# Helper functions
# ===============================================
def realization_number(realization):
    """
    Convert a realization identifier (e.g. 'r1i1p1f1') into an integer
    that can be used to identify the lowest/first ensemble member.

    Example
    -------
    'r1i1p1f1' -> 1111
    """
    digits = [str(i) for i in realization if i.isdigit()]
    return int("".join(digits))

def standardize_time(ds, start_year=1, end_year=150):
    """
    Replace the model time coordinate with a standardized monthly
    365-day no-leap calendar covering years 1–150.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    start_year : int
        First year of the standardized time coordinate.
    end_year : int
        Last year of the standardized time coordinate.

    Returns
    -------
    xarray.Dataset
        Dataset with the standardized time coordinate.
    """
    new_time = np.array(
        [
            cftime.DatetimeNoLeap(
                year, month, 1, 0, 0, 0, 0, has_year_zero=True
            )
            for year in range(start_year, end_year + 1)
            for month in range(1, 13)
        ],
        dtype=object,
    )

    return ds.assign_coords(time=new_time)

def time_to_year_month(ds):
    """
    Reshape a monthly time series into separate year and month coordinates.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing a monthly ``time`` coordinate.

    Returns
    -------
    xarray.Dataset
        Dataset with dimensions ``year`` and ``month``.
    """
    year = ds.time.dt.year.data
    month = ds.time.dt.month.data

    ds = ds.assign_coords(
        year=("time", year),
        month=("time", month),
    )

    ds = (
        ds.set_index(time=("year", "month"))
        .unstack("time")
    )

    return ds
# ===============================================
# Simulation input
# ===============================================
exp = "abrupt-4xCO2" # abrupt-4xCO2, piControl
table_id = "Amon" # SImon, Amon
var = "tas" # tas, ps, ta, hus, rsus, rsds, rlut, rlutcs, rsut, rsutcs, rsdt

# Output directory
OUT = "/path/to/data_output"
# set up your own file path here or use DKRZ CMIP data pool
ROOT = "/path/to/data/pool/"

# Search pattern for the requested experiment, table, and variable
path = (
    f"*/*/{exp}/*/{table_id}/{var}/**/"
    f"{var}_{table_id}_*_{exp}_*.nc"
)
# Change to the output directory
os.chdir(OUT)

# ===============================================
# Find available model files
# ===============================================
# Search recursively through the DKRZ data pool
directories = sorted(
    glob.glob(ROOT + path, recursive=True)
)
# Extract filenames from the matching paths
filenames = [
    os.path.basename(directory)
    for directory in directories
]
# Extract unique model names from the filenames
models = sorted(
    {
        filename.split("_")[2]
        for filename in filenames
    }
)

# # or only process one model:
# models = ['MPI-ESM1-2-LR']

print(models)
print(
    f"Number of models for {var} {exp}: "
    f"{len(models)}"
)

# ===============================================
# Preprocess each model
# ===============================================
print(f"\nSTARTING {var}")

# Keep track of files that have already been processed
already_existing = glob.glob(f"{var}*.nc")
# Count models removed because they contain fewer than 150 years
count_removed = 0

for model in models:
    print(f"\nProcessing {model}")

    # ------------------------------
    # Identify all files belonging to this model
    # ------------------------------
    file_load = []
    realizations = []

    for directory in directories:

        filename = os.path.basename(directory)

        # Skip files belonging to other models
        if model != filename.split("_")[2]:
            continue

        file_load.append(directory)
        realizations.append(filename.split("_")[4])

    # Keep only unique realization identifiers
    realizations = set(realizations)

    # ------------------------------
    # Select the first ensemble member
    # ------------------------------
    # Convert realization identifiers to sortable integers
    realization_numbers = [
        realization_number(realization)
        for realization in realizations
    ]
    # Sort realizations according to their numerical identifier
    sort_index = np.argsort(realization_numbers)
    sorted_realizations = np.array(list(realizations))[sort_index]

    # Select files belonging to the lowest/first realization number
    first_realization = sorted_realizations[0]

    first_run_files = [
        directory
        for directory in file_load
        if os.path.basename(directory).split("_")[4] == first_realization
    ]

    # ------------------------------
    # Construct output filename
    # ------------------------------
    new_filename = (
        "_".join(
            os.path.basename(first_run_files[0]).split("_")[0:6]
        )
        + ".nc"
    )
    # Skip processing if the standardized file already exists
    if new_filename in already_existing:
        print(f"{model} already in output directory")
        continue

    # ------------------------------
    # Check the start year of the source data
    # ------------------------------
    start_year = (
        xr.open_dataset(first_run_files[0])
        .time.dt.year[0]
        .data
    )
    print(
        f"{model}: first realization = {first_realization}, "
        f"start year = {start_year}"
    )

    # ------------------------------
    # Merge split NetCDF files
    # ------------------------------
    # Some CMIP simulations are stored in multiple files. CDO merges these
    # files into one continuous time series.
    cdo.mergetime(
        input=first_run_files,
        output=new_filename
    )
    # ------------------------------
    # Retain the first 150 years
    # ------------------------------
    ds = (
        xr.open_dataset(new_filename)
        .isel(time=slice(0, 150 * 12))
    )
    # ------------------------------
    # Check whether the model contains at least 150 years
    # ------------------------------

    if len(ds.time) < 1800:
        count_removed += 1
        print("❌ Not enough years - removing model")
        # Remove the temporary merged file
        os.remove(new_filename)

    else:
        # ------------------------------
        # Standardize the calendar and save the processed dataset
        # ------------------------------
        ds = standardize_time(ds)
        temporary_filename = f"{var}_{model}.nc"
        time_to_year_month(ds).to_netcdf(temporary_filename)
        
        print(
            "✅ Enough years - assigning standardized years 1–150"
        )
        # Rename the temporary file to the final standardized filename
        os.rename(
            temporary_filename,
            new_filename
        )
    # break

# ===============================================
# Finished
# ===============================================
print("\nFINISHED")
print(
    f"{count_removed} models removed due to short time series"
)
# %%
