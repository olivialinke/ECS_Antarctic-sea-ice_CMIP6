#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMIP6 sea-ice area and extent calculation
==========================================

This script computes Northern and Southern Hemisphere sea-ice extent (SIE) and sea-ice area (SIA) from preprocessed CMIP6 sea-ice concentration (siconc) data and corresponding grid-cell areas (areacello).

Requirement: siconc is already preprocessed via cmip_preprocessing.py

For each model, the script:

- Searches the DKRZ CMIP data pool for areacello files.
- Selects the first ensemble member (lowest realization number) for each model.
- Matches models with both siconc and areacello data.
- Applies model-specific coordinate adjustments where necessary.
- Computes SIE using a 15% sea-ice concentration threshold.
- Computes SIA as siconc × areacello.
- Separates SIE and SIA into Northern and Southern Hemisphere values.
- Combines all processed models into a single multi-model NetCDF dataset.

SIE and SIA are calculated on each model's native grid. No spatial regridding is performed.

The script is configured for the DKRZ file system and therefore requires access to the corresponding DKRZ CMIP data pool, or adaptation of the input paths to locally downloaded CMIP data.

Author: O. Linke
"""
#%%
# ====================================
# Imports
# ====================================
import glob
import os
import re
import shutil

import matplotlib.pyplot as plt
import xarray as xr

import warnings
warnings.filterwarnings("ignore")


# ====================================
# Configuration
# ====================================
EXP = "abrupt-4xCO2"
# DKRZ CMIP data pool
CMIP_ROOT = "/pool/data/CMIP6/data/CMIP/"
# Working/output directory
WORK_DIR = "/work/bb1036/b381106/data_output"
# Output directory for the final multi-model dataset
OUTPUT_DIR = os.path.join(WORK_DIR, "sea_ice")
# Set to True to produce diagnostic plots during processing
PLOT_DIAGNOSTICS = True

# ====================================
# Helper functions
# ====================================
def find_dirs(exp, table_id, variable):
    """
    Find CMIP6 files in the DKRZ data pool.

    Parameters
    ----------
    exp : str
        CMIP experiment, e.g. ``piControl``.
    table_id : str
        CMIP table identifier, e.g. ``Ofx``.
    variable : str
        Variable name, e.g. ``areacello``.

    Returns
    -------
    directories : list
        Full paths to matching files.
    filenames : list
        Matching filenames.
    models : list
        Unique model names.
    """
    search_path = (
        f"*/*/{exp}/*/{table_id}/{variable}/**/"
        f"{variable}_{table_id}_*_{exp}_*.nc"
    )

    directories = sorted(
        glob.glob(
            os.path.join(CMIP_ROOT, search_path),
            recursive=True,
        )
    )

    filenames = [
        os.path.basename(path)
        for path in directories
    ]

    models = sorted(
        {
            filename.split("_")[2]
            for filename in filenames
        }
    )

    print(models)
    print(f"Number of models: {len(models)}")

    return directories, filenames, models

def select_first_realization(directories, filenames):
    """
    Select the lowest realization number for each model.

    Parameters
    ----------
    directories : list
        Full paths to CMIP files.
    filenames : list
        Corresponding filenames.

    Returns
    -------
    dict
        Dictionary mapping each model to its selected file and
        realization number.
    """
    first_file = {}

    for path, filename in zip(directories, filenames):

        model = filename.split("_")[2]

        # Extract the realization number from identifiers such as
        # r1i1p1f1 or r10i1p1f1.
        match = re.search(r"r(\d+)i", filename)

        if match is None:
            continue

        realization = int(match.group(1))

        if (
            model not in first_file
            or realization < first_file[model][0]
        ):
            first_file[model] = (realization, path)

    return first_file

def get_models(variable, experiment):
    """
    Identify models with preprocessed data for a variable and experiment.
    """
    files = glob.glob(
        os.path.join(
            WORK_DIR,
            f"{variable}_*{experiment}*.nc"
        )
    )

    models = sorted(
        {
            os.path.basename(file).split("_")[2]
            for file in files
        }
    )

    print(
        f"{variable} -> {len(models)} models found"
    )

    return models

def plot_diagnostics(ds):
    """
    Plot annual-mean SIE and SIA diagnostics for one model.
    """
    if not PLOT_DIAGNOSTICS:
        return

    (
        ds.SIE_NH / 1e12
    ).mean(dim="month").plot(label="SIE NH")

    (
        ds.SIE_SH / 1e12
    ).mean(dim="month").plot(label="SIE SH")

    (
        ds.SIA_NH / 1e12
    ).mean(dim="month").plot(
        linestyle="--",
        label="SIA NH",
    )

    (
        ds.SIA_SH / 1e12
    ).mean(dim="month").plot(
        linestyle="--",
        label="SIA SH",
    )

    plt.legend()
    plt.show()
#%%
# ====================================
# Find and copy the first areacello file for each model
# ====================================
print("\nSearching for CMIP6 areacello files...")

table_id = "Ofx"
variable = "areacello"

directories, filenames, _ = find_dirs(
    EXP,
    table_id,
    variable,
)
first_files = select_first_realization(
    directories,
    filenames,
)
print(
    f"Selecting {len(first_files)} first-realization "
    "areacello files..."
)

os.makedirs(WORK_DIR, exist_ok=True)
available_area_files = glob.glob(
    os.path.join(WORK_DIR, "areacello_*.nc")
)
available_area_names = {
    os.path.basename(file)
    for file in available_area_files
}
#%%
for model, (_, source_file) in sorted(first_files.items()):

    filename = (
        "_".join(
            os.path.basename(source_file).split("_")[0:3]
        )
        + "_all-exps.nc"
    )
    destination = os.path.join(
        WORK_DIR,
        filename,
    )
    if filename in available_area_names:
        print(f"Already available: {filename}")
        continue

    shutil.copy2(
        source_file,
        destination,
    )
    print(f"Copied: {filename}")

#%%
# ====================================
# Identify models with both siconc and areacello
# ====================================
print("\nIdentifying models with complete sea-ice information...")

models_areacello = get_models(
    "areacello",
    "",
)
models_siconc = get_models(
    "siconc",
    EXP,
)
models = sorted(
    set(models_areacello).intersection(models_siconc)
)
print(
    f"{len(models)} common models found"
)

# ====================================
# Calculate SIE and SIA for each model
# ====================================
available_outputs = {
    os.path.basename(file)
    for file in glob.glob(
        os.path.join(
            WORK_DIR,
            f"SIE_SIA_*_{EXP}.nc",
        )
    )
}
#%%
for model in models:
    print(f"\nProcessing {model}")

    output_filename = (f"SIE_SIA_{model}_{EXP}.nc")
    output_path = os.path.join(
        WORK_DIR,
        output_filename,
    )
    if output_filename in available_outputs:
        print(f"Already processed: {model}")
        continue

    # ----------------------------------
    # Locate siconc and areacello files
    # ----------------------------------
    siconc_files = glob.glob(
        os.path.join(
            WORK_DIR,
            f"siconc_*_{model}_{EXP}_*.nc",
        )
    )
    area_files = glob.glob(
        os.path.join(
            WORK_DIR,
            f"areacello_*_{model}_all-exps.nc",
        )
    )
    if not siconc_files:
        print("No siconc data found")
        continue
    if not area_files:
        print("No areacello data found")
        continue

    siconc_file = siconc_files[0]
    areacello_file = area_files[0]

    # ----------------------------------
    # Open input data
    # ----------------------------------
    with xr.open_dataset(areacello_file) as ds_area:
        area = ds_area.areacello.load()
    with xr.open_dataset(siconc_file) as ds_sic:
        siconc = ds_sic.siconc.load()

    # ----------------------------------
    # Model-specific coordinate adjustments
    # ----------------------------------
    # These adjustments only rename dimensions so that the siconc and
    # areacello arrays can be combined on their native grids.
    if "CESM2" in model:
        print("  Adapting CESM2 dimension names")
        area = area.rename(
            {"nlat": "nj", "nlon": "ni"}
        )
    elif model == "CIESM":
        print("  Adapting CIESM dimension names")
        area = area.rename(
            {"j": "nj", "i": "ni"}
        )
    elif model == "GFDL-CM4":
        print("  Adapting GFDL-CM4 dimension names")
        area = area.rename(
            {"xh": "x", "yh": "y"}
        )
    elif model == "NESM3":
        print("  Adapting NESM3 dimension names")
        area = area.rename(
            {"i": "ni", "j": "nj"}
        )

    # ----------------------------------
    # Identify spatial dimensions
    # ----------------------------------
    spatial_dims = list(siconc.dims)
    spatial_dims.remove("year")
    spatial_dims.remove("month")

    print(f"  Coordinates: {list(siconc.coords)}")
    print(f"  Spatial dimensions: {spatial_dims}")
    
    # ----------------------------------
    # Identify Northern/Southern Hemisphere grid cells
    # ----------------------------------
    ice_mask = xr.where(siconc >= 15.0, 1, 0)
    
    if "latitude" in siconc.coords:
        latitude_coord = "latitude"
    elif "lat" in siconc.coords:
        latitude_coord = "lat"
    elif "nav_lat" in siconc.coords:
        latitude_coord = "nav_lat"
    else:
        raise ValueError(
            f"Could not identify latitude coordinate for {model}"
        )

    northern_hemisphere = (ice_mask[latitude_coord] > 0)
    southern_hemisphere = (ice_mask[latitude_coord] < 0)

    # ----------------------------------
    # Calculate Sea Ice Extent (SIE)
    # ----------------------------------
    # SIE is the sum of grid-cell areas for cells with at least 15%
    # sea-ice concentration.
    sie_nh = (
        ice_mask * area
    ).where(
        northern_hemisphere
    ).sum(
        dim=spatial_dims
    )
    sie_sh = (
        ice_mask * area
    ).where(
        southern_hemisphere
    ).sum(
        dim=spatial_dims
    )

    # ----------------------------------
    # Calculate Sea Ice Area (SIA)
    # ----------------------------------
    # siconc is provided in percent and is therefore divided by 100
    # before multiplying by grid-cell area.
    sia_nh = (
        (siconc / 100.0) * area
    ).where(
        northern_hemisphere
    ).sum(
        dim=spatial_dims
    )
    sia_sh = (
        (siconc / 100.0) * area
    ).where(
        southern_hemisphere
    ).sum(
        dim=spatial_dims
    )

    # ----------------------------------
    # Combine diagnostics
    # ----------------------------------
    sie_nh.name = "SIE_NH"
    sie_sh.name = "SIE_SH"
    sia_nh.name = "SIA_NH"
    sia_sh.name = "SIA_SH"

    ds = xr.merge(
        [
            sie_nh,
            sie_sh,
            sia_nh,
            sia_sh,
        ]
    )
    ds = ds.assign_coords(
        model=model
    )

    # ----------------------------------
    # Optional diagnostic plotting
    # ----------------------------------
    plot_diagnostics(ds)

    # ----------------------------------
    # Save standardized model output
    # ----------------------------------
    ds.to_netcdf(output_path)
    print(f"  Saved: {output_filename}")
    
#%%
# ====================================
# Combine all models into one dataset
# ====================================
print("\nCombining individual model files...")
files = sorted(
    glob.glob(
        os.path.join(
            WORK_DIR,
            f"SIE_SIA_*_{EXP}.nc",
        )
    )
)
datasets = []
for file in files:
    print(
        os.path.basename(file)
    )
    model = os.path.basename(file).split("_")[2]
    ds = (
        xr.open_dataset(file)
        .assign_coords(model=model)
        .load()
    )
    datasets.append(ds)

if not datasets:
    raise RuntimeError(
        "No individual SIE/SIA files were found."
    )
ds_all = xr.concat(datasets, dim="model")
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)
final_output = os.path.join(
    OUTPUT_DIR,
    f"SIE_SIA_{EXP}.nc",
)
ds_all.to_netcdf(final_output)
print(
    f"\nSaved multi-model dataset: {final_output}"
)
#%%
# ====================================
# Clean up intermediate model files
# ====================================
print("\nRemoving intermediate model files...")
for file in files:
    print(os.path.basename(file))
    os.remove(file)
print("\nDONE")
# %%
