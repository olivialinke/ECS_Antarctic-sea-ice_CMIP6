#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate spatially resolved climate feedbacks using Gregory regression
=============================

This script calculates spatially resolved climate feedback parameters from CMIP6 piControl and abrupt-4xCO2 simulations.

For each model, the Gregory regression is performed independently at each grid cell, using the global-annual-mean surface temperature anomaly as the predictor and the local radiative response as the dependent variable.

The script calculates:

- Net, shortwave, and longwave feedback parameters
- Clear-sky shortwave and longwave feedback parameters
- Surface-albedo, Planck, lapse-rate, and water-vapour feedbacks for multiple radiative kernels
- Shortwave and longwave cloud feedbacks using the adjustment method

The calculations use preprocessed monthly CMIP6 data with separate `year` and `month` dimensions.

The script is configured for the DKRZ file system. Users reproducing the analysis on another system should adapt the input and output paths.

The spatially resolved regressions are parallelized over models using multiprocessing.
"""

#%%
# =================================
# Imports
# =================================

import glob
import os
import warnings
from multiprocessing import Pool

import climkern as ck
import numpy as np
import xarray as xr
from scipy.stats import linregress

warnings.filterwarnings("ignore")


#%%
# =================================
# Paths and settings
# =================================

# Directory containing the preprocessed CMIP6 diagnostics.
DIR = "/work/bb1036/b381106/data_output"

# Directory containing the kernel-derived TOA feedback fields.
FDBFLX_DIR = os.path.join(DIR, "feedback_toa_flux")

# Directory where the spatially resolved feedback parameters are saved.
SAVE = os.path.join(DIR, "feedbacks")

os.makedirs(SAVE, exist_ok=True)


# Radiative kernels used for the feedback calculation.
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


# Feedback variables calculated from radiative kernels.
FB_VARS = [
    "SA",
    "Planck",
    "LR",
    "WV",
]

FB_VARNAMES = [
    "SAF",
    "PF",
    "LRF",
    "WVF",
]


#%%
# =================================
# Utility functions
# =================================

def linregress_slope_intercept(x, y):
    """
    Compute a linear regression between two 1D arrays.

    Parameters
    ----------
    x, y : array-like
        Independent and dependent variables.

    Returns
    -------
    slope, intercept, r_value, p_value, std_err : float
        Components of scipy.stats.linregress.
    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, intercept, r_value, p_value, std_err

def read(var, model, exp, directory):
    """
    Read a preprocessed CMIP6 variable.

    Parameters
    ----------
    var : str
        Variable name, e.g. ``tas`` or ``rsut``.
    model : str
        CMIP6 model name.
    exp : str
        Experiment name, e.g. ``piControl`` or ``abrupt-4xCO2``.
    directory : str
        Directory containing the preprocessed files.

    Returns
    -------
    xr.Dataset or None
        Open dataset if a matching file is found, otherwise None.
    """
    pattern = f"{directory}/{var}_Amon_{model}_{exp}_*.nc"
    files = glob.glob(pattern)

    if not files:
        print(f"  WARNING: Missing {var} for {model} ({exp})")
        return None

    try:
        return xr.open_dataset(files[0])
    except Exception as exc:
        print(f"  WARNING: Could not open {files[0]}: {exc}")
        return None

def get_dataarray(var, model, exp, directory):
    """
    Read a variable and return it as an xarray.DataArray.

    Returns None if the required file or variable is unavailable.
    """
    ds = read(var, model, exp, directory)

    if ds is None:
        return None

    if var not in ds:
        print(f"  WARNING: Variable {var} not found in dataset.")
        ds.close()
        return None

    return ds[var]

def gam(ds):
    """
    Compute the global-annual mean.

    Spatial averaging uses cosine-of-latitude weighting. If a monthly
    dimension is present after spatial averaging, the monthly values
    are averaged to obtain an annual mean.

    Parameters
    ----------
    ds : xr.DataArray or xr.Dataset
        Input data containing ``lat`` and ``lon`` dimensions.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Global-annual mean.
    """
    if "lat" not in ds.dims or "lon" not in ds.dims:
        raise ValueError(
            "Input data must contain 'lat' and 'lon' dimensions."
        )

    weights = np.cos(np.deg2rad(ds.lat))

    wmean = ds.weighted(weights).mean(dim=["lat", "lon"])

    if "month" in wmean.dims:
        wmean = wmean.mean(dim="month")

    return wmean

def common_model_list(directory, variables):
    """
    Find models that provide all requested preprocessed variables.

    Parameters
    ----------
    directory : str
        Directory containing the preprocessed NetCDF files.
    variables : list of str
        Variables required for the calculation.

    Returns
    -------
    list of str
        Sorted list of models available for all requested variables.
    """
    common_models = None

    for variable in variables:

        models = {
            filename.split("_")[2]
            for filename in os.listdir(directory)
            if filename.endswith(".nc")
            and f"{variable}_" in filename
        }

        print(f"{variable}: {len(models)} models found")

        common_models = (
            models
            if common_models is None
            else common_models & models
        )

    if not common_models:
        print("WARNING: No common models found.")
        return []

    print(f"{len(common_models)} models found")

    return sorted(common_models)

def regression_grid(x, y):
    """
    Perform a linear regression along the year dimension.

    The regression is vectorized over all remaining dimensions, so that
    a separate regression is calculated at each grid cell.

    Parameters
    ----------
    x : xr.DataArray
        Global-annual-mean temperature anomaly with dimension ``year``.
    y : xr.DataArray
        Radiative response with dimension ``year`` and spatial dimensions.

    Returns
    -------
    tuple of xr.DataArray
        Slope, intercept, r-value, p-value, and standard error.
    """
    return xr.apply_ufunc(
        linregress_slope_intercept,
        x,
        y,
        input_core_dims=[["year"], ["year"]],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float, float, float],
    )

#%%
# =================================
# Spatially resolved feedbacks
# =================================
def compute_feedbacks(model):
    """
    Calculate spatially resolved feedback parameters for one model.

    The Gregory regression is performed at every grid cell using the
    global-annual-mean temperature anomaly as the predictor.
    """

    print(f"\n========== MODEL: {model} ==========")

    savefile = f"feedbacks_{model}.nc"
    savepath = os.path.join(SAVE, savefile)

    # ---------------------------------
    # Read surface temperature
    # ---------------------------------

    tas_piC = get_dataarray("tas", model, "piControl", DIR)
    tas_co2 = get_dataarray("tas", model, "abrupt-4xCO2", DIR)

    if tas_piC is None or tas_co2 is None:
        print(f"  WARNING: Missing tas for {model}. Skipping.")
        return

    # Local temperature anomaly.
    d_tas = tas_co2 - tas_piC

    # Global-annual-mean temperature anomaly used as the Gregory
    # regression predictor.
    x = gam(d_tas)

    # ---------------------------------
    # Read TOA radiative fluxes
    # ---------------------------------

    rsut_piC = get_dataarray("rsut", model, "piControl", DIR)
    rsut_co2 = get_dataarray("rsut", model, "abrupt-4xCO2", DIR)

    rlut_piC = get_dataarray("rlut", model, "piControl", DIR)
    rlut_co2 = get_dataarray("rlut", model, "abrupt-4xCO2", DIR)

    rsutcs_piC = get_dataarray("rsutcs", model, "piControl", DIR)
    rsutcs_co2 = get_dataarray("rsutcs", model, "abrupt-4xCO2", DIR)

    rlutcs_piC = get_dataarray("rlutcs", model, "piControl", DIR)
    rlutcs_co2 = get_dataarray("rlutcs", model, "abrupt-4xCO2", DIR)

    required = [
        rsut_piC,
        rsut_co2,
        rlut_piC,
        rlut_co2,
        rsutcs_piC,
        rsutcs_co2,
        rlutcs_piC,
        rlutcs_co2,
    ]

    if any(data is None for data in required):
        print(f"  WARNING: Missing TOA flux data for {model}. Skipping.")
        return

    # ---------------------------------
    # Compute radiative perturbations
    # ---------------------------------
    #
    # Positive downward:
    #
    #   dR_SW = -(ΔRSUT)
    #   dR_LW = -(ΔRLUT)
    #   dR_net = dR_SW + dR_LW
    #
    # The perturbations retain their monthly and spatial dimensions.
    # The regression below is performed independently at each grid cell.

    d_r = (
        -(rsut_co2 - rsut_piC)
        -(rlut_co2 - rlut_piC)
    )

    d_r_sw = -(rsut_co2 - rsut_piC)
    d_r_lw = -(rlut_co2 - rlut_piC)

    d_r_sw_cs = -(rsutcs_co2 - rsutcs_piC)
    d_r_lw_cs = -(rlutcs_co2 - rlutcs_piC)

    # ---------------------------------
    # Gregory regressions
    # ---------------------------------

    print("  Calculating net feedback parameters")

    net, erf, _, _, _ = regression_grid(x, d_r)

    net_sw, erf_sw, _, _, _ = regression_grid(x, d_r_sw)

    net_lw, erf_lw, _, _, _ = regression_grid(x, d_r_lw)

    net_sw_cs, erf_sw_cs, _, _, _ = regression_grid(
        x, d_r_sw_cs
    )

    net_lw_cs, erf_lw_cs, _, _, _ = regression_grid(
        x, d_r_lw_cs
    )

    # ---------------------------------
    # Pack net feedback diagnostics
    # ---------------------------------

    ds_net = xr.Dataset(
        {
            "erf": erf,
            "net": net,
            "erf_sw": erf_sw,
            "erf_lw": erf_lw,
            "erf_sw_cs": erf_sw_cs,
            "erf_lw_cs": erf_lw_cs,
            "net_sw": net_sw,
            "net_lw": net_lw,
            "net_sw_cs": net_sw_cs,
            "net_lw_cs": net_lw_cs,

            # Temperature anomaly averaged over the final 20 years.
            "dT": d_tas.sel(year=slice(131, 150)).mean(dim="year"),
        }
    )

    # =================================
    # Kernel-derived feedbacks
    # =================================

    print("\n  Calculating kernel-derived feedbacks")

    all_fb_datasets = []

    for fb_var, fb_name in zip(FB_VARS, FB_VARNAMES):

        print(f"\n  ---> {fb_var}")

        kernel_datasets = []

        for k_count, kernel in enumerate(kernels):

            print(
                f"    {kernel} "
                f"({k_count + 1}/{len(kernels)})"
            )

            file_as = os.path.join(
                FDBFLX_DIR,
                f"{fb_var}_{model}_{kernel}.nc"
            )

            file_cs = os.path.join(
                FDBFLX_DIR,
                f"{fb_var}_cs_{model}_{kernel}.nc"
            )

            # ---------------------------------
            # Check input files
            # ---------------------------------

            if not (
                os.path.exists(file_as)
                and os.path.exists(file_cs)
            ):
                print(
                    f"      WARNING: Missing {fb_var} "
                    f"for {kernel} → NaN"
                )

                # Create a spatial NaN field based on the first
                # available temperature field.
                template = d_tas.isel(year=0).drop_vars(
                    "year",
                    errors="ignore"
                )

                fb_as = xr.full_like(template, np.nan)
                fb_cs = xr.full_like(template, np.nan)

            else:

                dR_fdb_as = xr.open_dataset(file_as)[fb_var]
                dR_fdb_cs = xr.open_dataset(file_cs)[
                    f"{fb_var}_cs"
                ]

                # ---------------------------------
                # Local Gregory regression
                # ---------------------------------

                fb_as, _, _, _, _ = regression_grid(
                    x, dR_fdb_as
                )

                fb_cs, _, _, _, _ = regression_grid(
                    x, dR_fdb_cs
                )

            ds_kernel = xr.Dataset(
                {
                    fb_name: fb_as,
                    f"{fb_name}_cs": fb_cs,
                }
            ).expand_dims(kernel=[kernel])

            kernel_datasets.append(ds_kernel)

        # Concatenate all kernels for this feedback.
        ds_fb = xr.concat(
            kernel_datasets,
            dim="kernel"
        )

        all_fb_datasets.append(ds_fb)

    # Merge all kernel-derived feedbacks.
    ds_feedbacks = xr.merge(all_fb_datasets)

    # ---------------------------------
    # Final model dataset
    # ---------------------------------

    ds_final = xr.merge(
        [
            ds_net,
            ds_feedbacks,
        ]
    )

    # Add model coordinate.
    ds_final = ds_final.assign_coords(model=model)

    # ---------------------------------
    # Diagnostics
    # ---------------------------------

    print("\n  Global mean feedback parameters:")

    for var in ["net", "net_sw", "net_lw"]:
        value = np.round(
            gam(ds_final[var]).data,
            2
        )
        print(f"    {var} = {value}")

    for var in ["PF", "LRF", "SAF", "WVF"]:
        value = np.round(
            gam(ds_final[var]).data,
            2
        )
        print(f"    {var} = {value}")

    # ---------------------------------
    # Save
    # ---------------------------------

    print(f"\n  Saving: {savepath}")

    ds_final.to_netcdf(savepath)

    print(f"  Finished {model}")

#%%
# =================================
# Spatially resolved cloud feedbacks
# =================================
def compute_cloud_feedbacks(model):
    """
    Calculate spatially resolved shortwave and longwave cloud feedbacks.

    Cloud feedbacks are derived using the adjustment method implemented
    in climkern. The cloud-induced TOA radiative response is regressed
    against the global-annual-mean temperature anomaly at each grid cell.
    """

    print(f"\n========== CLOUDS: {model} ==========")

    # ---------------------------------
    # Read previously calculated feedbacks
    # ---------------------------------

    feedback_file = os.path.join(
        SAVE,
        f"feedbacks_{model}.nc"
    )

    if not os.path.exists(feedback_file):
        print(
            f"  WARNING: {feedback_file} not found. "
            "Skipping cloud feedbacks."
        )
        return

    ds_feedbacks = xr.open_dataset(feedback_file)

    # ---------------------------------
    # Read surface temperature
    # ---------------------------------

    tas_piC = get_dataarray("tas", model, "piControl", DIR)
    tas_co2 = get_dataarray("tas", model, "abrupt-4xCO2", DIR)

    if tas_piC is None or tas_co2 is None:
        print(f"  WARNING: Missing tas for {model}. Skipping.")
        return

    d_tas = tas_co2 - tas_piC

    # Global-annual-mean temperature anomaly.
    x = gam(d_tas)

    # ---------------------------------
    # Read SW TOA fluxes
    # ---------------------------------

    rsut_piC = get_dataarray("rsut", model, "piControl", DIR)
    rsut_co2 = get_dataarray("rsut", model, "abrupt-4xCO2", DIR)

    rsdt_piC = get_dataarray("rsdt", model, "piControl", DIR)
    rsdt_co2 = get_dataarray("rsdt", model, "abrupt-4xCO2", DIR)

    rsutcs_piC = get_dataarray("rsutcs", model, "piControl", DIR)
    rsutcs_co2 = get_dataarray("rsutcs", model, "abrupt-4xCO2", DIR)

    required_sw = [
        rsut_piC,
        rsut_co2,
        rsdt_piC,
        rsdt_co2,
        rsutcs_piC,
        rsutcs_co2,
    ]

    if any(data is None for data in required_sw):
        print(
            f"  WARNING: Missing SW cloud input data "
            f"for {model}."
        )
        return

    # Net SW TOA flux, positive downward.
    rsnt_piC = rsdt_piC - rsut_piC
    rsnt_co2 = rsdt_co2 - rsut_co2

    rsntcs_piC = rsdt_piC - rsutcs_piC
    rsntcs_co2 = rsdt_co2 - rsutcs_co2

    # Cloud-radiative-effect anomaly.
    dCRE_SW = ck.calc_dCRE_SW(
        rsnt_piC,
        rsnt_co2,
        rsntcs_piC,
        rsntcs_co2,
    )

    # ---------------------------------
    # Read LW TOA fluxes
    # ---------------------------------

    rlut_piC = get_dataarray("rlut", model, "piControl", DIR)
    rlut_co2 = get_dataarray("rlut", model, "abrupt-4xCO2", DIR)

    rlutcs_piC = get_dataarray("rlutcs", model, "piControl", DIR)
    rlutcs_co2 = get_dataarray("rlutcs", model, "abrupt-4xCO2", DIR)

    required_lw = [
        rlut_piC,
        rlut_co2,
        rlutcs_piC,
        rlutcs_co2,
    ]

    if any(data is None for data in required_lw):
        print(
            f"  WARNING: Missing LW cloud input data "
            f"for {model}."
        )
        return

    dCRE_LW = ck.calc_dCRE_LW(
        rlut_piC,
        rlut_co2,
        rlutcs_piC,
        rlutcs_co2,
    )

    # ---------------------------------
    # ERF components
    # ---------------------------------

    erf_sw = ds_feedbacks.erf_sw
    erf_sw_cs = ds_feedbacks.erf_sw_cs
    erf_lw = ds_feedbacks.erf_lw
    erf_lw_cs = ds_feedbacks.erf_lw_cs

    # ---------------------------------
    # Calculate cloud feedbacks
    # ---------------------------------

    kernel_datasets = []

    for k_count, kernel in enumerate(kernels):

        print(
            f"  {kernel} "
            f"({k_count + 1}/{len(kernels)})"
        )

        # Input feedback fields.
        sa_file = os.path.join(
            FDBFLX_DIR,
            f"SA_{model}_{kernel}.nc"
        )

        sa_cs_file = os.path.join(
            FDBFLX_DIR,
            f"SA_cs_{model}_{kernel}.nc"
        )

        wv_file = os.path.join(
            FDBFLX_DIR,
            f"WV_{model}_{kernel}.nc"
        )

        wv_cs_file = os.path.join(
            FDBFLX_DIR,
            f"WV_cs_{model}_{kernel}.nc"
        )

        lr_file = os.path.join(
            FDBFLX_DIR,
            f"LR_{model}_{kernel}.nc"
        )

        lr_cs_file = os.path.join(
            FDBFLX_DIR,
            f"LR_cs_{model}_{kernel}.nc"
        )

        planck_file = os.path.join(
            FDBFLX_DIR,
            f"Planck_{model}_{kernel}.nc"
        )

        planck_cs_file = os.path.join(
            FDBFLX_DIR,
            f"Planck_cs_{model}_{kernel}.nc"
        )

        # =================================
        # Shortwave cloud feedback
        # =================================

        sw_files = [
            sa_file,
            sa_cs_file,
            wv_file,
            wv_cs_file,
        ]

        if all(os.path.exists(file) for file in sw_files):

            SA = xr.open_dataset(sa_file)["SA"]
            SA_cs = xr.open_dataset(sa_cs_file)["SA_cs"]
            WV = xr.open_dataset(wv_file)["WV"]
            WV_cs = xr.open_dataset(wv_cs_file)["WV_cs"]

            dR_cloud_SW = ck.calc_cloud_SW(
                SA,
                SA_cs,
                WV,
                WV_cs,
                dCRE_SW,
                erf_sw,
                erf_sw_cs,
            )

            CF_SW, _, _, _, _ = regression_grid(
                x,
                dR_cloud_SW,
            )

        else:

            print(
                "    WARNING: Missing SA/WV files "
                "→ CF_SW = NaN"
            )

            template = d_tas.isel(year=0).drop_vars(
                "year",
                errors="ignore"
            )

            CF_SW = xr.full_like(
                template,
                np.nan
            )

        # =================================
        # Longwave cloud feedback
        # =================================

        lw_files = [
            lr_file,
            lr_cs_file,
            planck_file,
            planck_cs_file,
            wv_file,
            wv_cs_file,
        ]

        if all(os.path.exists(file) for file in lw_files):

            LR = xr.open_dataset(lr_file)["LR"]
            LR_cs = xr.open_dataset(lr_cs_file)["LR_cs"]

            Planck = xr.open_dataset(
                planck_file
            )["Planck"]

            Planck_cs = xr.open_dataset(
                planck_cs_file
            )["Planck_cs"]

            WV = xr.open_dataset(wv_file)["WV"]
            WV_cs = xr.open_dataset(wv_cs_file)["WV_cs"]

            # Total non-cloud LW feedback response.
            t_as = LR + Planck
            t_cs = LR_cs + Planck_cs

            dR_cloud_LW = ck.calc_cloud_LW(
                t_as,
                t_cs,
                WV,
                WV_cs,
                dCRE_LW,
                erf_lw,
                erf_lw_cs,
            )

            CF_LW, _, _, _, _ = regression_grid(
                x,
                dR_cloud_LW,
            )

        else:

            print(
                "    WARNING: Missing LR/Planck/WV files "
                "→ CF_LW = NaN"
            )

            template = d_tas.isel(year=0).drop_vars(
                "year",
                errors="ignore"
            )

            CF_LW = xr.full_like(
                template,
                np.nan
            )

        # ---------------------------------
        # Combine SW and LW
        # ---------------------------------

        ds_kernel = xr.Dataset(
            {
                "CF_SW": CF_SW,
                "CF_LW": CF_LW,
            }
        ).expand_dims(kernel=[kernel])

        kernel_datasets.append(ds_kernel)

    # ---------------------------------
    # Merge kernels
    # ---------------------------------

    ds_cloud = xr.concat(
        kernel_datasets,
        dim="kernel"
    )

    # ---------------------------------
    # Save
    # ---------------------------------

    savefile = os.path.join(
        SAVE,
        f"cloud-feedbacks_{model}.nc"
    )

    print(f"\n  Saving: {savefile}")

    ds_cloud.to_netcdf(savefile)

    print(f"  Finished cloud feedbacks for {model}")

#%%
# =================================
# Run calculations
# =================================

if __name__ == "__main__":

    models = common_model_list(
        DIR,
        ["tas", "rsut", "rlut"]
    )

    print(
        f"\nRunning spatially resolved feedback "
        f"calculations for {len(models)} models."
    )

    # ---------------------------------
    # Part 1: feedback parameters
    # ---------------------------------

    with Pool(8) as pool:
        pool.map(
            compute_feedbacks,
            models
        )

    # ---------------------------------
    # Part 2: cloud feedbacks
    # ---------------------------------

    with Pool(8) as pool:
        pool.map(
            compute_cloud_feedbacks,
            models
        )

    print("\nAll spatially resolved feedback calculations completed.")
    
#%%
# =================================
# Merge feedback and cloud datasets
# =================================

def merge_model_feedbacks(model):
    """Merge feedback and cloud-feedback datasets for one model."""

    feedback_file = os.path.join(
        SAVE,
        f"feedbacks_{model}.nc",
    )

    cloud_file = os.path.join(
        SAVE,
        f"cloud-feedbacks_{model}.nc",
    )

    merged_file = os.path.join(
        SAVE,
        f"feedbacks-monthly_{model}.nc",
    )

    if not os.path.exists(feedback_file):
        print(f"[{model}] feedback file not found")
        return

    if not os.path.exists(cloud_file):
        print(f"[{model}] cloud-feedback file not found")
        return

    print(f"[{model}] merging feedback datasets")

    ds_feedback = xr.open_dataset(feedback_file)
    ds_cloud = xr.open_dataset(cloud_file)

    # Combine non-cloud and cloud feedback diagnostics.
    ds_merged = xr.merge(
        [
            ds_feedback,
            ds_cloud,
        ],
        combine_attrs="override",
    )

    # Load the merged dataset before closing the input files.
    ds_merged.load()

    ds_feedback.close()
    ds_cloud.close()

    # Save the complete per-model dataset.
    ds_merged.to_netcdf(merged_file)
    ds_merged.close()

    print(f"[{model}] saved {merged_file}")

    # Remove intermediate files after successful merge.
    os.remove(feedback_file)
    os.remove(cloud_file)

    print(f"[{model}] removed intermediate feedback files")

# ---------------------------------
# Merge all models
# ---------------------------------
model_files = sorted(
    glob.glob(f"{SAVE}/feedbacks_*.nc")
)

models = [
    os.path.basename(file)
    .replace("feedbacks_", "")
    .replace(".nc", "")
    for file in model_files
]

for model in models:
    merge_model_feedbacks(model)
# %%
# =================================
# OPTIONAL: Common-grid remapping and merge
# =================================

# Define a fixed 2.5° × 2.5° global latitude-longitude grid.
# This grid is independent of any individual CMIP6 model.
target_lat = np.arange(-90.0, 90.0 + 2.5, 2.5)
target_lon = np.arange(0.0, 360.0, 2.5)

print("Common target grid:")
print(f"  latitude:  {len(target_lat)} points")
print(f"  longitude: {len(target_lon)} points")

def remap_to_common_grid(model):
    """Interpolate one model's feedback dataset to the common grid."""

    infile = os.path.join(SAVE, f"feedbacks-monthly_{model}.nc")

    if not os.path.exists(infile):
        print(f"[{model}] feedback file not found")
        return None

    print(f"[{model}] remapping to common grid")

    ds = xr.open_dataset(infile)

    # Interpolate only variables with spatial dimensions.
    spatial_vars = [
        var
        for var in ds.data_vars
        if "lat" in ds[var].dims and "lon" in ds[var].dims
    ]

    ds_common = ds[spatial_vars].interp(
        lat=target_lat,
        lon=target_lon,
    )

    # Add variables without spatial dimensions back to the dataset.
    non_spatial_vars = [
        var
        for var in ds.data_vars
        if var not in spatial_vars
    ]

    if non_spatial_vars:
        ds_common = xr.merge(
            [
                ds_common,
                ds[non_spatial_vars],
            ]
        )

    # Add model as a new dimension.
    ds_common = ds_common.expand_dims(model=[model])

    return ds_common


# ---------------------------------
# Remap all models
# ---------------------------------

model_files = sorted(glob.glob(f"{SAVE}/feedbacks-monthly_*.nc"))

if not model_files:
    raise FileNotFoundError(
        f"No per-model feedback files found in {SAVE}"
    )

models = [
    os.path.basename(file)
    .replace("feedbacks-monthly_", "")
    .replace(".nc", "")
    for file in model_files
]

print(f"Found {len(models)} model feedback files")


datasets_common = []

for model in models:
    ds_common = remap_to_common_grid(model)

    if ds_common is not None:
        datasets_common.append(ds_common)


# ---------------------------------
# Concatenate models
# ---------------------------------
if not datasets_common:
    raise RuntimeError(
        "No model datasets were available for merging."
    )

ds_common_allmodels = xr.concat(
    datasets_common,
    dim="model",
    combine_attrs="override",
)

# Ensure a consistent model order.
ds_common_allmodels = ds_common_allmodels.sel(
    model=sorted(ds_common_allmodels.model.values)
)

# ---------------------------------
# Save common-grid dataset
# ---------------------------------

COMMON_GRID_FILE = os.path.join(
    SAVE,
    "feedbacks_pergrid-monthly_common-grid_allmodels.nc",
)

ds_common_allmodels.to_netcdf(COMMON_GRID_FILE)

print("Common-grid dataset saved:")
print(f"  {COMMON_GRID_FILE}")
print(ds_common_allmodels)
# %%
