#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate climate feedbacks, ERF and ECS
=========================================

This script calculates equilibrium climate sensitivity (ECS), effective radiative forcing (ERF), net climate feedback parameters, individual radiative-kernel feedbacks, and cloud feedbacks from CMIP6 piControl and abrupt-4xCO2 simulations.

The calculations are based on global-annual-mean TOA radiative fluxes and near-surface air temperature. Individual surface-albedo, Planck, lapse-rate, and water-vapour feedbacks are obtained from radiative-kernel calculations performed with the ClimKern package. Cloud feedbacks are subsequently calculated using the cloud adjustment method.

The script is configured for the DKRZ file system. Users reproducing the analysis on another system should adapt the input and output paths.

Gregory regressions use the full available time series after conversion to global-annual means by `gam()`.
"""

# %%
# =================================
# Imports
# =================================
import datetime
import gc
import glob
import os
import warnings

import climkern as ck
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import linregress

warnings.filterwarnings("ignore")

# %%
# =================================
# Paths and settings
# =================================

# Directory containing the preprocessed CMIP data.
DIR = "/path/to/data_output"
# Directory containing the preprocessed radiative-kernel-derived TOA feedback fields.
FDBFLX_DIR = os.path.join(DIR, "feedback_toa_flux")
# Directory to store the final feedback datasets.
SAVE_DIR = os.path.join(DIR, "feedbacks")

os.makedirs(SAVE_DIR, exist_ok=True)

# Radiative kernels used for the feedback decomposition.
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

# Individual feedback variables and corresponding output names.
FB_VARS = ["SA", "Planck", "LR", "WV"]
FB_VARNAMES = ["SAF", "PF", "LRF", "WVF"]

# Generate diagnostic plots for the selected kernel.
PLOT_DIAGNOSTICS = True

# %%
# =================================
# Utility functions
# =================================

def linregress_slope_intercept(x, y):
    """Compute a linear regression between two 1D arrays."""
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, intercept, r_value, p_value, std_err

def lin_fit(x, y, plot=False, label=None):
    """
    Fit a linear relationship between two 1D datasets.

    Parameters
    ----------
    x, y : array-like
        Independent and dependent variables.
    plot : bool, optional
        Generate a scatter plot and fitted line.
    label : str, optional
        Label for the fitted line.

    Returns
    -------
    m, n : float
        Slope and intercept of the fitted line.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        raise ValueError("Not enough valid data points for linear fit.")

    m, n = np.polyfit(x, y, 1)

    if plot:
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = m * x_fit + n

        plt.scatter(x, y, color="k", s=5)
        plt.plot(x_fit, y_fit, label=label)
        plt.axhline(0, ls=":", color="k")
        plt.xlim(x.min(), x.max())

    return m, n

def read(var, model, exp, directory=DIR):
    """
    Read a preprocessed CMIP6 variable.

    Parameters
    ----------
    var : str
        Variable name.
    model : str
        CMIP6 model name.
    exp : str
        Experiment name.
    directory : str, optional
        Directory containing the NetCDF files.

    Returns
    -------
    xr.Dataset or None
        Dataset if the file exists, otherwise None.
    """
    pattern = f"{directory}/{var}_Amon_{model}_{exp}_*.nc"
    files = glob.glob(pattern)

    if not files:
        print(f"  WARNING: Missing {var} for {model}, {exp}")
        return None

    try:
        return xr.open_dataset(files[0])
    except Exception as exc:
        print(f"  WARNING: Could not open {files[0]}: {exc}")
        return None

def get_dataarray(var, model, exp, directory=DIR):
    """
    Read a variable and return it as an xarray DataArray.

    Returns None if the requested file or variable is unavailable.
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
    Calculate the global-annual mean.

    Spatial averaging uses cosine-of-latitude weighting. If a month
    dimension is present, the monthly means are subsequently averaged
    to obtain an annual mean.

    Parameters
    ----------
    ds : xr.DataArray or xr.Dataset
        Input data containing latitude and longitude dimensions.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Global-annual mean.
    """
    if "lat" not in ds.dims or "lon" not in ds.dims:
        raise ValueError(
            "Input dataset must contain 'lat' and 'lon' dimensions."
        )

    weights = np.cos(np.deg2rad(ds.lat))
    wmean = ds.weighted(weights).mean(dim=["lat", "lon"])

    if "month" in wmean.dims:
        wmean = wmean.mean(dim="month")

    return wmean

def common_model_list(directory, variables):
    """
    Find models that provide all requested variables.

    Parameters
    ----------
    directory : str
        Directory containing the NetCDF files.
    variables : list of str
        Variables that must be available.

    Returns
    -------
    list of str
        Sorted list of common models.
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

    print(f"{len(common_models)} model(s) found")
    return sorted(common_models)

def all_feedback_files_exist(model, kernel, feedback):
    """Check whether all-sky and clear-sky feedback files exist."""
    file_as = os.path.join(
        FDBFLX_DIR, f"{feedback}_{model}_{kernel}.nc"
    )
    file_cs = os.path.join(
        FDBFLX_DIR, f"{feedback}_cs_{model}_{kernel}.nc"
    )

    return os.path.exists(file_as) and os.path.exists(file_cs)

def read_feedback(model, kernel, feedback):
    """
    Read and globally average an all-sky and clear-sky feedback field.
    """
    file_as = os.path.join(
        FDBFLX_DIR, f"{feedback}_{model}_{kernel}.nc"
    )
    file_cs = os.path.join(
        FDBFLX_DIR, f"{feedback}_cs_{model}_{kernel}.nc"
    )

    if not all(
        os.path.exists(filename)
        for filename in [file_as, file_cs]
    ):
        return None, None

    with xr.open_dataset(file_as) as ds_as:
        dR_as = gam(ds_as[feedback]).load()

    with xr.open_dataset(file_cs) as ds_cs:
        dR_cs = gam(ds_cs[f"{feedback}_cs"]).load()

    return dR_as, dR_cs


# %%
# =================================
# Part 1 — ECS, ERF and feedbacks
# =================================

"""
Calculate ECS, effective radiative forcing and net feedback parameters using Gregory regression.

For each model, TOA radiative perturbations are calculated from the
difference between abrupt-4xCO2 and piControl simulations.

Outgoing TOA shortwave and longwave fluxes are converted to a
positive-downward radiative perturbation. Global-annual means are calculated using cosine-of-latitude weighting.

The Gregory regression of radiative perturbation against global-mean
near-surface air temperature provides the feedback parameter and
effective radiative forcing.

ECS is calculated as:

    ECS = -ERF / lambda * 0.5

where ERF and lambda are obtained from the abrupt-4xCO2 Gregory
regression and the factor 0.5 converts the 4xCO2 result to the
corresponding 2xCO2 ECS.

Individual surface-albedo, Planck, lapse-rate and water-vapour
feedbacks are calculated for each available radiative kernel.
"""

# Models with the basic variables required for the Gregory regression.
models = common_model_list(
    DIR,
    ["tas", "rsut", "rlut"],
)
ds_all_models = []
# %%
# =================================
# ECS and global feedback calculation
# =================================
for m_count, model in enumerate(models):
    print(
        f"\n===== MODEL: {model} "
        f"({m_count + 1}/{len(models)}) ====="
    )
    # ---------------------------------
    # Read temperature
    # ---------------------------------
    tas_piC = get_dataarray("tas", model, "piControl")
    tas_co2 = get_dataarray("tas", model, "abrupt-4xCO2")

    if tas_piC is None or tas_co2 is None:
        print(f"  WARNING: Missing temperature data for {model}")
        continue

    d_tas = gam(tas_co2 - tas_piC)

    # ---------------------------------
    # Read all-sky TOA fluxes
    # ---------------------------------
    rsut_piC = get_dataarray("rsut", model, "piControl")
    rsut_co2 = get_dataarray("rsut", model, "abrupt-4xCO2")
    rlut_piC = get_dataarray("rlut", model, "piControl")
    rlut_co2 = get_dataarray("rlut", model, "abrupt-4xCO2")

    # ---------------------------------
    # Read clear-sky TOA fluxes
    # ---------------------------------
    rsutcs_piC = get_dataarray("rsutcs", model, "piControl")
    rsutcs_co2 = get_dataarray("rsutcs", model, "abrupt-4xCO2")
    rlutcs_piC = get_dataarray("rlutcs", model, "piControl")
    rlutcs_co2 = get_dataarray("rlutcs", model, "abrupt-4xCO2")

    required_fluxes = [
        rsut_piC,
        rsut_co2,
        rlut_piC,
        rlut_co2,
        rsutcs_piC,
        rsutcs_co2,
        rlutcs_piC,
        rlutcs_co2,
    ]

    if any(value is None for value in required_fluxes):
        print(f"  WARNING: Missing TOA flux data for {model}")
        continue

    # ---------------------------------
    # Radiative perturbations
    # Positive downward
    # ---------------------------------
    d_r_sw = gam(-(rsut_co2 - rsut_piC))
    d_r_lw = gam(-(rlut_co2 - rlut_piC))
    d_r = gam(
        -(rsut_co2 - rsut_piC)
        -(rlut_co2 - rlut_piC)
    )

    d_r_sw_cs = gam(-(rsutcs_co2 - rsutcs_piC))
    d_r_lw_cs = gam(-(rlutcs_co2 - rlutcs_piC))

    # ---------------------------------
    # Gregory regressions
    # ---------------------------------
    plot_regression = (
        PLOT_DIAGNOSTICS
        and model == models[0]
    )

    net, erf = lin_fit(
        x=d_tas,
        y=d_r,
        plot=plot_regression,
        label="Net",
    )

    net_sw, erf_sw = lin_fit(
        x=d_tas,
        y=d_r_sw,
    )

    net_lw, erf_lw = lin_fit(
        x=d_tas,
        y=d_r_lw,
    )

    net_sw_cs, erf_sw_cs = lin_fit(
        x=d_tas,
        y=d_r_sw_cs,
    )

    net_lw_cs, erf_lw_cs = lin_fit(
        x=d_tas,
        y=d_r_lw_cs,
    )

    # ECS for a doubling of CO2.
    ecs = -erf / net * 0.5

    # ---------------------------------
    # Sanity checks
    # ---------------------------------
    print(
        "  SW + LW ERF = total ERF:",
        np.isclose(erf_sw + erf_lw, erf),
    )
    print(
        "  SW + LW feedback = total feedback:",
        np.isclose(net_sw + net_lw, net),
    )

    # ---------------------------------
    # Pack global diagnostics
    # ---------------------------------
    ds_ecs = xr.Dataset(
        {
            "ecs": ecs,
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
            "dT": d_tas,
        }
    )

    # ---------------------------------
    # Decomposed feedback
    # ---------------------------------
    kernel_feedbacks = []
    for v_count, (fb_var, fb_name) in enumerate(
        zip(FB_VARS, FB_VARNAMES)
    ):
        print(f"\n---> {fb_var} feedback")
        feedback_per_kernel = []

        for k_count, kernel in enumerate(kernels):
            print(
                f"  kernel: {kernel} "
                f"({k_count + 1}/{len(kernels)})"
            )
            if not all_feedback_files_exist(
                model,
                kernel,
                fb_var,
            ):
                print(
                    f"    WARNING: Missing {fb_var} "
                    f"for {kernel} → NaN"
                )
                feedback_per_kernel.append(
                    xr.Dataset(
                        {
                            fb_name: np.nan,
                            f"{fb_name}_cs": np.nan,
                        },
                        coords={"kernel": [kernel]},
                    )
                )
                continue

            dR_fdb_as, dR_fdb_cs = read_feedback(
                model,
                kernel,
                fb_var,
            )
            if dR_fdb_as is None or dR_fdb_cs is None:
                feedback_per_kernel.append(
                    xr.Dataset(
                        {
                            fb_name: np.nan,
                            f"{fb_name}_cs": np.nan,
                        },
                        coords={"kernel": [kernel]},
                    )
                )
                continue

            plot_feedback = (
                PLOT_DIAGNOSTICS
                and k_count == 0
            )

            fb_as, _ = lin_fit(
                x=d_tas,
                y=dR_fdb_as,
                plot=plot_feedback,
                label=fb_var,
            )
            fb_cs, _ = lin_fit(
                x=d_tas,
                y=dR_fdb_cs,
            )
            ds_feedback = xr.Dataset(
                {
                    fb_name: fb_as,
                    f"{fb_name}_cs": fb_cs,
                },
                coords={"kernel": [kernel]},
            )
            feedback_per_kernel.append(ds_feedback)

            # Store representative radiative perturbations
            # only for one kernel
            if k_count == 0:
                dR_fdb_as.name = f"dR_{fb_var}"

                if "dR_feedback" not in locals():
                    dR_feedback = dR_fdb_as
                else:
                    dR_feedback = xr.merge(
                        [dR_feedback, dR_fdb_as]
                    )

        kernel_feedbacks.append(
            xr.concat(
                feedback_per_kernel,
                dim="kernel",
            )
        )

    # ---------------------------------
    # Diagnostic plot
    # ---------------------------------
    if PLOT_DIAGNOSTICS and model == models[0]:
        plt.title(
            f"{model} - "
            f"ECS={ecs:.2f}, "
            f"ERF2xCO2={erf / 2:.2f}, "
            f"λ={net:.2f}"
        )
        plt.legend()
        plt.show()

    # ---------------------------------
    # Merge model output
    # ---------------------------------
    if "dR_feedback" not in locals():
        dR_feedback = xr.Dataset()

    d_r.name = "dR_net"

    ds_model = xr.merge(
        [
            ds_ecs,
            *kernel_feedbacks,
            dR_feedback,
            d_r,
        ]
    )

    ds_model = ds_model.expand_dims(model=[model])

    ds_all_models.append(ds_model)

    # Remove model-specific temporary variables.
    if "dR_feedback" in locals():
        del dR_feedback

    gc.collect()

    print(f"Finished {model}")

# %%
# =================================
# Save Part 1
# =================================
if ds_all_models:
    ds_ecs_all = xr.concat(
        ds_all_models,
        dim="model",
    )
    ecs_file = os.path.join(
        SAVE_DIR,
        "feedbacks_gam_allmodels.nc",
    )
    print(f"\nSaving ECS and feedback diagnostics → {ecs_file}")

    ds_ecs_all.to_netcdf(ecs_file)

else:
    raise RuntimeError(
        "No model results were produced in Part 1."
    )

# %%
# =================================
# Part 2 — Cloud feedbacks
# =================================

"""
Calculate shortwave and longwave cloud feedbacks using the cloud
adjustment method.

The cloud-induced radiative perturbation is calculated from changes
in the cloud radiative effect after accounting for the contributions
from surface albedo, water vapour, lapse rate and Planck feedbacks,
together with the corresponding effective radiative forcing terms.

Cloud feedback parameters are obtained by Gregory regression of the
cloud-induced radiative perturbation against the global-mean
temperature response.

The calculations are performed separately for each radiative kernel.
"""

# =================================
# Load Part 1 results
# =================================

ds_ecs = xr.open_dataset(
    os.path.join(
        SAVE_DIR,
        "feedbacks_gam_allmodels.nc",
    )
)
models = list(ds_ecs.model.values)

# %%
# =================================
# Cloud feedback calculation
# =================================
cloud_results = []

for m_count, model in enumerate(models):
    print(
        f"\n===== CLOUD FEEDBACK: {model} "
        f"({m_count + 1}/{len(models)}) ====="
    )

    # ---------------------------------
    # Temperature response
    # ---------------------------------
    tas_piC = get_dataarray("tas", model, "piControl")
    tas_co2 = get_dataarray("tas", model, "abrupt-4xCO2")

    if tas_piC is None or tas_co2 is None:
        print(f"  WARNING: Missing temperature data for {model}")
        continue

    d_tas = gam(tas_co2 - tas_piC)

    # ---------------------------------
    # Shortwave TOA fluxes
    # ---------------------------------
    rsut_piC = get_dataarray("rsut", model, "piControl")
    rsut_co2 = get_dataarray("rsut", model, "abrupt-4xCO2")
    rsdt_piC = get_dataarray("rsdt", model, "piControl")
    rsdt_co2 = get_dataarray("rsdt", model, "abrupt-4xCO2")
    rsutcs_piC = get_dataarray("rsutcs", model, "piControl")
    rsutcs_co2 = get_dataarray("rsutcs", model, "abrupt-4xCO2")

    sw_inputs = [
        rsut_piC,
        rsut_co2,
        rsdt_piC,
        rsdt_co2,
        rsutcs_piC,
        rsutcs_co2,
    ]

    if any(value is None for value in sw_inputs):

        print(
            f"  WARNING: Missing SW flux data for {model}"
        )

        dCRE_SW = None

    else:

        rsnt_piC = gam(rsdt_piC - rsut_piC)
        rsnt_co2 = gam(rsdt_co2 - rsut_co2)

        rsntcs_piC = gam(rsdt_piC - rsutcs_piC)
        rsntcs_co2 = gam(rsdt_co2 - rsutcs_co2)

        dCRE_SW = ck.calc_dCRE_SW(
            rsnt_piC,
            rsnt_co2,
            rsntcs_piC,
            rsntcs_co2,
        )

    # ---------------------------------
    # Longwave TOA fluxes
    # ---------------------------------
    rlut_piC = get_dataarray("rlut", model, "piControl")
    rlut_co2 = get_dataarray("rlut", model, "abrupt-4xCO2")
    rlutcs_piC = get_dataarray("rlutcs", model, "piControl")
    rlutcs_co2 = get_dataarray("rlutcs", model, "abrupt-4xCO2")

    lw_inputs = [
        rlut_piC,
        rlut_co2,
        rlutcs_piC,
        rlutcs_co2,
    ]

    if any(value is None for value in lw_inputs):

        print(
            f"  WARNING: Missing LW flux data for {model}"
        )

        dCRE_LW = None

    else:

        rlnt_piC = gam(rlut_piC)
        rlnt_co2 = gam(rlut_co2)

        rlntcs_piC = gam(rlutcs_piC)
        rlntcs_co2 = gam(rlutcs_co2)

        dCRE_LW = ck.calc_dCRE_LW(
            rlnt_piC,
            rlnt_co2,
            rlntcs_piC,
            rlntcs_co2,
        )

    # ---------------------------------
    # ERF components from Part 1
    # ---------------------------------
    erf_sw = ds_ecs.erf_sw.sel(model=model)
    erf_sw_cs = ds_ecs.erf_sw_cs.sel(model=model)
    erf_lw = ds_ecs.erf_lw.sel(model=model)
    erf_lw_cs = ds_ecs.erf_lw_cs.sel(model=model)

    kernel_sw = []
    kernel_lw = []

    dR_cloud_SW = None
    dR_cloud_LW = None

    # ---------------------------------
    # Loop over kernels
    # ---------------------------------
    for k_count, kernel in enumerate(kernels):

        print(
            f"  kernel: {kernel} "
            f"({k_count + 1}/{len(kernels)})"
        )

        # Feedback files.
        sa_file = os.path.join(
            FDBFLX_DIR,
            f"SA_{model}_{kernel}.nc",
        )
        sa_cs_file = os.path.join(
            FDBFLX_DIR,
            f"SA_cs_{model}_{kernel}.nc",
        )

        wv_file = os.path.join(
            FDBFLX_DIR,
            f"WV_{model}_{kernel}.nc",
        )
        wv_cs_file = os.path.join(
            FDBFLX_DIR,
            f"WV_cs_{model}_{kernel}.nc",
        )

        lr_file = os.path.join(
            FDBFLX_DIR,
            f"LR_{model}_{kernel}.nc",
        )
        lr_cs_file = os.path.join(
            FDBFLX_DIR,
            f"LR_cs_{model}_{kernel}.nc",
        )

        planck_file = os.path.join(
            FDBFLX_DIR,
            f"Planck_{model}_{kernel}.nc",
        )
        planck_cs_file = os.path.join(
            FDBFLX_DIR,
            f"Planck_cs_{model}_{kernel}.nc",
        )

        # ---------------------------------
        # Shortwave cloud feedback
        # ---------------------------------
        sw_files = [
            sa_file,
            sa_cs_file,
            wv_file,
            wv_cs_file,
        ]

        if (
            dCRE_SW is not None
            and all(os.path.exists(file) for file in sw_files)
        ):

            SA, SA_cs = read_feedback(
                model,
                kernel,
                "SA",
            )

            WV, WV_cs = read_feedback(
                model,
                kernel,
                "WV",
            )

            dR_cloud_SW = ck.calc_cloud_SW(
                SA,
                SA_cs,
                WV,
                WV_cs,
                dCRE_SW,
                erf_sw,
                erf_sw_cs,
            )

            plot_feedback = (
                PLOT_DIAGNOSTICS
                and k_count == 0
            )

            CF_SW, _ = lin_fit(
                x=d_tas,
                y=dR_cloud_SW,
                plot=plot_feedback,
                label="CF_SW",
            )

        else:
            print(
                "    WARNING: Missing SW cloud inputs "
                "→ CF_SW = NaN"
            )
            CF_SW = np.nan

        kernel_sw.append(
            xr.Dataset(
                {
                    "CF_SW": CF_SW,
                },
                coords={"kernel": [kernel]},
            )
        )

        # ---------------------------------
        # Longwave cloud feedback
        # ---------------------------------
        lw_files = [
            lr_file,
            lr_cs_file,
            planck_file,
            planck_cs_file,
            wv_file,
            wv_cs_file,
        ]

        if (
            dCRE_LW is not None
            and all(os.path.exists(file) for file in lw_files)
        ):

            LR, LR_cs = read_feedback(
                model,
                kernel,
                "LR",
            )

            Planck, Planck_cs = read_feedback(
                model,
                kernel,
                "Planck",
            )

            WV, WV_cs = read_feedback(
                model,
                kernel,
                "WV",
            )

            # Non-cloud LW feedback contribution.
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

            plot_feedback = (
                PLOT_DIAGNOSTICS
                and k_count == 0
            )

            CF_LW, _ = lin_fit(
                x=d_tas,
                y=dR_cloud_LW,
                plot=plot_feedback,
                label="CF_LW",
            )

        else:
            print(
                "    WARNING: Missing LW cloud inputs "
                "→ CF_LW = NaN"
            )
            CF_LW = np.nan

        kernel_lw.append(
            xr.Dataset(
                {
                    "CF_LW": CF_LW,
                },
                coords={"kernel": [kernel]},
            )
        )

    # ---------------------------------
    # Merge kernel results
    # ---------------------------------
    ds_cloud = xr.merge(
        [
            xr.concat(kernel_sw, dim="kernel"),
            xr.concat(kernel_lw, dim="kernel"),
        ]
    )

    # Store representative cloud perturbations.
    if (
        dR_cloud_SW is not None
        and dR_cloud_LW is not None
    ):

        dR_cloud_SW.name = "dR_cloud_SW"
        dR_cloud_LW.name = "dR_cloud_LW"

        ds_cloud = xr.merge(
            [
                ds_cloud,
                dR_cloud_SW,
                dR_cloud_LW,
            ]
        )

    ds_cloud = ds_cloud.expand_dims(
        model=[model]
    )

    cloud_results.append(ds_cloud)

    # ---------------------------------
    # Diagnostic plot
    # ---------------------------------
    if PLOT_DIAGNOSTICS:

        swvalue = ds_cloud.CF_SW.sel(
            model=model
        )[0,...].item()

        lwvalue = ds_cloud.CF_LW.sel(
            model=model
        )[0,...].item()

        plt.legend()
        plt.title(
            f"{model} - "
            f"CF={swvalue + lwvalue:.2f}, "
            f"SW={swvalue:.2f}, "
            f"LW={lwvalue:.2f}"
        )
        plt.show()

    gc.collect()

# %%
# =================================
# Save cloud feedbacks
# =================================

if cloud_results:

    ds_cloud_all = xr.concat(
        cloud_results,
        dim="model",
    )

    cloud_file = os.path.join(
        SAVE_DIR,
        "cloud_feedbacks_gam_allmodels.nc",
    )

    print(
        f"\nSaving cloud feedback diagnostics → "
        f"{cloud_file}"
    )

    ds_cloud_all.to_netcdf(cloud_file)

else:

    raise RuntimeError(
        "No cloud-feedback results were produced."
    )


# %%
# =================================
# Part 3 — Merge and metadata
# =================================

"""
Merge the ECS/feedback and cloud-feedback datasets and assign descriptive metadata to the final variables.
"""

def add_feedback_attributes(ds):
    """Assign metadata to the global feedback dataset."""

    attr_map = {

        # ---------------------------------
        # Core climate metrics
        # ---------------------------------
        "ecs": dict(
            long_name="Equilibrium climate sensitivity",
            units="K",
            description=(
                "Equilibrium global-mean surface temperature response "
                "to a doubling of atmospheric CO2 concentration."
            ),
        ),

        "erf": dict(
            long_name="Effective radiative forcing (all-sky)",
            units="W m-2",
            description=(
                "Total effective radiative forcing at the TOA under "
                "all-sky conditions for abrupt-4xCO2."
            ),
        ),

        "erf_sw": dict(
            long_name="Shortwave effective radiative forcing (all-sky)",
            units="W m-2",
            description=(
                "Shortwave component of effective radiative forcing "
                "at the TOA under all-sky conditions for abrupt-4xCO2."
            ),
        ),

        "erf_lw": dict(
            long_name="Longwave effective radiative forcing (all-sky)",
            units="W m-2",
            description=(
                "Longwave component of effective radiative forcing "
                "at the TOA under all-sky conditions for abrupt-4xCO2."
            ),
        ),

        "erf_sw_cs": dict(
            long_name="Shortwave effective radiative forcing (clear-sky)",
            units="W m-2",
            description=(
                "Shortwave component of effective radiative forcing "
                "at the TOA under clear-sky conditions for abrupt-4xCO2."
            ),
        ),

        "erf_lw_cs": dict(
            long_name="Longwave effective radiative forcing (clear-sky)",
            units="W m-2",
            description=(
                "Longwave component of effective radiative forcing "
                "at the TOA under clear-sky conditions for abrupt-4xCO2."
            ),
        ),

        # ---------------------------------
        # Net feedbacks
        # ---------------------------------
        "net": dict(
            long_name="Net feedback parameter (all-sky)",
            units="W m-2 K-1",
            description=(
                "Net all-sky climate feedback parameter derived from "
                "Gregory regression of TOA radiative perturbation "
                "against global-mean temperature response."
            ),
        ),

        "net_sw": dict(
            long_name="Shortwave feedback parameter (all-sky)",
            units="W m-2 K-1",
            description=(
                "Shortwave component of the net all-sky feedback "
                "parameter derived from Gregory regression."
            ),
        ),

        "net_lw": dict(
            long_name="Longwave feedback parameter (all-sky)",
            units="W m-2 K-1",
            description=(
                "Longwave component of the net all-sky feedback "
                "parameter derived from Gregory regression."
            ),
        ),

        "net_sw_cs": dict(
            long_name="Shortwave feedback parameter (clear-sky)",
            units="W m-2 K-1",
            description=(
                "Shortwave component of the net clear-sky feedback "
                "parameter derived from Gregory regression."
            ),
        ),

        "net_lw_cs": dict(
            long_name="Longwave feedback parameter (clear-sky)",
            units="W m-2 K-1",
            description=(
                "Longwave component of the net clear-sky feedback "
                "parameter derived from Gregory regression."
            ),
        ),

        # ---------------------------------
        # Kernel-derived feedbacks
        # ---------------------------------
        "SAF": dict(
            long_name="Surface albedo feedback (all-sky)",
            units="W m-2 K-1",
            description=(
                "Surface albedo feedback derived using radiative "
                "kernels under all-sky conditions."
            ),
        ),

        "SAF_cs": dict(
            long_name="Surface albedo feedback (clear-sky)",
            units="W m-2 K-1",
            description=(
                "Surface albedo feedback derived using radiative "
                "kernels under clear-sky conditions."
            ),
        ),

        "PF": dict(
            long_name="Planck feedback (all-sky)",
            units="W m-2 K-1",
            description=(
                "Planck feedback derived using radiative kernels "
                "under all-sky conditions."
            ),
        ),

        "PF_cs": dict(
            long_name="Planck feedback (clear-sky)",
            units="W m-2 K-1",
            description=(
                "Planck feedback derived using radiative kernels "
                "under clear-sky conditions."
            ),
        ),

        "LRF": dict(
            long_name="Lapse-rate feedback (all-sky)",
            units="W m-2 K-1",
            description=(
                "Lapse-rate feedback derived using radiative kernels "
                "under all-sky conditions."
            ),
        ),

        "LRF_cs": dict(
            long_name="Lapse-rate feedback (clear-sky)",
            units="W m-2 K-1",
            description=(
                "Lapse-rate feedback derived using radiative kernels "
                "under clear-sky conditions."
            ),
        ),

        "WVF": dict(
            long_name="Water vapour feedback (all-sky)",
            units="W m-2 K-1",
            description=(
                "Water vapour feedback derived using radiative kernels "
                "under all-sky conditions."
            ),
        ),

        "WVF_cs": dict(
            long_name="Water vapour feedback (clear-sky)",
            units="W m-2 K-1",
            description=(
                "Water vapour feedback derived using radiative kernels "
                "under clear-sky conditions."
            ),
        ),

        # ---------------------------------
        # Cloud feedbacks
        # ---------------------------------
        "CF_SW": dict(
            long_name="Shortwave cloud feedback",
            units="W m-2 K-1",
            description=(
                "Shortwave cloud feedback derived using the cloud "
                "adjustment method."
            ),
        ),

        "CF_LW": dict(
            long_name="Longwave cloud feedback",
            units="W m-2 K-1",
            description=(
                "Longwave cloud feedback derived using the cloud "
                "adjustment method."
            ),
        ),

        # ---------------------------------
        # Regression diagnostics
        # ---------------------------------
        "dT": dict(
            long_name="Near-surface air temperature anomaly",
            units="K",
            description=(
                "Global-annual-mean near-surface air temperature "
                "response between abrupt-4xCO2 and piControl."
            ),
        ),

        "dR_net": dict(
            long_name="TOA radiative perturbation",
            units="W m-2",
            description=(
                "Global-annual-mean net TOA radiative perturbation "
                "used as the dependent variable in the Gregory regression."
            ),
        ),

        "dR_SA": dict(
            long_name="TOA radiative perturbation associated with surface albedo",
            units="W m-2",
            description=(
                "All-sky TOA radiative perturbation associated with "
                "surface-albedo changes, calculated using the "
                "ECMWF-RRTM kernel."
            ),
        ),

        "dR_Planck": dict(
            long_name="TOA radiative perturbation associated with Planck feedback",
            units="W m-2",
            description=(
                "All-sky TOA radiative perturbation associated with "
                "Planck feedback, calculated using the ECMWF-RRTM kernel."
            ),
        ),

        "dR_LR": dict(
            long_name="TOA radiative perturbation associated with lapse-rate feedback",
            units="W m-2",
            description=(
                "All-sky TOA radiative perturbation associated with "
                "lapse-rate feedback, calculated using the ECMWF-RRTM kernel."
            ),
        ),

        "dR_WV": dict(
            long_name="TOA radiative perturbation associated with water-vapour feedback",
            units="W m-2",
            description=(
                "All-sky TOA radiative perturbation associated with "
                "water-vapour feedback, calculated using the ECMWF-RRTM kernel."
            ),
        ),

        "dR_cloud_SW": dict(
            long_name="TOA shortwave cloud radiative perturbation",
            units="W m-2",
            description=(
                "Shortwave cloud-induced TOA radiative perturbation "
                "calculated using the cloud adjustment method and "
                "the ECMWF-RRTM kernel."
            ),
        ),

        "dR_cloud_LW": dict(
            long_name="TOA longwave cloud radiative perturbation",
            units="W m-2",
            description=(
                "Longwave cloud-induced TOA radiative perturbation "
                "calculated using the cloud adjustment method and "
                "the ECMWF-RRTM kernel."
            ),
        ),
    }

    # Apply variable attributes.
    for var in ds.data_vars:

        if var in attr_map:
            ds[var].attrs = attr_map[var]

        else:
            ds[var].attrs = {
                "long_name": var,
                "description": "No metadata assigned.",
            }

    # ---------------------------------
    # Global attributes
    # ---------------------------------
    ds.attrs["title"] = (
        "CMIP6 climate sensitivity, forcing and feedback diagnostics"
    )

    ds.attrs["description"] = (
        "Global-annual-mean CMIP6 climate sensitivity, forcing and "
        "feedback diagnostics derived from piControl and "
        "abrupt-4xCO2 experiments."
    )

    ds.attrs["reference"] = (
        "Gregory et al. (2004), "
        "doi:10.1029/2003GL018747; "
        "Soden et al. (2008), "
        "doi:10.1175/2007JCLI2110.1."
    )

    ds.attrs["software"] = (
        "Python and ClimKern"
    )

    ds.attrs["source"] = (
        "CMIP6 piControl and abrupt-4xCO2 experiments"
    )

    ds.attrs["created_by"] = (
        "Climate feedback diagnostics pipeline"
    )

    ds.attrs["author"] = "Olivia Linke"

    ds.attrs["institution"] = (
        "Department of Meteorology and Bolin Centre for Climate "
        "Research, Stockholm University"
    )

    ds.attrs["date_created"] = (
        datetime.datetime.now().strftime("%Y-%m-%d")
    )

    ds.attrs["acknowledgement"] = (
        "This dataset was generated using CMIP6 model output. "
        "We acknowledge the World Climate Research Programme "
        "(WCRP) and participating modelling centres for producing "
        "and making their model output available."
    )

    # ---------------------------------
    # Preferred variable order
    # ---------------------------------
    order = [
        "ecs",
        "erf",
        "erf_sw",
        "erf_lw",
        "erf_sw_cs",
        "erf_lw_cs",
        "net",
        "net_sw",
        "net_lw",
        "net_sw_cs",
        "net_lw_cs",
        "PF",
        "PF_cs",
        "LRF",
        "LRF_cs",
        "SAF",
        "SAF_cs",
        "WVF",
        "WVF_cs",
        "CF_SW",
        "CF_LW",
        "dT",
        "dR_net",
        "dR_Planck",
        "dR_LR",
        "dR_SA",
        "dR_WV",
        "dR_cloud_SW",
        "dR_cloud_LW",
    ]

    existing = [
        var
        for var in order
        if var in ds
    ]

    ds = ds[existing]

    if "height" in ds.coords:
        ds = ds.drop_vars("height")

    return ds


# %%
# =================================
# Merge Part 1 and Part 2
# =================================

main_file = os.path.join(
    SAVE_DIR,
    "feedbacks_gam_allmodels.nc",
)
cloud_file = os.path.join(
    SAVE_DIR,
    "cloud_feedbacks_gam_allmodels.nc",
)
final_file = os.path.join(
    SAVE_DIR,
    "feedbacks_gam_allmodels_final.nc",
)

print("\nMerging ECS/feedback and cloud-feedback datasets.")

with xr.open_dataset(main_file) as ds1, \
     xr.open_dataset(cloud_file) as ds2:

    ds_final = xr.merge(
        [ds1, ds2]
    )
    ds_final = add_feedback_attributes(
        ds_final
    )
    ds_final.to_netcdf(
        final_file
    )

print(
    f"Final feedback dataset saved to:\n"
    f"{final_file}"
)

os.remove(main_file)
os.remove(cloud_file)

# %%
