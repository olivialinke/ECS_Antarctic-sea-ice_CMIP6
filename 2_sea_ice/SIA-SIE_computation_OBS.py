#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate observational Antarctic sea-ice area and extent
===========================================================

This script calculates monthly Antarctic sea-ice area (SIA) and
sea-ice extent (SIE) from three observational sea-ice concentration
(SIC) products:

    - NSIDC Sea Ice Index (NSIDC-0051)
    - Bootstrap Sea Ice Concentrations (NSIDC-0079)
    - OSI SAF sea-ice concentration

The observational SIC datasets were downloaded and time-merged
prior to this calculation. The analysis period is restricted to
1979-2024.

SIA is calculated as the sum of fractional SIC multiplied by the
area of each native grid cell.

SIE is calculated as the sum of grid-cell areas where SIC is greater
than or equal to 15%.

The resulting SIA and SIE time series are saved separately for each
observational product and then combined into a single NetCDF dataset
with a `prod` dimension.
"""

#%%
# =================================
# Imports
# =================================
import os
import numpy as np
import xarray as xr


# =================================
# Paths and settings
# =================================

# Directory containing the time-merged observational SIC datasets.
DIR = "/work/bb1036/b381106/data_output/obs"

# Analysis period.
YEAR_START = 1979
YEAR_END = 2024

# SIC threshold used to calculate sea-ice extent.
SIE_THRESHOLD = 0.15

# Native grid-cell area of the 25-km observational grids.
GRID_CELL_AREA = 25000**2


# =================================
# Utility functions
# =================================

def select_analysis_period(
    data,
    year_start=YEAR_START,
    year_end=YEAR_END,
):
    """Restrict a time series to the selected analysis period."""

    years = data.time.dt.year

    return data.sel(
        time=years.isin(
            np.arange(year_start, year_end + 1)
        )
    )

def calculate_sia_sie(
    siconc,
    x_dim,
    y_dim,
    area=GRID_CELL_AREA,
):
    """
    Calculate monthly sea-ice area and sea-ice extent.

    SIA is calculated as the sum of fractional SIC multiplied by
    grid-cell area.

    SIE is calculated as the sum of grid-cell area where SIC is
    greater than or equal to the 15% threshold.
    """

    siconc = select_analysis_period(siconc)

    # ---------------------------------
    # Sea-ice area
    # ---------------------------------

    sia = (
        siconc * area
    ).sum(
        dim=[x_dim, y_dim]
    )

    sia.name = "sia_sh"

    # ---------------------------------
    # Sea-ice extent
    # ---------------------------------

    mask = xr.where(
        siconc >= SIE_THRESHOLD,
        1.0,
        0.0,
    )

    sie = (
        mask * area
    ).sum(
        dim=[x_dim, y_dim]
    )

    sie.name = "sie_sh"

    return xr.Dataset(
        {
            "sia_sh": sia,
            "sie_sh": sie,
        }
    )


# =================================
# NSIDC Sea Ice Index
# =================================

print("Calculating NSIDC Sea Ice Index SIA/SIE")

siconc = xr.open_dataset(
    os.path.join(
        DIR,
        "siconc_NT_NSIDC0051_S25km_1978-2024.nc",
    )
)["siconc"]

ds_nt = calculate_sia_sie(
    siconc,
    x_dim="x",
    y_dim="y",
)

nt_file = os.path.join(
    DIR,
    f"sia_sie_SH_NT_{YEAR_START}-{YEAR_END}.nc",
)

ds_nt.to_netcdf(nt_file)

print(f"  saved: {nt_file}")


# =================================
# Bootstrap
# =================================

print("Calculating Bootstrap SIA/SIE")

siconc = xr.open_dataset(
    os.path.join(
        DIR,
        "siconc_BOOTSTRAP_NSIDC0079_S25km_1978-2025.nc",
    )
)["siconc"]

ds_bootstrap = calculate_sia_sie(
    siconc,
    x_dim="x",
    y_dim="y",
)

bootstrap_file = os.path.join(
    DIR,
    f"sia_sie_SH_BOOTSTRAP_{YEAR_START}-{YEAR_END}.nc",
)

ds_bootstrap.to_netcdf(bootstrap_file)

print(f"  saved: {bootstrap_file}")


# =================================
# OSI SAF
# =================================

print("Calculating OSI SAF SIA/SIE")

# OSI SAF SIC is provided in percent and is converted to a fraction.
siconc = (
    xr.open_dataset(
        os.path.join(
            DIR,
            "siconc_sh_OSI-SAF_1979-2024.nc",
        )
    )["ice_conc"]
    / 100.0
)

ds_osi = calculate_sia_sie(
    siconc,
    x_dim="xc",
    y_dim="yc",
)

osi_file = os.path.join(
    DIR,
    f"sia_sie_SH_OSI-SAF_{YEAR_START}-{YEAR_END}.nc",
)

ds_osi.to_netcdf(osi_file)

print(f"  saved: {osi_file}")


# =================================
# Combine observational products
# =================================

print("Combining observational products")

ds_nt = xr.open_dataset(nt_file).assign_coords(
    prod="nt"
)

ds_bootstrap = xr.open_dataset(
    bootstrap_file
).assign_coords(
    prod="bootstrap"
)

ds_osi = xr.open_dataset(
    osi_file
).assign_coords(
    prod="osi"
)

ds_obs = xr.concat(
    [
        ds_nt,
        ds_bootstrap,
        ds_osi,
    ],
    dim="prod",
)

combined_file = os.path.join(
    DIR,
    f"sia_sie_SH_{YEAR_START}-{YEAR_END}_all.nc",
)

ds_obs.to_netcdf(combined_file)

print("Combined observational dataset saved:")
print(f"  {combined_file}")
print(ds_obs)
# %%
