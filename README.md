#### Reproducibility repository

This repository contains the preprocessing and analysis scripts used to produce the results presented in "Uncertainty in climate sensitivity links to Antarctic sea ice".

The workflow includes CMIP6 preprocessing, Antarctic sea-ice diagnostics, radiative-kernel feedback calculations, Gregory-regression ECS estimates, and observational sea-ice analysis.

The repository is designed to document and facilitate reproduction of the analysis. The original CMIP6 model data are not included because of their large volume and should be obtained from ESGF or, where applicable, through ESMValTool. Users without access to DKRZ can download the corresponding datasets from ESGF or ESMValTool and adapt the input paths in the scripts accordingly.

The scripts were developed and executed using Python and CDO. The radiative-kernel calculations use [ClimKern](https://github.com/tyfolino/climkern).

#### Python Dependencies

The main Python packages required by the analysis are:

```text
numpy
xarray
netCDF4
dask
scipy
matplotlib
psutil
git+https://github.com/tyfolino/climkern.git
```

CDO is also used for data preprocessing and must be installed separately.

#### Data requirements

The preprocessing scripts operate on the original CMIP6 model output and generate the diagnostics required by the downstream analysis scripts.

The original CMIP6 input data are not included in this repository because of their large volume.

The main required CMIP6 variables are:

- **Antarctic sea ice:** `siconc` and `areacello`.
- **Feedbacks and ECS:** `tas`, `ps`, `ta`, `hus`, `rsus`, `rsds`, `rlut`, `rlutcs`, `rsut`, `rsutcs`, and `rsdt`.
- **Additional diagnostics:** `uas`, `vas`, `tauu`, `tauv`, `huss`, `clt`, `hfls`, and `hfss`.

For the ECS and net feedback calculations, only `tas`, `rlut`, `rsut`, and `rsdt` are required.

### 1. `cmip_preprocessing.py`

Preprocesses raw CMIP6 data into a standardized format used throughout the feedback and ECS analysis.

#### Data access

The CMIP model output was accessed from the CMIP data pool at the German Climate Computing Centre (DKRZ). The underlying CMIP5/CMIP6 datasets remain publicly available through the Earth System Grid Federation (ESGF), including the [ESGF CMIP6 search portal](https://esgf-node.llnl.gov/search/cmip6/).

The preprocessing script is configured to search the DKRZ CMIP data pool and therefore contains DKRZ-specific input directory paths. Users without access to the DKRZ data pool will need to download the corresponding CMIP datasets from ESGF and adapt the input paths in the script accordingly. Consequently, the preprocessing workflow is publicly available and reproducible in principle, but reproducing it from the raw data requires access to the relevant CMIP files, either through DKRZ or by obtaining them from ESGF.

#### Purpose

- Searches the DKRZ CMIP data pool for a requested variable and experiment (`piControl` or `abrupt-4xCO2`).
- Identifies all available models.
- Selects the **first ensemble member** (lowest realization number, e.g. `r1i1p1f1`) for each model.
- Merges split NetCDF files into a single continuous time series using CDO.
- Retains only the first **150 years** (1800 monthly time steps).
- Removes models with shorter simulations.
- Replaces the original calendar with a standardized **365-day no-leap calendar** (years 1–150) to ensure consistency across models
- Converts the `time` coordinate to separate `year` and `month` dimensions, applied consistently to all preprocessed files.
- Saves one processed NetCDF file per model.

#### Output
One NetCDF file per model:

```text
<variable>_<table>_<model>_<experiment>_<variant>_<grid>.nc
```

containing

- a continuous 150-year monthly time series,
- standardized time coordinates,
- only the first ensemble member.

#### Notes
- Supports both **CMIP5** and **CMIP6**.
- Can be executed independently for each required variable and experiment.
- The processed files are used as input for all subsequent scripts.


### 2. `SIE_SIA_computation.py`

Computes **Sea Ice Extent (SIE)** and **Sea Ice Area (SIA)** for both hemispheres from the preprocessed CMIP sea-ice concentration (`siconc`) data.

Requirements: process `siconc` for `piControl` and `abrupt-4xCO2` with `cmip_preprocessing.py`

#### Purpose
- Searches the CMIP archive for available `areacello` files and copies a single representative file (first ensemble member) for each model.
- Matches `siconc` and `areacello` files to identify models with complete sea-ice information.
- Computes monthly Northern and Southern Hemisphere sea-ice extent and sea-ice area for each model.
- Combines all processed models into a single multi-model dataset.

#### Method
For each model:

- Reads the preprocessed `siconc` simulation together with the corresponding `areacello` file.
- Applies model-specific coordinate/dimension adjustments where necessary to align siconc and areacello on their native grids.
- Computes:
  - **Sea Ice Extent (SIE):** sum of the area of all grid cells with `siconc ≥ 15%`.
  - **Sea Ice Area (SIA):** sum of `siconc × areacello`.
- Calculates both quantities separately for the Northern and Southern Hemisphere.

#### Output

Individual model files:

```text
SIE_SIA_<model>_<experiment>.nc
```

containing

| Variable | Description |
|----------|-------------|
| `SIE_NH` | Northern Hemisphere Sea Ice Extent |
| `SIE_SH` | Southern Hemisphere Sea Ice Extent |
| `SIA_NH` | Northern Hemisphere Sea Ice Area |
| `SIA_SH` | Southern Hemisphere Sea Ice Area |

Final multi-model dataset:

```text
SIE_SIA_<experiment>.nc
```

with dimensions

```text
(model, year, month)
```

#### Notes
- Uses the standard **15% sea-ice concentration threshold** for SIE.
- Grid-cell areas are taken from the corresponding `areacello` files.
- For consistency, avoid using `siarean` / `siareas` and `siextentn` / `siextents` output
- Includes handling of model-specific grid conventions (dimension naming only); no regridding is performed.
- Individual model files are merged into a single dataset and removed during the final cleanup step.


### 3. `compute_feedback-TOA-fluxes/<feedback>.py`

Computes TOA radiative feedback fields for individual climate feedback
components using the radiative kernel method implemented in the
`climkern` package.

The scripts operate on preprocessed CMIP6 monthly data generated by
the preceding preprocessing scripts. The required input variables
depend on the feedback component.

#### Purpose

For each CMIP6 model and each available radiative kernel, the scripts
calculate the radiative feedback contribution associated with an
individual physical process from `piControl` and `abrupt-4xCO2`
simulations.

Separate scripts are provided for:

- `Surface-albedo_feedback.py`
- `Temperature_feedbacks.py`
- `Water-vapour_feedback.py`
- Cloud-feedback inputs (computed separately)

Where applicable, both all-sky and clear-sky feedback components are
calculated.

#### Input data

The required input variables depend on the feedback component:

- **Surface-albedo feedback:** `rsus` and `rsds` from `piControl` and
  `abrupt-4xCO2`.
- **Temperature feedbacks:** `tas`, `ta` and `ps` from `piControl` and
  `abrupt-4xCO2`.
- **Water-vapour feedback:** `ta` (only `piControl`), `hus` and `ps` from `piControl` and `abrupt-4xCO2`.
- **Additional radiative fluxes used in the feedback analysis:**
  `rlut`, `rlutcs`, `rsut`, `rsutcs` and `rsdt`.

All required input variables are generated from the corresponding CMIP6
output by the preprocessing scripts described above.

#### Method

For each model:

- Reads the required preprocessed CMIP6 variables from the `piControl`
  and `abrupt-4xCO2` simulations.
- Computes the radiative feedback using the corresponding function from
  the `climkern` package.
- Repeats the calculation for each available radiative kernel.
- Stores the resulting monthly feedback fields on the native model grid.

Each model--kernel combination is processed independently.

#### Output

The scripts produce NetCDF files containing monthly feedback fields on
the native model grid. For example:

```
SA_<model>_<kernel>.nc
SA_cs_<model>_<kernel>.nc
PF_<model>_<kernel>.nc
PF_cs_<model>_<kernel>.nc
LRF_<model>_<kernel>.nc
LRF_cs_<model>_<kernel>.nc
WVF_<model>_<kernel>.nc
WVF_cs_<model>_<kernel>.nc
```

### 4) `compute_climate_feedbacks.py` and `compute_climate_feedbacks_on-grid.py`

Computes climate feedback parameters, effective radiative forcing (ERF), and equilibrium climate sensitivity (ECS) from the pre-calculated TOA feedback responses.

The scripts apply Gregory regression methods to CMIP6 abrupt-4xCO₂ experiments relative to piControl and derive:

1. **Global annual-mean feedbacks (`compute_climate_feedbacks.py`)**

   - Calculates ECS, ERF (SW/LW, all-sky/clear-sky), and net feedback parameters.
   - Derives individual kernel-based feedbacks:
     - Planck feedback (PF)
     - Lapse-rate feedback (LRF)
     - Water-vapour feedback (WVF)
     - Surface-albedo feedback (SAF)
   - Computes cloud feedbacks (CF_SW, CF_LW) using the adjustment method.
   - Stores regression variables (`dT`, `dR_*`) for diagnostics.

2. **Monthly and spatially resolved feedbacks (`compute_climate_feedbacks_on-grid.py`)**

   - Computes feedbacks on each model's native grid with monthly resolution.
   - Uses the global annual-mean temperature response as the regression predictor while retaining the monthly and spatial dimensions of the feedback fields.
   - Derives spatially resolved:
     - ERF and net feedbacks
     - Kernel-based feedback components
     - Cloud feedbacks
   - Remaps the native-grid results to a fixed 2.5° × 2.5° regular latitude–longitude grid for multi-model comparison and plotting.

Outputs are saved as NetCDF datasets containing climate sensitivity, forcing, and feedback diagnostics for models and kernels with the required input data.

Methods follow Gregory et al. (2004) for ECS/ERF estimation and Soden et al. (2008) for cloud feedback decomposition.

## 5) Observational sea ice data

Observational sea-ice concentration (SIC) products were downloaded from satellite-based datasets consistent with those used by Roach et al. (2020) for the evaluation of CMIP6 Antarctic sea ice.

SIC data were obtained from the Copernicus Climate Data Store for OSI SAF ([https://doi.org/10.24381/cds.3cd8b812](https://doi.org/10.24381/cds.3cd8b812)), and from the National Snow and Ice Data Center for the NASA Team ([https://doi.org/10.5067/MPYG15WAA4WX](https://doi.org/10.5067/MPYG15WAA4WX)) and Bootstrap ([https://doi.org/10.5067/X5LG68MH013O](https://doi.org/10.5067/X5LG68MH013O)) algorithms.

The downloaded monthly SIC files were time-merged prior to the calculation of sea-ice diagnostics.

`preprocess_sea_ice_obs.py`

Calculates Antarctic sea-ice area (SIA) and sea-ice extent (SIE) from the time-merged observational SIC datasets.

The script:

- calculates sea-ice area (SIA) from fractional SIC using the native 25 km grid-cell area,
- calculates sea-ice extent (SIE) using a 15% SIC threshold,
- restricts the analysis to 1979–2024,
- calculates annual-mean SIA and SIE time series for comparison with CMIP6 simulations,
- combines the three observational products into a single NetCDF dataset while retaining the individual products separately.



### OPTIONAL `Gregory-ECS_preprocessing.py`

If you only need ECS, the global net climate feedback parameter, and ERF from `tas`, `rlut`, `rsut`, and `rsdt`, this script provides a simplified and computationally efficient alternative to the full feedback-kernel workflow.

Computes the global-mean climate response of each CMIP model and derives **Equilibrium Climate Sensitivity (ECS)**, **Effective Radiative Forcing for 2×CO₂ (ERF₂ₓCO₂)**, and the **net climate feedback parameter (λ)** using the Gregory regression method.

#### Purpose

- Reads the preprocessed `piControl` and `abrupt-4xCO2` simulations generated by `cmip_preprocessing.py`.
- Identifies models with complete data for the required variables (`tas`, `rsdt`, `rsut`, `rlut`).
- Computes global-annual mean surface temperature and TOA radiative imbalance.
- Removes internal variability by subtracting a **21-year running mean** of the `piControl` simulation, following Caldwell et al. (2016) and Zelinka et al. (2020).
- Performs a Gregory regression of TOA radiative imbalance against global-mean surface temperature anomaly.
- Derives model-specific estimates of:
  - Equilibrium Climate Sensitivity (ECS)
  - Effective Radiative Forcing for 2×CO₂ (ERF₂ₓCO₂)
  - Net climate feedback parameter (λ)

The script also provides a rapid overview of the model-specific Gregory regressions, avoiding the substantially more computationally demanding feedback-kernel calculations.

#### Output

Creates a multi-model dataset:

```text
ecs_feedbacks_erf_allmodels.nc