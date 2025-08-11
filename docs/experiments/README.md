# Aurora Weather Steering Experiments

This directory contains the experimental notebooks for the Weather Jiu-Jitsu research project, investigating the potential for steering extreme weather systems using the Aurora deep learning foundation model.

## Directory Structure

### `TC/` - Tropical Cyclone Experiments
- **`baseline/`**: Standard Aurora TC tracking without perturbations
  - `example_tc_tracking.ipynb` - Typhoon Nanmadol baseline tracking (Sept 17, 2022)
  
- **`steering_attempts/`**: Perturbation-based TC steering experiments  
  - `v_u_jet_slight.ipynb` - Primary steering experiment with upstream wind perturbations
  - `change_u_48hrs.ipynb` - 48-hour u-wind perturbation experiment
  - `change_u_72hrs.ipynb` - Extended 72-hour u-wind perturbation
  - `change_v_1time.ipynb` - Single-time v-wind perturbation test
  - `claude-steer.ipynb` - AI-assisted steering experiment design
  - `u_jet_300.ipynb` - 300 hPa jet stream perturbation
  - `u_jet_slight_shift.ipynb` - Slight jet stream displacement experiment

### `AR/` - Atmospheric River Experiments
- **`analysis/`**: AR structure and evolution analysis
  - `ivt_25to29.ipynb` - Primary AR analysis (Jan 25-29, 2021 California event)
  - `ivt_96hrs.ipynb` - Extended 96-hour AR forecast
  - `ivt_test.ipynb` - AR detection and tracking tests
  - `aurora_vs_era5_ivt.gif` - Animated comparison of Aurora vs ERA5 IVT

- **`baseline/`**: (Reserved for future AR baseline experiments)

## Key Research Findings

### Tropical Cyclone Steering
- **Limited effectiveness**: Large perturbations (80+ m/s) produced minimal track deviations (2-6 km)
- **Model resilience**: Aurora maintains physical consistency, dampening artificial perturbations
- **Need for systematic approach**: Current ad-hoc perturbations miss dynamically critical scales

### Atmospheric River Analysis  
- **Good forecast skill**: Aurora accurately captures AR structure and evolution
- **Validation capability**: IVT comparisons with ERA5 show reasonable agreement
- **Steering potential**: AR experiments focused on analysis rather than perturbation

## Methodology Notes

### Perturbation Strategy (TC Experiments)
- **Target regions**: Upstream steering flow, typically 30° west of storm center
- **Vertical levels**: Focus on 850-500 hPa (primary steering levels)
- **Spatial scales**: Usually 17×17 grid points (~4°×4°)
- **Variables**: Primarily u/v wind components, some geopotential height

### Evaluation Metrics
- **Track deviation**: Euclidean distance between perturbed and original tracks
- **Visual comparison**: Side-by-side MSL pressure field evolution
- **Time series analysis**: Step-by-step deviation quantification

## Recommended Next Steps

1. **Systematic parameter exploration**: Test realistic perturbation magnitudes (5-40 m/s)
2. **Enhanced targeting**: Focus on geopotential height and vorticity fields  
3. **Improved metrics**: Implement Track Angular Deviation (TAD) and landfall timing
4. **Physical understanding**: Analyze Aurora's learned atmospheric dynamics

## Usage Instructions

Each notebook is self-contained but may require:
- Aurora model checkpoints (`microsoft/aurora`)
- ERA5/HRES data downloads (automated via cdsapi/WeatherBench2)
- GPU memory for model inference (~40GB recommended)

For systematic experiments, consider using the batch processing framework outlined in the main research summary document.