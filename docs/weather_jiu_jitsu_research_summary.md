# Weather Jiu-Jitsu Research Summary
*Aurora Deep Learning Foundation Model for Extreme Weather Steering*

## Research Objective
Investigate the feasibility of using small, strategic perturbations in atmospheric initial conditions to influence the trajectories of extreme weather systems (tropical cyclones and atmospheric rivers) using the Aurora deep learning foundation model. The goal is to explore "Weather Jiu-Jitsu" - minimal energy interventions that could potentially redirect catastrophic weather events.

## Methodology Overview
The research employs Aurora, a transformer-based foundation model for weather prediction, to test whether upstream atmospheric perturbations can meaningfully alter the predicted tracks of extreme weather systems. Three main approaches were tested:

1. **Baseline Forecasting**: Standard Aurora predictions without intervention
2. **Tropical Cyclone Steering**: Perturbation of upstream steering flow
3. **Atmospheric River Analysis**: Evaluation of AR structure and evolution

## Experimental Notebooks Analysis

### 1. Baseline TC Tracking (`example_tc_tracking.ipynb`)

**System**: Typhoon Nanmadol (September 17, 2022)
**Method**: Standard Aurora forecasting workflow
- Data source: HRES T0 (WeatherBench2) + ERA5 static variables  
- Input variables: MSL pressure, 2m temperature, 10m winds, atmospheric profile (T, u, v, q, z)
- Forecast horizon: 48 hours (8 × 6-hour steps)
- Tracking: Aurora's built-in Tracker class for automatic TC center detection

**Key Results**:
- Successfully tracked TC center using MSL pressure minima
- Generated realistic storm structure and evolution
- Established baseline track for perturbation comparison

### 2. TC Steering Experiments (`TC/v_u_jet_slight.ipynb`)

**System**: Typhoon Nanmadol (September 14, 2022) - 3 days before landfall
**Perturbation Strategy**:
- **Target region**: 28°N, 120°E (~30° west of TC, upstream steering flow)
- **Spatial scale**: 17×17 grid points (~4°×4° area)
- **Vertical levels**: 850, 700, 500 hPa (steering flow levels)
- **Perturbation magnitude**: +80 m/s u-wind, +30 m/s v-wind
- **Duration**: Up to 48 hours across multiple pressure levels

**Results**:
- **Track deviation**: Only 2-6 km over 16 forecast steps (96 hours)
- **Effectiveness**: Minimal despite extreme perturbation magnitudes
- **Model behavior**: Aurora maintained meteorological consistency, dampening artificial perturbations

**Key Observation**: Even massive, unrealistic perturbations (80+ m/s) produced negligible steering effects, suggesting Aurora's internal physics strongly resist artificial modifications.

### 3. Atmospheric River Analysis (`AR/ivt_25to29.ipynb`)

**System**: California AR event (January 25-29, 2021)
**Method**: Forecast validation rather than steering
- Computed Integrated Vapor Transport (IVT) from Aurora predictions
- Compared against ERA5 reanalysis for validation
- 24-step rollout (96-hour forecast)

**Key Results**:
- Aurora accurately captured AR structure and temporal evolution
- RMSE and bias analysis showed reasonable forecast skill
- Regional mean IVT time series closely matched observations
- **No perturbation experiments conducted** - purely evaluative

## Summary of Findings

### Aurora Model Sensitivities
1. **Low track sensitivity**: Minimal response to large-scale perturbations
2. **Physical consistency**: Model maintains meteorological balance despite artificial inputs  
3. **Pressure field dominance**: Large-scale pressure gradients appear to control storm motion
4. **Internal stability**: Aurora's learned physics resist unphysical perturbations

### Common Challenges Across Experiments
- **Perturbation magnitude vs. effect**: Inverse relationship observed
- **Spatial/temporal targeting**: Current approach may miss dynamically critical scales
- **Lack of systematic exploration**: No parameter sensitivity analysis conducted
- **Evaluation metrics**: Distance-based metrics may be insensitive to subtle changes

### Research Gaps Identified
1. No systematic parameter space exploration
2. Missing analysis of dynamically relevant perturbation locations
3. Limited evaluation metrics (only track distance)
4. No investigation of geopotential height or vorticity perturbations
5. Insufficient understanding of Aurora's internal steering mechanisms

## Recommendations for Future Research

### Immediate Next Steps

#### 1. Systematic Parameter Study
- **Perturbation magnitudes**: Test realistic values (5, 10, 20, 40 m/s vs 80+ m/s)
- **Spatial scales**: Vary from 1°×1° to 5°×5° grid boxes
- **Temporal windows**: 6h, 12h, 24h, 48h perturbation durations
- **Variables**: Test geopotential height, vorticity, divergence (not just winds)

#### 2. Enhanced Targeting Strategy
- **Steering flow analysis**: Identify Rossby wave patterns and jet stream interactions
- **Ridge-trough targeting**: Focus on 500 hPa geopotential height anomalies
- **Beta-drift considerations**: Account for TC interaction with planetary vorticity gradient
- **Environmental wind shear**: Target vertical wind shear patterns

#### 3. Improved Evaluation Metrics
- **Track Angular Deviation (TAD)**: More sensitive than Euclidean distance
- **Landfall timing shifts**: Hours of difference in coastal impact
- **Intensity metrics**: Minimum central pressure, maximum wind speed changes
- **Recurvature analysis**: Changes in storm track curvature

#### 4. Recommended Next Experiment

**Target Event**: Typhoon Nanmadol (maintain consistency)
**Perturbation Design**:
- **Variable**: 500 hPa geopotential height (±25-50 gpm)
- **Location**: Ridge/trough pattern at 25-30°N, 115-125°E
- **Duration**: 24-hour initialization window
- **Assessment**: TAD, landfall shift, timing changes

**Code Framework**:
```python
def create_systematic_perturbation_batch(base_batch, pert_params):
    # pert_params = {variable, levels, lat_range, lon_range, magnitude, duration}
    return perturbed_batch

def evaluate_steering_effectiveness(orig_track, pert_track):
    # Return TAD, landfall_shift_km, timing_shift_hours, intensity_change
    return metrics_dict
```

### Long-term Research Directions

1. **Ensemble perturbation studies**: Multiple realizations for statistical significance
2. **Multi-model validation**: Compare Aurora results with other NWP models
3. **Physical mechanism analysis**: Understand Aurora's learned atmospheric dynamics
4. **Operational feasibility**: Assess real-world implementation constraints
5. **Ethical considerations**: Evaluate potential unintended consequences

## Conclusions

The initial Weather Jiu-Jitsu experiments demonstrate Aurora's capability for extreme weather prediction but reveal significant challenges in achieving meaningful steering through perturbations. The model's physical consistency and resistance to artificial modifications suggest that successful intervention strategies will require:

- More sophisticated understanding of atmospheric dynamics
- Targeted perturbations at dynamically critical scales
- Systematic exploration of parameter space
- Enhanced evaluation metrics sensitive to subtle but meaningful changes

While the current results show limited steering effectiveness, they establish a foundation for more systematic investigation into the conditions under which minimal interventions might influence extreme weather trajectories.

---

**Research Timeline**: 2024-2025  
**Model Version**: Aurora 0.25 (pretrained and finetuned checkpoints)  
**Primary Investigator**: Research team investigating ML-based weather intervention