# Project Organization Guide

This document outlines the reorganized structure of the Aurora weather steering research project.

## Directory Structure

```
docs/
├── weather_jiu_jitsu_research_summary.md    # Main research findings documentation
├── project_organization.md                  # This file
│
├── aurora_docs/                             # Official Aurora documentation
│   ├── README.md                           # Aurora documentation guide
│   ├── user_guides/                        # Getting started guides
│   │   ├── intro.md                        # Aurora introduction
│   │   └── usage.md                        # Basic usage instructions
│   ├── tutorials/                          # Step-by-step tutorials
│   │   ├── batch.md                        # Batch objects and data
│   │   └── tracking.md                     # Feature tracking
│   ├── advanced/                           # Advanced topics
│   │   ├── models.md                       # Model architecture details
│   │   └── finetuning.md                   # Model customization
│   └── api_reference/                      # API documentation
│       ├── ai_models_plugin.md             # Plugin integration
│       └── beware.md                       # Warnings and limitations
│
├── experiments/                             # Core research experiments
│   ├── README.md                           # Experiment overview and methodology
│   ├── TC/                                 # Tropical Cyclone experiments  
│   │   ├── baseline/
│   │   │   └── example_tc_tracking.ipynb   # Typhoon Nanmadol baseline tracking
│   │   └── steering_attempts/
│   │       ├── v_u_jet_slight.ipynb        # Main steering experiment
│   │       ├── change_u_48hrs.ipynb        # 48h u-wind perturbation
│   │       ├── change_u_72hrs.ipynb        # 72h u-wind perturbation  
│   │       ├── change_v_1time.ipynb        # Single-time v-wind test
│   │       ├── claude-steer.ipynb          # AI-assisted steering design
│   │       ├── u_jet_300.ipynb             # 300 hPa jet perturbation
│   │       └── u_jet_slight_shift.ipynb    # Jet displacement experiment
│   └── AR/                                 # Atmospheric River experiments
│       ├── analysis/
│       │   ├── ivt_25to29.ipynb            # Main AR analysis (Jan 2021)
│       │   ├── ivt_96hrs.ipynb             # Extended AR forecast
│       │   ├── ivt_test.ipynb              # AR detection tests
│       │   └── aurora_vs_era5_ivt.gif      # AR visualization animation
│       └── baseline/                       # (Reserved for future experiments)
│
├── examples/                               # General Aurora usage examples
│   ├── README.md                          # Examples documentation
│   ├── basic/                             # Basic Aurora tutorials
│   │   ├── example_era5.ipynb             # ERA5 data usage
│   │   ├── example_hres_0.1.ipynb         # High-resolution HRES
│   │   ├── example_hres_t0.ipynb          # HRES T+0 forecasting
│   │   └── example_cams.ipynb             # CAMS atmospheric composition
│   └── advanced/                          # (Reserved for advanced tutorials)
│
└── [other existing docs...]               # Existing Aurora documentation
    ├── foundry/
    ├── gifs/
    ├── api.rst
    ├── tracking.md
    ├── etc...
```

## Key Changes Made

### 1. Reorganization Summary
- **perturb-trials** → **TC** → **experiments/TC/steering_attempts**
- Added **experiments/TC/baseline** for reference tracking
- Created **experiments/AR/analysis** for atmospheric river work
- Moved basic examples to **examples/basic/**
- Added comprehensive documentation

### 2. File Movements
**Tropical Cyclone Files:**
- `example_tc_tracking.ipynb` → `experiments/TC/baseline/`
- All TC steering notebooks → `experiments/TC/steering_attempts/`

**Atmospheric River Files:**  
- `AR/ivt_25to29.ipynb` → `experiments/AR/analysis/`
- All other AR notebooks and visualizations → `experiments/AR/analysis/`

**Basic Examples:**
- `example_*.ipynb` files → `examples/basic/`

### 3. Documentation Added
- **`weather_jiu_jitsu_research_summary.md`**: Comprehensive research findings
- **`project_organization.md`**: This organizational guide
- **`experiments/README.md`**: Detailed experiment methodology and results  
- **`examples/README.md`**: Usage guide for basic tutorials
- **`aurora_docs/README.md`**: Official Aurora documentation guide

### 4. Documentation Organization
**Official Aurora Docs** moved to `aurora_docs/`:
- User guides: `intro.md`, `usage.md`
- Tutorials: `batch.md`, `tracking.md` 
- Advanced: `models.md`, `finetuning.md`
- API reference: `ai_models_plugin.md`, `beware.md`

**Research Documentation** kept at top level:
- Main findings and methodology summaries
- Project organization and quick reference guides

## Quick Reference

### Main Research Findings
📍 `weather_jiu_jitsu_research_summary.md`

### TC Steering Experiments
📍 `experiments/TC/steering_attempts/v_u_jet_slight.ipynb` (primary experiment)
📍 `experiments/TC/baseline/example_tc_tracking.ipynb` (reference)

### AR Analysis
📍 `experiments/AR/analysis/ivt_25to29.ipynb` (main AR study)

### Getting Started with Aurora
📍 `aurora_docs/user_guides/intro.md` (Aurora introduction)
📍 `aurora_docs/user_guides/usage.md` (Basic usage)
📍 `examples/basic/example_era5.ipynb` (Practical tutorial)

## Next Steps for Research

The organized structure now supports:
1. **Systematic experimentation**: Clear separation of baseline vs. perturbation experiments
2. **Reproducibility**: Documented methodology and findings
3. **Future expansion**: Reserved directories for additional experiment types
4. **Collaboration**: Clear documentation for other researchers

See `weather_jiu_jitsu_research_summary.md` for specific research recommendations and next experimental steps.