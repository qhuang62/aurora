# Aurora Examples

This directory contains example notebooks demonstrating basic and advanced usage of the Aurora weather prediction model.

## Basic Examples (`basic/`)

- **`example_era5.ipynb`**: Getting started with ERA5 data and basic Aurora predictions
- **`example_hres_0.1.ipynb`**: High-resolution HRES forecasting at 0.1° resolution  
- **`example_hres_t0.ipynb`**: HRES T+0 initialization and forecasting
- **`example_cams.ipynb`**: CAMS atmospheric composition forecasting

## Advanced Examples (`advanced/`)

*Reserved for future advanced tutorials and specialized use cases*

## Usage Notes

These examples provide the foundation for understanding Aurora's basic functionality before proceeding to the experimental weather steering research in the `experiments/` directory.

Each notebook includes:
- Data downloading and preprocessing
- Aurora model initialization and loading
- Basic prediction workflows  
- Visualization and analysis

## Prerequisites

- Aurora package installation
- Required API keys (CDS, etc.)
- Sufficient disk space for weather data downloads
- GPU access recommended for faster inference