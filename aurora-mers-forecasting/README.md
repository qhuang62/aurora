# Aurora-ERA5 Automated Weather Forecasting System

A comprehensive automated weather forecasting system using Microsoft's Aurora foundation model with ECMWF ERA5 reanalysis data, designed for deployment on ASU's Sol supercomputer.

## Overview

This system establishes ASU as the first university to offer operational AI-driven weather forecasting using the Aurora foundation model. The system provides:

- **Automated Data Ingestion**: ECMWF ERA5 reanalysis data downloaded every 12 hours
- **High-Resolution Forecasting**: 1-2 week forecasts using Aurora model
- **Regional Analysis**: Customizable regional domains with high-resolution interpolation
- **Extreme Weather Detection**: Automated detection of heat waves, high winds, and severe weather
- **Web Interface**: Public-facing website for forecast dissemination
- **Supercomputer Integration**: Optimized for Sol's GPU compute nodes

## Quick Start

### 1. Initial Setup

```bash
# Clone or navigate to the project directory
cd /home/qhuang62/aurora/aurora-mers-forecasting

# Run the automated setup
python setup.py
```

### 2. Activate Environment

```bash
# Activate the conda environment
conda activate aurora-forecasting
```

### 3. Configure ECMWF Access

Edit `~/.cdsapirc` with your ECMWF credentials:
```
url: https://cds.climate.copernicus.eu/api
key: YOUR_API_KEY_HERE
```

Get your API key from: https://cds.climate.copernicus.eu/api-how-to

### 4. Test the System

```bash
# Test data download
python scripts/download/ecmwf_downloader.py --mode latest

# Test data processing
python scripts/processing/aurora_data_converter.py --mode latest

# Test forecast generation
python scripts/forecasting/aurora_forecaster.py --mode test --test-steps 5
```

### 5. Submit to SLURM

```bash
# Submit a test job
python scripts/deployment/sol_deployment_manager.py submit --mode test

# Submit operational forecast
python scripts/deployment/sol_deployment_manager.py submit --mode operational

# Check job status
python scripts/deployment/sol_deployment_manager.py list
```

## System Architecture

```
ECMWF MARS → Data Download → Aurora Processing → Regional Analysis → Web Output
     ↓             ↓              ↓              ↓             ↓
  12hr cycle   Sol storage    GPU compute    Interpolation   Website
```

### Core Components

1. **Data Pipeline** (`scripts/download/`)
   - `ecmwf_downloader.py`: Automated ECMWF data download
   - `data_scheduler.py`: 12-hour scheduling system

2. **Processing Engine** (`scripts/processing/`)
   - `aurora_data_converter.py`: Convert ECMWF data to Aurora format
   - `regional_processor.py`: Regional interpolation and analysis

3. **Forecasting System** (`scripts/forecasting/`)
   - `aurora_forecaster.py`: Aurora model execution and forecast generation

4. **Deployment Tools** (`scripts/deployment/`)
   - `sol_deployment_manager.py`: SLURM job management
   - `sol_job_template.slurm`: SLURM job template

5. **Configuration** (`config/`)
   - `system_config.yaml`: Central system configuration

## Configuration

The system is configured through `config/system_config.yaml`. Key sections:

### ECMWF Data Configuration
```yaml
ecmwf:
  update_frequency: "0 */12 * * *"  # Every 12 hours
  surface_variables:
    - "2m_temperature"
    - "10m_u_component_of_wind"
    - "mean_sea_level_pressure"
  grid: "0.25/0.25"  # 0.25 degree resolution
```

### Aurora Model Configuration
```yaml
aurora:
  model_name: "Aurora"
  device: "cuda"
  forecast_steps: 56  # 14 days at 6-hour intervals
  memory_optimization: true
```

### Regional Domains
```yaml
regions:
  conus:
    name: "Continental US"
    bbox: [20, -130, 55, -65]
    resolution: 0.1
  arizona:
    name: "Arizona"
    bbox: [31, -115, 37, -109]
    resolution: 0.01
```

## Usage Examples

### Manual Operations

#### Download Latest Data
```bash
python scripts/download/ecmwf_downloader.py --mode latest --cleanup
```

#### Process Specific Date
```bash
python scripts/processing/aurora_data_converter.py --mode date --date "2024-01-15 12:00"
```

#### Generate Regional Forecasts
```bash
python scripts/processing/regional_processor.py --mode latest --region arizona
```

#### Run Complete Pipeline
```bash
python scripts/download/data_scheduler.py --mode manual
```

### SLURM Operations

#### Submit Different Job Types
```bash
# Operational forecast (full pipeline)
python scripts/deployment/sol_deployment_manager.py submit --mode operational

# Data download only
python scripts/deployment/sol_deployment_manager.py submit --mode data-only

# Forecast generation only
python scripts/deployment/sol_deployment_manager.py submit --mode forecast-only

# Test run with limited steps
python scripts/deployment/sol_deployment_manager.py submit --mode test
```

#### Monitor Jobs
```bash
# List active jobs
python scripts/deployment/sol_deployment_manager.py list

# Check specific job status
python scripts/deployment/sol_deployment_manager.py status JOB_ID

# Monitor job performance
python scripts/deployment/sol_deployment_manager.py monitor JOB_ID

# Cancel job
python scripts/deployment/sol_deployment_manager.py cancel JOB_ID
```

### Automated Scheduling

#### Set up 12-hour recurring jobs
```bash
python scripts/deployment/sol_deployment_manager.py schedule --frequency 12
```

This creates a cron job that automatically submits forecasting jobs every 12 hours.

## Output Products

### Forecast Files
- **Global Forecasts**: `output/forecasts/complete_forecast_YYYYMMDD_HHMM.nc`
- **Regional Forecasts**: `output/forecasts/regional/regional_forecast_REGION_YYYYMMDD_HHMM.nc`
- **Extreme Events**: `output/forecasts/regional/extreme_events_REGION_YYYYMMDD_HHMM.yaml`

### Visualizations
- **Global Maps**: Temperature, pressure, wind patterns
- **Regional Analysis**: High-resolution regional forecasts
- **Time Series**: Forecast evolution plots
- **Extreme Weather**: Heat wave and severe weather maps

### Data Products
- **NetCDF Files**: CF-compliant forecast data
- **JSON/YAML**: Metadata and event summaries
- **CSV**: Time series and statistics

## System Requirements

### Computational Resources
- **GPU**: NVIDIA A100 or equivalent (40GB+ memory recommended)
- **CPU**: 16+ cores for data processing
- **Memory**: 64GB+ RAM
- **Storage**: 50TB+ for operational data retention
- **Network**: High-bandwidth for ECMWF downloads

### Software Dependencies
- **Python 3.10+**
- **PyTorch with CUDA support**
- **Aurora model** (`microsoft-aurora`)
- **Scientific Python stack** (numpy, scipy, xarray, etc.)
- **ECMWF tools** (cdsapi, eccodes, cfgrib)
- **SLURM** for job scheduling

## Monitoring and Maintenance

### Log Files
- **System Logs**: `logs/aurora_forecaster.log`
- **Download Logs**: `logs/ecmwf_downloader.log`
- **Job Logs**: `logs/sol_deployment.log`
- **SLURM Output**: `logs/aurora_forecast_*.out`

### Status Monitoring
```bash
# Check system status
python scripts/forecasting/aurora_forecaster.py --status

# View pipeline status
cat logs/pipeline_status.yaml

# Monitor disk usage
df -h /home/qhuang62/aurora/aurora-mers-forecasting/data
```

### Maintenance Tasks
```bash
# Clean up old data files
python scripts/download/ecmwf_downloader.py --cleanup

# Update forecast skill metrics
python scripts/processing/forecast_validator.py --mode latest

# Generate system health report
python scripts/deployment/system_monitor.py --report
```

## Troubleshooting

### Common Issues

#### ECMWF Download Failures
```bash
# Check credentials
cat ~/.cdsapirc

# Test API access
python -c "import cdsapi; c = cdsapi.Client(); print('API access OK')"

# Check download logs
tail -f logs/ecmwf_downloader.log
```

#### Aurora Model Issues
```bash
# Check GPU availability
nvidia-smi

# Test model loading
python -c "from aurora import AuroraSmallPretrained; print('Model import OK')"

# Check forecast logs
tail -f logs/aurora_forecaster.log
```

#### SLURM Job Problems
```bash
# Check SLURM status
squeue -u $USER

# View job details
scontrol show job JOB_ID

# Check job output
cat logs/aurora_forecast_JOB_ID.out
```

### Performance Optimization

#### GPU Memory Issues
- Reduce `forecast_steps` in configuration
- Enable `memory_optimization: true`
- Use smaller batch sizes for regional processing

#### Slow Downloads
- Check network connectivity to ECMWF
- Reduce spatial/temporal resolution if needed
- Use parallel download workers

#### Storage Management
- Implement automated data archival
- Use compression for older forecast files
- Monitor disk usage regularly

## Development and Customization

### Adding New Regions
1. Edit `config/system_config.yaml`:
```yaml
regions:
  new_region:
    name: "New Region"
    bbox: [south, west, north, east]
    resolution: 0.05
```

2. Test regional processing:
```bash
python scripts/processing/regional_processor.py --region new_region
```

### Custom Variables
1. Modify variable lists in `config/system_config.yaml`
2. Update processing scripts to handle new variables
3. Test with sample data

### Additional Models
1. Integrate new models in `scripts/forecasting/`
2. Update configuration schema
3. Modify deployment scripts for new resource requirements

## Contributing

### Code Structure
- Follow PEP 8 style guidelines
- Add comprehensive logging
- Include error handling and validation
- Write docstrings for all functions

### Testing
```bash
# Run unit tests (when available)
python -m pytest tests/

# Run integration tests
python scripts/forecasting/aurora_forecaster.py --mode test

# Validate configuration
python -c "import yaml; yaml.safe_load(open('config/system_config.yaml'))"
```

## Support and Documentation

### Getting Help
- **Issues**: Report bugs and feature requests via GitHub issues
- **Documentation**: See `docs/` directory for detailed guides
- **Configuration**: Review `config/system_config.yaml` comments

### Useful Resources
- **Aurora Model**: https://github.com/microsoft/aurora
- **ECMWF API**: https://cds.climate.copernicus.eu/api-how-to
- **Sol Documentation**: ASU Research Computing documentation
- **SLURM**: https://slurm.schedmd.com/documentation.html

## License and Acknowledgments

This project builds upon:
- **Microsoft Aurora**: Foundation weather prediction model
- **ECMWF**: Reanalysis data products
- **ASU Sol**: Supercomputing infrastructure

Please cite appropriate sources when using this system for research or operational purposes.

---

**Contact**: qhuang62@asu.edu
**Last Updated**: August 13 2025
**Version**: 1.0.0