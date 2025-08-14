#!/usr/bin/env python3
"""
Setup script for Aurora-MERS Forecasting System

Handles installation of dependencies, environment configuration,
and initial system setup on Sol supercomputer.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import yaml
import logging

def setup_logging():
    """Set up basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or higher is required")
    return True

def install_conda_environment(logger):
    """Install conda environment with required packages."""
    try:
        # Check if conda is available
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Conda not found. Please install Miniconda or Anaconda first.")
            return False
            
        logger.info("Creating conda environment: aurora-forecasting")
        
        # Create environment
        env_yaml = """
name: aurora-forecasting
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.10
  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda=11.8
  - numpy
  - scipy
  - pandas
  - xarray
  - netcdf4
  - yaml
  - matplotlib
  - cartopy
  - pip
  - pip:
    - microsoft-aurora
    - cdsapi
    - eccodes
    - cfgrib
    - schedule
    - psutil
    - fastapi
    - uvicorn
    - jinja2
        """
        
        # Write environment file
        env_file = Path("aurora_environment.yml")
        with open(env_file, 'w') as f:
            f.write(env_yaml)
            
        # Create environment
        cmd = ['conda', 'env', 'create', '-f', str(env_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Conda environment created successfully")
            env_file.unlink()  # Clean up
            return True
        else:
            logger.error(f"Failed to create conda environment: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error setting up conda environment: {e}")
        return False

def setup_ecmwf_credentials(logger):
    """Set up ECMWF CDS API credentials."""
    try:
        cdsapirc_path = Path.home() / ".cdsapirc"
        
        if cdsapirc_path.exists():
            logger.info("ECMWF credentials already configured")
            return True
            
        logger.info("Setting up ECMWF CDS API credentials")
        logger.info("Please visit https://cds.climate.copernicus.eu/api-how-to to get your API key")
        
        url = input("Enter CDS API URL (default: https://cds.climate.copernicus.eu/api): ").strip()
        if not url:
            url = "https://cds.climate.copernicus.eu/api"
            
        api_key = input("Enter your CDS API key: ").strip()
        
        if not api_key:
            logger.warning("No API key provided. You'll need to set this up manually later.")
            return False
            
        # Create .cdsapirc file
        with open(cdsapirc_path, 'w') as f:
            f.write(f"url: {url}\n")
            f.write(f"key: {api_key}\n")
            
        # Set appropriate permissions
        cdsapirc_path.chmod(0o600)
        
        logger.info(f"ECMWF credentials saved to {cdsapirc_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up ECMWF credentials: {e}")
        return False

def create_directories(config_path, logger):
    """Create necessary directories."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        logger.info("Creating project directories")
        
        for path_key, path_value in config['paths'].items():
            path_obj = Path(path_value)
            path_obj.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {path_obj}")
            
        logger.info("All directories created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        return False

def make_scripts_executable(logger):
    """Make all Python scripts executable."""
    try:
        logger.info("Making scripts executable")
        
        script_dirs = [
            "scripts/download",
            "scripts/processing", 
            "scripts/forecasting",
            "scripts/deployment"
        ]
        
        for script_dir in script_dirs:
            script_path = Path(script_dir)
            if script_path.exists():
                for script_file in script_path.glob("*.py"):
                    script_file.chmod(0o755)
                    logger.debug(f"Made executable: {script_file}")
                    
        logger.info("Scripts made executable")
        return True
        
    except Exception as e:
        logger.error(f"Error making scripts executable: {e}")
        return False

def test_aurora_installation(logger):
    """Test Aurora model installation."""
    try:
        logger.info("Testing Aurora installation")
        
        import torch
        from aurora import AuroraSmallPretrained, Batch, Metadata
        from datetime import datetime
        
        # Create a small test batch
        batch = Batch(
            surf_vars={k: torch.randn(1, 2, 17, 32) for k in ("2t", "10u", "10v", "msl")},
            static_vars={k: torch.randn(17, 32) for k in ("lsm", "z", "slt")},
            atmos_vars={k: torch.randn(1, 2, 4, 17, 32) for k in ("z", "u", "v", "t", "q")},
            metadata=Metadata(
                lat=torch.linspace(90, -90, 17),
                lon=torch.linspace(0, 360, 32 + 1)[:-1],
                time=(datetime(2020, 6, 1, 12, 0),),
                atmos_levels=(100, 250, 500, 850),
            ),
        )
        
        # Load model (this will download the checkpoint)
        logger.info("Loading Aurora model (this may take a while for first run)...")
        model = AuroraSmallPretrained()
        model.load_checkpoint()
        model.eval()
        
        # Test prediction
        with torch.inference_mode():
            prediction = model.forward(batch)
            
        logger.info("Aurora installation test successful")
        logger.info(f"Test prediction shape: {prediction.surf_vars['2t'].shape}")
        return True
        
    except Exception as e:
        logger.error(f"Aurora installation test failed: {e}")
        return False

def generate_slurm_submission_script(logger):
    """Generate a sample SLURM submission script."""
    try:
        logger.info("Generating SLURM submission script")
        
        script_content = """#!/bin/bash
# Sample submission script for Aurora-MERS forecasting
# Customize the account, partition, and resource requirements as needed

# Submit operational forecast
python scripts/deployment/sol_deployment_manager.py submit --mode operational

# Submit test forecast  
python scripts/deployment/sol_deployment_manager.py submit --mode test

# Check job status
python scripts/deployment/sol_deployment_manager.py list

# Manual forecast run (for testing)
python scripts/deployment/sol_deployment_manager.py submit --mode forecast-only
"""
        
        script_path = Path("submit_jobs.sh")
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        script_path.chmod(0o755)
        
        logger.info(f"SLURM submission script created: {script_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating SLURM script: {e}")
        return False

def main():
    """Main setup function."""
    logger = setup_logging()
    logger.info("Starting Aurora-MERS Forecasting System setup")
    
    try:
        # Check Python version
        check_python_version()
        logger.info("Python version check passed")
        
        # Check if we're in the right directory
        config_path = Path("config/system_config.yaml")
        if not config_path.exists():
            logger.error("Configuration file not found. Make sure you're in the project root directory.")
            return 1
            
        # Create directories
        if not create_directories(config_path, logger):
            logger.error("Failed to create directories")
            return 1
            
        # Make scripts executable
        if not make_scripts_executable(logger):
            logger.error("Failed to make scripts executable")
            return 1
            
        # Install conda environment
        logger.info("Setting up conda environment...")
        if not install_conda_environment(logger):
            logger.error("Failed to set up conda environment")
            return 1
            
        # Set up ECMWF credentials
        if not setup_ecmwf_credentials(logger):
            logger.warning("ECMWF credentials not configured. You'll need to do this manually.")
            
        # Test Aurora installation
        logger.info("Testing Aurora installation (this may take several minutes)...")
        if not test_aurora_installation(logger):
            logger.error("Aurora installation test failed")
            return 1
            
        # Generate SLURM submission script
        if not generate_slurm_submission_script(logger):
            logger.warning("Failed to generate SLURM submission script")
            
        logger.info("Setup completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Activate the conda environment: conda activate aurora-forecasting")
        logger.info("2. Configure ECMWF credentials if not done: edit ~/.cdsapirc")
        logger.info("3. Test data download: python scripts/download/ecmwf_downloader.py --mode latest")
        logger.info("4. Run a test forecast: python scripts/forecasting/aurora_forecaster.py --mode test")
        logger.info("5. Submit jobs to SLURM: python scripts/deployment/sol_deployment_manager.py submit --mode test")
        
        return 0
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())