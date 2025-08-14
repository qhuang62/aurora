#!/usr/bin/env python3
"""
Aurora Data Converter for ERA5-Aurora Forecasting System

Converts downloaded ECMWF ERA5 data to Aurora Batch format for
use with the Aurora weather forecasting model.
"""

import os
import sys
import yaml
import logging
import argparse
import numpy as np
import xarray as xr
import torch
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import Aurora components
from aurora import Batch, Metadata

class AuroraDataConverter:
    """
    Convert ECMWF data to Aurora Batch format.
    
    Handles the conversion of downloaded ECMWF ERA5 reanalysis data
    into the format required by the Aurora weather prediction model.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the data converter.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "system_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_logging()
        self.setup_variable_mapping()
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = Path(self.config['paths']['logs'])
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(log_dir / 'data_converter.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_variable_mapping(self):
        """Set up mapping between ECMWF variable names and Aurora names."""
        self.surface_var_mapping = {
            '2m_temperature': '2t',
            '10m_u_component_of_wind': '10u',
            '10m_v_component_of_wind': '10v',
            'mean_sea_level_pressure': 'msl'
        }
        
        self.atmospheric_var_mapping = {
            'temperature': 't',
            'u_component_of_wind': 'u', 
            'v_component_of_wind': 'v',
            'specific_humidity': 'q',
            'geopotential': 'z'
        }
        
        self.static_var_mapping = {
            'geopotential': 'z',
            'land_sea_mask': 'lsm',
            'soil_type': 'slt'
        }
        
    def find_latest_data_files(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Find the most recent surface and atmospheric data files.
        
        Returns:
            Tuple of (surface_file, atmospheric_file) paths
        """
        raw_data_dir = Path(self.config['paths']['raw_data'])
        
        surface_files = list(raw_data_dir.glob("*_surface.nc"))
        atmospheric_files = list(raw_data_dir.glob("*_atmospheric.nc"))
        
        if not surface_files or not atmospheric_files:
            self.logger.error("No surface or atmospheric data files found")
            return None, None
            
        # Sort by modification time and get the latest
        surface_file = max(surface_files, key=lambda x: x.stat().st_mtime)
        atmospheric_file = max(atmospheric_files, key=lambda x: x.stat().st_mtime)
        
        self.logger.info(f"Latest surface file: {surface_file}")
        self.logger.info(f"Latest atmospheric file: {atmospheric_file}")
        
        return surface_file, atmospheric_file
        
    def find_data_files_for_date(self, target_date: datetime) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Find data files for a specific date.
        
        Args:
            target_date: Target date to find files for
            
        Returns:
            Tuple of (surface_file, atmospheric_file) paths
        """
        raw_data_dir = Path(self.config['paths']['raw_data'])
        date_str = target_date.strftime('%Y%m%d_%H%M')
        
        surface_file = raw_data_dir / f"{date_str}_surface.nc"
        atmospheric_file = raw_data_dir / f"{date_str}_atmospheric.nc"
        
        if not surface_file.exists() or not atmospheric_file.exists():
            self.logger.error(f"Data files not found for {target_date}")
            return None, None
            
        return surface_file, atmospheric_file
        
    def load_static_variables(self) -> Dict[str, torch.Tensor]:
        """
        Load static variables from file.
        
        Returns:
            Dictionary of static variables as tensors
        """
        static_file = Path(self.config['paths']['static_data']) / "static.nc"
        
        if not static_file.exists():
            self.logger.error(f"Static variables file not found: {static_file}")
            raise FileNotFoundError(f"Static variables file not found: {static_file}")
            
        try:
            ds = xr.open_dataset(static_file, engine="netcdf4")
            
            static_vars = {}
            for ecmwf_name, aurora_name in self.static_var_mapping.items():
                if ecmwf_name in ds:
                    # Get the first time step for static variables
                    data = ds[ecmwf_name].isel(time=0).values
                    
                    # Ensure latitudes are decreasing (Aurora requirement)
                    if ds.latitude.values[0] < ds.latitude.values[-1]:
                        data = data[::-1, :]
                        
                    static_vars[aurora_name] = torch.from_numpy(data.copy()).float()
                else:
                    self.logger.warning(f"Static variable {ecmwf_name} not found in dataset")
                    
            self.logger.info(f"Loaded {len(static_vars)} static variables")
            return static_vars
            
        except Exception as e:
            self.logger.error(f"Failed to load static variables: {e}")
            raise
            
    def prepare_surface_variables(self, surface_file: Path, num_history_steps: int = 2) -> Dict[str, torch.Tensor]:
        """
        Load and prepare surface variables.
        
        Args:
            surface_file: Path to surface variables file
            num_history_steps: Number of time steps to include
            
        Returns:
            Dictionary of surface variables as tensors
        """
        try:
            ds = xr.open_dataset(surface_file, engine="netcdf4")
            
            surf_vars = {}
            for ecmwf_name, aurora_name in self.surface_var_mapping.items():
                if ecmwf_name in ds:
                    # Get the required number of time steps
                    data = ds[ecmwf_name].values
                    
                    if data.shape[0] < num_history_steps:
                        self.logger.warning(f"Not enough time steps for {ecmwf_name}: {data.shape[0]} < {num_history_steps}")
                        # Pad with the last available time step
                        while data.shape[0] < num_history_steps:
                            data = np.concatenate([data, data[-1:]], axis=0)
                    
                    # Take only the required number of time steps
                    data = data[:num_history_steps]
                    
                    # Ensure latitudes are decreasing (Aurora requirement)
                    if ds.latitude.values[0] < ds.latitude.values[-1]:
                        data = data[:, ::-1, :]
                        
                    # Add batch dimension and convert to tensor: (batch, time, lat, lon)
                    surf_vars[aurora_name] = torch.from_numpy(data[None, ...].copy()).float()
                else:
                    self.logger.warning(f"Surface variable {ecmwf_name} not found in dataset")
                    
            self.logger.info(f"Loaded {len(surf_vars)} surface variables with shape {list(surf_vars.values())[0].shape if surf_vars else 'N/A'}")
            return surf_vars
            
        except Exception as e:
            self.logger.error(f"Failed to load surface variables from {surface_file}: {e}")
            raise
            
    def prepare_atmospheric_variables(self, atmospheric_file: Path, num_history_steps: int = 2) -> Dict[str, torch.Tensor]:
        """
        Load and prepare atmospheric variables.
        
        Args:
            atmospheric_file: Path to atmospheric variables file
            num_history_steps: Number of time steps to include
            
        Returns:
            Dictionary of atmospheric variables as tensors
        """
        try:
            ds = xr.open_dataset(atmospheric_file, engine="netcdf4")
            
            atmos_vars = {}
            for ecmwf_name, aurora_name in self.atmospheric_var_mapping.items():
                if ecmwf_name in ds:
                    # Get the required number of time steps
                    data = ds[ecmwf_name].values
                    
                    if data.shape[0] < num_history_steps:
                        self.logger.warning(f"Not enough time steps for {ecmwf_name}: {data.shape[0]} < {num_history_steps}")
                        # Pad with the last available time step
                        while data.shape[0] < num_history_steps:
                            data = np.concatenate([data, data[-1:]], axis=0)
                    
                    # Take only the required number of time steps
                    data = data[:num_history_steps]
                    
                    # Ensure latitudes are decreasing (Aurora requirement)
                    if ds.latitude.values[0] < ds.latitude.values[-1]:
                        data = data[:, :, ::-1, :]
                        
                    # Add batch dimension and convert to tensor: (batch, time, level, lat, lon)
                    atmos_vars[aurora_name] = torch.from_numpy(data[None, ...].copy()).float()
                else:
                    self.logger.warning(f"Atmospheric variable {ecmwf_name} not found in dataset")
                    
            self.logger.info(f"Loaded {len(atmos_vars)} atmospheric variables with shape {list(atmos_vars.values())[0].shape if atmos_vars else 'N/A'}")
            return atmos_vars
            
        except Exception as e:
            self.logger.error(f"Failed to load atmospheric variables from {atmospheric_file}: {e}")
            raise
            
    def create_metadata(self, surface_file: Path) -> Metadata:
        """
        Create metadata object for Aurora batch.
        
        Args:
            surface_file: Path to surface variables file (for time/coordinates)
            
        Returns:
            Metadata object
        """
        try:
            ds = xr.open_dataset(surface_file, engine="netcdf4")
            
            # Get coordinates
            lats = ds.latitude.values
            lons = ds.longitude.values
            
            # Ensure latitudes are decreasing
            if lats[0] < lats[-1]:
                lats = lats[::-1]
                
            # Get time information - use the last time step as the current time
            times = ds.time.values
            current_time = pd.to_datetime(times[-1]).to_pydatetime()
            
            # Get pressure levels from config
            pressure_levels = tuple(self.config['ecmwf']['pressure_levels'])
            
            metadata = Metadata(
                lat=torch.from_numpy(lats.copy()).float(),
                lon=torch.from_numpy(lons.copy()).float(),
                time=(current_time,),
                atmos_levels=pressure_levels
            )
            
            self.logger.info(f"Created metadata with {len(lats)} latitudes, {len(lons)} longitudes, time: {current_time}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to create metadata: {e}")
            raise
            
    def create_aurora_batch(self, surface_file: Path, atmospheric_file: Path) -> Batch:
        """
        Create a complete Aurora Batch from data files.
        
        Args:
            surface_file: Path to surface variables file
            atmospheric_file: Path to atmospheric variables file
            
        Returns:
            Aurora Batch object
        """
        try:
            # Load all variable types
            static_vars = self.load_static_variables()
            surf_vars = self.prepare_surface_variables(surface_file)
            atmos_vars = self.prepare_atmospheric_variables(atmospheric_file)
            metadata = self.create_metadata(surface_file)
            
            # Create Aurora batch
            batch = Batch(
                surf_vars=surf_vars,
                static_vars=static_vars,
                atmos_vars=atmos_vars,
                metadata=metadata
            )
            
            self.logger.info("Successfully created Aurora batch")
            return batch
            
        except Exception as e:
            self.logger.error(f"Failed to create Aurora batch: {e}")
            raise
            
    def save_aurora_batch(self, batch: Batch, output_path: Path):
        """
        Save Aurora batch to file.
        
        Args:
            batch: Aurora batch to save
            output_path: Output file path
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            batch.to_netcdf(output_path)
            self.logger.info(f"Saved Aurora batch to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save Aurora batch to {output_path}: {e}")
            raise
            
    def convert_latest_data(self) -> Optional[Path]:
        """
        Convert the latest available data to Aurora format.
        
        Returns:
            Path to the output file if successful
        """
        surface_file, atmospheric_file = self.find_latest_data_files()
        
        if surface_file is None or atmospheric_file is None:
            self.logger.error("Could not find latest data files")
            return None
            
        try:
            # Create Aurora batch
            batch = self.create_aurora_batch(surface_file, atmospheric_file)
            
            # Generate output filename
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M')
            output_file = Path(self.config['paths']['processed_data']) / f"aurora_batch_{timestamp}.nc"
            
            # Save batch
            self.save_aurora_batch(batch, output_file)
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to convert latest data: {e}")
            return None
            
    def convert_data_for_date(self, target_date: datetime) -> Optional[Path]:
        """
        Convert data for a specific date to Aurora format.
        
        Args:
            target_date: Date to convert data for
            
        Returns:
            Path to the output file if successful
        """
        surface_file, atmospheric_file = self.find_data_files_for_date(target_date)
        
        if surface_file is None or atmospheric_file is None:
            self.logger.error(f"Could not find data files for {target_date}")
            return None
            
        try:
            # Create Aurora batch
            batch = self.create_aurora_batch(surface_file, atmospheric_file)
            
            # Generate output filename
            timestamp = target_date.strftime('%Y%m%d_%H%M')
            output_file = Path(self.config['paths']['processed_data']) / f"aurora_batch_{timestamp}.nc"
            
            # Save batch
            self.save_aurora_batch(batch, output_file)
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to convert data for {target_date}: {e}")
            return None
            
    def validate_aurora_batch(self, batch: Batch) -> bool:
        """
        Validate the created Aurora batch.
        
        Args:
            batch: Aurora batch to validate
            
        Returns:
            True if valid
        """
        try:
            # Check required surface variables
            required_surf_vars = set(self.surface_var_mapping.values())
            available_surf_vars = set(batch.surf_vars.keys())
            missing_surf_vars = required_surf_vars - available_surf_vars
            
            if missing_surf_vars:
                self.logger.error(f"Missing surface variables: {missing_surf_vars}")
                return False
                
            # Check required atmospheric variables
            required_atmos_vars = set(self.atmospheric_var_mapping.values())
            available_atmos_vars = set(batch.atmos_vars.keys())
            missing_atmos_vars = required_atmos_vars - available_atmos_vars
            
            if missing_atmos_vars:
                self.logger.error(f"Missing atmospheric variables: {missing_atmos_vars}")
                return False
                
            # Check required static variables
            required_static_vars = set(self.static_var_mapping.values())
            available_static_vars = set(batch.static_vars.keys())
            missing_static_vars = required_static_vars - available_static_vars
            
            if missing_static_vars:
                self.logger.error(f"Missing static variables: {missing_static_vars}")
                return False
                
            # Check data shapes and types
            for var_name, tensor in batch.surf_vars.items():
                if not isinstance(tensor, torch.Tensor):
                    self.logger.error(f"Surface variable {var_name} is not a tensor")
                    return False
                if tensor.dim() != 4:  # (batch, time, lat, lon)
                    self.logger.error(f"Surface variable {var_name} has wrong dimensions: {tensor.shape}")
                    return False
                    
            for var_name, tensor in batch.atmos_vars.items():
                if not isinstance(tensor, torch.Tensor):
                    self.logger.error(f"Atmospheric variable {var_name} is not a tensor")
                    return False
                if tensor.dim() != 5:  # (batch, time, level, lat, lon)
                    self.logger.error(f"Atmospheric variable {var_name} has wrong dimensions: {tensor.shape}")
                    return False
                    
            self.logger.info("Aurora batch validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during batch validation: {e}")
            return False


def main():
    """Main function for command line usage."""
    import pandas as pd
    
    parser = argparse.ArgumentParser(description="Aurora Data Converter")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--mode", choices=['latest', 'date'], default='latest',
                       help="Conversion mode: latest data or specific date")
    parser.add_argument("--date", type=str, help="Date for conversion (YYYY-MM-DD HH:MM)")
    parser.add_argument("--validate", action='store_true', help="Validate created batch")
    
    args = parser.parse_args()
    
    try:
        converter = AuroraDataConverter(args.config)
        
        if args.mode == 'latest':
            output_file = converter.convert_latest_data()
        elif args.mode == 'date':
            if not args.date:
                parser.error("Date mode requires --date argument")
                
            target_date = datetime.strptime(args.date, '%Y-%m-%d %H:%M')
            output_file = converter.convert_data_for_date(target_date)
            
        if output_file:
            print(f"Conversion successful: {output_file}")
            
            if args.validate:
                batch = Batch.from_netcdf(output_file)
                if converter.validate_aurora_batch(batch):
                    print("Batch validation passed")
                else:
                    print("Batch validation failed")
                    sys.exit(1)
                    
            sys.exit(0)
        else:
            print("Conversion failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()