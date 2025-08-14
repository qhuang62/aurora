#!/usr/bin/env python3
"""
Regional Processor for Aurora-MERS Forecasting System

Handles regional focusing, high-resolution interpolation, and 
specialized processing for areas of interest.
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
from scipy.interpolate import RegularGridInterpolator
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import Aurora components
from aurora import Batch, Metadata

class RegionalProcessor:
    """
    Regional weather forecast processor.
    
    Provides high-resolution interpolation, regional extraction,
    and specialized analysis for defined areas of interest.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the regional processor.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "system_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_logging()
        self.setup_regions()
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = Path(self.config['paths']['logs'])
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(log_dir / 'regional_processor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_regions(self):
        """Set up regional domain configurations."""
        self.regions = self.config['regions']
        self.logger.info(f"Configured {len(self.regions)} regional domains")
        
        for region_name, region_config in self.regions.items():
            bbox = region_config['bbox']
            self.logger.debug(f"Region {region_name}: {bbox} at {region_config['resolution']}° resolution")
            
    def find_latest_forecast(self) -> Optional[Path]:
        """
        Find the latest complete forecast file.
        
        Returns:
            Path to latest forecast file
        """
        forecast_dir = Path(self.config['paths']['forecasts'])
        forecast_files = list(forecast_dir.glob("complete_forecast_*.nc"))
        
        if not forecast_files:
            self.logger.error("No complete forecast files found")
            return None
            
        # Sort by modification time and get the latest
        latest_file = max(forecast_files, key=lambda x: x.stat().st_mtime)
        self.logger.info(f"Using latest forecast: {latest_file}")
        
        return latest_file
        
    def load_forecast_data(self, forecast_file: Path) -> Optional[xr.Dataset]:
        """
        Load forecast data from NetCDF file.
        
        Args:
            forecast_file: Path to forecast file
            
        Returns:
            Loaded forecast dataset
        """
        try:
            self.logger.info(f"Loading forecast data from {forecast_file}")
            ds = xr.open_dataset(forecast_file)
            
            self.logger.info(f"Loaded forecast with {len(ds.time)} time steps")
            self.logger.info(f"Variables: {list(ds.data_vars.keys())}")
            self.logger.info(f"Spatial extent: {ds.latitude.min().values:.2f}° to {ds.latitude.max().values:.2f}°N, "
                           f"{ds.longitude.min().values:.2f}° to {ds.longitude.max().values:.2f}°E")
            
            return ds
            
        except Exception as e:
            self.logger.error(f"Failed to load forecast data: {e}")
            return None
            
    def extract_regional_domain(self, dataset: xr.Dataset, region_name: str) -> Optional[xr.Dataset]:
        """
        Extract a regional domain from the global forecast.
        
        Args:
            dataset: Global forecast dataset
            region_name: Name of region to extract
            
        Returns:
            Regional dataset
        """
        try:
            if region_name not in self.regions:
                self.logger.error(f"Unknown region: {region_name}")
                return None
                
            region_config = self.regions[region_name]
            bbox = region_config['bbox']  # [south, west, north, east]
            
            # Convert negative longitudes to 0-360 convention if needed
            lon_west = bbox[1] if bbox[1] >= 0 else bbox[1] + 360
            lon_east = bbox[3] if bbox[3] >= 0 else bbox[3] + 360
            
            # Extract spatial subset
            if lon_west > lon_east:  # Crosses 0° meridian
                # Handle wraparound case (e.g., 350° to 10°)
                regional_ds1 = dataset.sel(
                    latitude=slice(bbox[2], bbox[0]),
                    longitude=slice(lon_west, 360)
                )
                regional_ds2 = dataset.sel(
                    latitude=slice(bbox[2], bbox[0]),
                    longitude=slice(0, lon_east)
                )
                regional_ds = xr.concat([regional_ds1, regional_ds2], dim='longitude')
            else:
                regional_ds = dataset.sel(
                    latitude=slice(bbox[2], bbox[0]),  # north to south (xarray wants decreasing for slice)
                    longitude=slice(lon_west, lon_east)  # west to east
                )
            
            self.logger.info(f"Extracted {region_name} domain: "
                           f"{len(regional_ds.latitude)} x {len(regional_ds.longitude)} grid points")
            
            # Add regional metadata
            regional_ds.attrs.update({
                'region_name': region_name,
                'region_bbox': bbox,
                'extraction_time': datetime.utcnow().isoformat()
            })
            
            return regional_ds
            
        except Exception as e:
            self.logger.error(f"Failed to extract regional domain {region_name}: {e}")
            return None
            
    def interpolate_to_high_resolution(self, dataset: xr.Dataset, target_resolution: float) -> xr.Dataset:
        """
        Interpolate dataset to higher spatial resolution.
        
        Args:
            dataset: Input dataset
            target_resolution: Target resolution in degrees
            
        Returns:
            High-resolution dataset
        """
        try:
            self.logger.info(f"Interpolating to {target_resolution}° resolution")
            
            # Get current grid
            old_lats = dataset.latitude.values
            old_lons = dataset.longitude.values
            
            # Create new high-resolution grid
            lat_min, lat_max = old_lats.min(), old_lats.max()
            lon_min, lon_max = old_lons.min(), old_lons.max()
            
            new_lats = np.arange(lat_max, lat_min - target_resolution, -target_resolution)
            new_lons = np.arange(lon_min, lon_max + target_resolution, target_resolution)
            
            self.logger.info(f"New grid: {len(new_lats)} x {len(new_lons)} points")
            
            # Interpolate each variable and time step
            interpolated_vars = {}
            
            for var_name in dataset.data_vars:
                self.logger.debug(f"Interpolating variable: {var_name}")
                
                var_data = dataset[var_name]
                
                # Create interpolated array
                interpolated_data = np.zeros((len(var_data.time), len(new_lats), len(new_lons)))
                
                for t, time_step in enumerate(var_data.time):
                    # Get data for this time step
                    time_data = var_data.sel(time=time_step).values
                    
                    # Create interpolator
                    interpolator = RegularGridInterpolator(
                        (old_lats, old_lons),
                        time_data,
                        method='linear',
                        bounds_error=False,
                        fill_value=np.nan
                    )
                    
                    # Create coordinate grids
                    lat_grid, lon_grid = np.meshgrid(new_lats, new_lons, indexing='ij')
                    
                    # Interpolate
                    interpolated_data[t] = interpolator((lat_grid, lon_grid))
                    
                interpolated_vars[var_name] = (['time', 'latitude', 'longitude'], interpolated_data)
                
            # Create new dataset
            interpolated_ds = xr.Dataset(
                interpolated_vars,
                coords={
                    'time': dataset.time,
                    'latitude': new_lats,
                    'longitude': new_lons
                },
                attrs=dataset.attrs
            )
            
            # Update metadata
            interpolated_ds.attrs.update({
                'interpolated_resolution': target_resolution,
                'interpolation_method': 'linear',
                'interpolation_time': datetime.utcnow().isoformat()
            })
            
            self.logger.info("High-resolution interpolation completed")
            return interpolated_ds
            
        except Exception as e:
            self.logger.error(f"Failed to interpolate to high resolution: {e}")
            raise
            
    def compute_derived_variables(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Compute derived meteorological variables.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with additional derived variables
        """
        try:
            self.logger.info("Computing derived variables")
            
            derived_vars = {}
            
            # Temperature in Celsius
            if '2t' in dataset:
                derived_vars['temp_celsius'] = dataset['2t'] - 273.15
                derived_vars['temp_celsius'].attrs = {
                    'long_name': '2-meter temperature',
                    'units': 'degrees_C'
                }
                
            # Wind speed and direction
            if '10u' in dataset and '10v' in dataset:
                wind_speed = np.sqrt(dataset['10u']**2 + dataset['10v']**2)
                wind_direction = np.arctan2(dataset['10v'], dataset['10u']) * 180 / np.pi
                wind_direction = (wind_direction + 360) % 360  # Convert to 0-360 degrees
                
                derived_vars['wind_speed'] = wind_speed
                derived_vars['wind_speed'].attrs = {
                    'long_name': '10-meter wind speed',
                    'units': 'm/s'
                }
                
                derived_vars['wind_direction'] = wind_direction
                derived_vars['wind_direction'].attrs = {
                    'long_name': '10-meter wind direction',
                    'units': 'degrees'
                }
                
            # Pressure in hPa
            if 'msl' in dataset:
                derived_vars['pressure_hpa'] = dataset['msl'] / 100
                derived_vars['pressure_hpa'].attrs = {
                    'long_name': 'mean sea level pressure',
                    'units': 'hPa'
                }
                
            # Heat index (simplified approximation)
            if 'temp_celsius' in derived_vars:
                temp_c = derived_vars['temp_celsius']
                # Simplified heat index for temperatures > 26.7°C (80°F)
                heat_index = xr.where(
                    temp_c > 26.7,
                    temp_c + 0.1 * (temp_c - 26.7),  # Simplified formula
                    temp_c
                )
                derived_vars['heat_index'] = heat_index
                derived_vars['heat_index'].attrs = {
                    'long_name': 'heat index (simplified)',
                    'units': 'degrees_C'
                }
                
            # Add derived variables to dataset
            for var_name, var_data in derived_vars.items():
                dataset[var_name] = var_data
                
            self.logger.info(f"Computed {len(derived_vars)} derived variables")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Failed to compute derived variables: {e}")
            return dataset
            
    def detect_extreme_weather_events(self, dataset: xr.Dataset, region_name: str) -> Dict:
        """
        Detect extreme weather events in the regional forecast.
        
        Args:
            dataset: Regional forecast dataset
            region_name: Name of the region
            
        Returns:
            Dictionary with detected events
        """
        try:
            self.logger.info(f"Detecting extreme weather events for {region_name}")
            
            events = {
                'region': region_name,
                'analysis_time': datetime.utcnow().isoformat(),
                'heat_waves': [],
                'high_winds': [],
                'extreme_temperatures': []
            }
            
            # Heat wave detection
            if 'temp_celsius' in dataset:
                heat_threshold = self.config['extreme_weather']['heat_wave']['temperature_threshold_percentile']
                
                # Calculate 95th percentile as threshold (simplified)
                temp_data = dataset['temp_celsius']
                temp_percentile_95 = temp_data.quantile(0.95, dim=['latitude', 'longitude'])
                
                for t, time_step in enumerate(temp_data.time):
                    temp_slice = temp_data.sel(time=time_step)
                    extreme_temp_mask = temp_slice > temp_percentile_95.sel(time=time_step)
                    
                    if extreme_temp_mask.sum() > 0:
                        max_temp = temp_slice.max().values
                        mean_temp = temp_slice.where(extreme_temp_mask).mean().values
                        
                        if max_temp > 35:  # Above 35°C
                            events['heat_waves'].append({
                                'time': str(time_step.values),
                                'max_temperature': float(max_temp),
                                'mean_extreme_temperature': float(mean_temp),
                                'affected_area_percent': float(extreme_temp_mask.sum() / extreme_temp_mask.size * 100)
                            })
                            
            # High wind detection
            if 'wind_speed' in dataset:
                wind_threshold = self.config['extreme_weather']['wind']['high_wind_threshold_ms']
                
                wind_data = dataset['wind_speed']
                
                for t, time_step in enumerate(wind_data.time):
                    wind_slice = wind_data.sel(time=time_step)
                    high_wind_mask = wind_slice > wind_threshold
                    
                    if high_wind_mask.sum() > 0:
                        max_wind = wind_slice.max().values
                        mean_wind = wind_slice.where(high_wind_mask).mean().values
                        
                        events['high_winds'].append({
                            'time': str(time_step.values),
                            'max_wind_speed': float(max_wind),
                            'mean_high_wind_speed': float(mean_wind),
                            'affected_area_percent': float(high_wind_mask.sum() / high_wind_mask.size * 100)
                        })
                        
            # Extreme temperature detection (both hot and cold)
            if 'temp_celsius' in dataset:
                temp_data = dataset['temp_celsius']
                
                for t, time_step in enumerate(temp_data.time):
                    temp_slice = temp_data.sel(time=time_step)
                    max_temp = temp_slice.max().values
                    min_temp = temp_slice.min().values
                    
                    if max_temp > 40 or min_temp < -20:  # Extreme thresholds
                        events['extreme_temperatures'].append({
                            'time': str(time_step.values),
                            'max_temperature': float(max_temp),
                            'min_temperature': float(min_temp),
                            'temperature_range': float(max_temp - min_temp)
                        })
                        
            # Summarize events
            total_events = (len(events['heat_waves']) + 
                          len(events['high_winds']) + 
                          len(events['extreme_temperatures']))
            
            events['summary'] = {
                'total_events': total_events,
                'heat_wave_events': len(events['heat_waves']),
                'high_wind_events': len(events['high_winds']),
                'extreme_temperature_events': len(events['extreme_temperatures'])
            }
            
            self.logger.info(f"Detected {total_events} extreme weather events")
            return events
            
        except Exception as e:
            self.logger.error(f"Failed to detect extreme weather events: {e}")
            return {}
            
    def process_region(self, region_name: str, forecast_file: Path) -> Optional[Dict]:
        """
        Process a single regional domain.
        
        Args:
            region_name: Name of region to process
            forecast_file: Path to global forecast file
            
        Returns:
            Processing results
        """
        try:
            self.logger.info(f"Processing region: {region_name}")
            
            # Load global forecast
            global_ds = self.load_forecast_data(forecast_file)
            if global_ds is None:
                return None
                
            # Extract regional domain
            regional_ds = self.extract_regional_domain(global_ds, region_name)
            if regional_ds is None:
                return None
                
            # Get target resolution
            target_resolution = self.regions[region_name]['resolution']
            
            # Interpolate to high resolution if needed
            current_resolution = abs(regional_ds.latitude.values[1] - regional_ds.latitude.values[0])
            if target_resolution < current_resolution:
                regional_ds = self.interpolate_to_high_resolution(regional_ds, target_resolution)
                
            # Compute derived variables
            regional_ds = self.compute_derived_variables(regional_ds)
            
            # Detect extreme weather events
            events = self.detect_extreme_weather_events(regional_ds, region_name)
            
            # Save regional forecast
            output_dir = Path(self.config['paths']['forecasts']) / "regional"
            output_dir.mkdir(exist_ok=True)
            
            base_time = regional_ds.attrs.get('base_time', datetime.utcnow().strftime('%Y%m%d_%H%M'))
            output_file = output_dir / f"regional_forecast_{region_name}_{base_time}.nc"
            
            regional_ds.to_netcdf(output_file)
            
            # Save events summary
            events_file = output_dir / f"extreme_events_{region_name}_{base_time}.yaml"
            with open(events_file, 'w') as f:
                yaml.dump(events, f)
                
            results = {
                'region_name': region_name,
                'output_file': str(output_file),
                'events_file': str(events_file),
                'grid_size': f"{len(regional_ds.latitude)} x {len(regional_ds.longitude)}",
                'resolution': target_resolution,
                'extreme_events': events['summary'],
                'variables': list(regional_ds.data_vars.keys())
            }
            
            self.logger.info(f"Regional processing completed for {region_name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to process region {region_name}: {e}")
            return None
            
    def process_all_regions(self, forecast_file: Path = None) -> Dict:
        """
        Process all configured regional domains.
        
        Args:
            forecast_file: Path to global forecast file (optional)
            
        Returns:
            Results for all regions
        """
        try:
            if forecast_file is None:
                forecast_file = self.find_latest_forecast()
                if forecast_file is None:
                    return {}
                    
            self.logger.info("Processing all regional domains")
            
            results = {
                'processing_time': datetime.utcnow().isoformat(),
                'forecast_file': str(forecast_file),
                'regions': {}
            }
            
            for region_name in self.regions:
                if region_name == 'global':  # Skip global domain
                    continue
                    
                region_results = self.process_region(region_name, forecast_file)
                if region_results:
                    results['regions'][region_name] = region_results
                    
            self.logger.info(f"Processed {len(results['regions'])} regional domains")
            
            # Save overall results summary
            results_file = Path(self.config['paths']['forecasts']) / "regional_processing_summary.yaml"
            with open(results_file, 'w') as f:
                yaml.dump(results, f)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to process regional domains: {e}")
            return {}


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Regional Processor for Aurora-MERS")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--mode", choices=['latest', 'file'], default='latest',
                       help="Processing mode: latest forecast or specific file")
    parser.add_argument("--file", type=str, help="Path to specific forecast file")
    parser.add_argument("--region", type=str, help="Process specific region only")
    
    args = parser.parse_args()
    
    try:
        processor = RegionalProcessor(args.config)
        
        if args.mode == 'latest':
            forecast_file = processor.find_latest_forecast()
        elif args.mode == 'file':
            if not args.file:
                parser.error("File mode requires --file argument")
            forecast_file = Path(args.file)
            
        if forecast_file is None or not forecast_file.exists():
            print("No forecast file found or specified")
            return 1
            
        if args.region:
            # Process single region
            results = processor.process_region(args.region, forecast_file)
            if results:
                print(f"Regional processing completed for {args.region}")
                print(f"Output: {results['output_file']}")
                return 0
            else:
                print(f"Regional processing failed for {args.region}")
                return 1
        else:
            # Process all regions
            results = processor.process_all_regions(forecast_file)
            if results['regions']:
                print(f"Regional processing completed for {len(results['regions'])} regions")
                for region_name, region_results in results['regions'].items():
                    print(f"  {region_name}: {region_results['grid_size']} @ {region_results['resolution']}°")
                return 0
            else:
                print("Regional processing failed")
                return 1
                
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())