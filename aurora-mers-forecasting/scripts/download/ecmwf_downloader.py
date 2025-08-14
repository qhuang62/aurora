#!/usr/bin/env python3
"""
ECMWF ERA5 Data Downloader for Aurora-ERA5 Forecasting System

This module handles automated downloading of ECMWF ERA5 reanalysis data
for use with the Aurora weather forecasting model.
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import requests
import cdsapi
import xarray as xr
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

class ECMWFDownloader:
    """
    ECMWF Data Downloader for Aurora forecasting system.
    
    Handles automated download of ERA5 (ECMWF Reanalysis v5) data
    every 12 hours for use with Aurora weather prediction model.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the ECMWF downloader.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "system_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_logging()
        self.setup_directories()
        self.setup_cds_client()
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = Path(self.config['paths']['logs'])
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(log_dir / 'ecmwf_downloader.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories for data storage."""
        for path_key in ['raw_data', 'processed_data', 'static_data']:
            Path(self.config['paths'][path_key]).mkdir(parents=True, exist_ok=True)
            
    def setup_cds_client(self):
        """Initialize the CDS API client."""
        try:
            self.cds_client = cdsapi.Client()
            self.logger.info("CDS API client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize CDS client: {e}")
            self.logger.info("Please ensure ~/.cdsapirc is configured with your credentials")
            raise
            
    def get_latest_available_time(self) -> datetime:
        """
        Get the latest available ERA5 data time.
        
        Returns:
            Latest available data time
        """
        # ERA5 data is typically available with 2-3 hour delay
        now = datetime.utcnow()
        
        # Round down to nearest 6-hour interval
        hour = (now.hour // 6) * 6
        latest_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        
        # Subtract 3 hours to account for processing delay
        latest_time -= timedelta(hours=3)
        
        return latest_time
        
    def check_data_exists(self, date: datetime, data_type: str) -> bool:
        """
        Check if data already exists for given date and type.
        
        Args:
            date: Date to check
            data_type: Type of data (surface, atmospheric, static)
            
        Returns:
            True if data exists
        """
        raw_data_dir = Path(self.config['paths']['raw_data'])
        filename = f"{date.strftime('%Y%m%d_%H%M')}_{data_type}.nc"
        file_path = raw_data_dir / filename
        
        return file_path.exists()
        
    def download_surface_variables(self, date: datetime, output_path: Path) -> bool:
        """
        Download surface-level variables.
        
        Args:
            date: Date to download
            output_path: Output file path
            
        Returns:
            True if successful
        """
        try:
            request = {
                'product_type': 'reanalysis',
                'variable': self.config['ecmwf']['surface_variables'],
                'year': str(date.year),
                'month': f"{date.month:02d}",
                'day': f"{date.day:02d}",
                'time': f"{date.hour:02d}:00",
                'grid': self.config['ecmwf']['grid'],
                'area': self.config['ecmwf']['area'],
                'format': self.config['ecmwf']['format'],
            }
            
            self.logger.info(f"Downloading surface variables for {date}")
            self.cds_client.retrieve(
                self.config['ecmwf']['dataset'],
                request,
                str(output_path)
            )
            
            self.logger.info(f"Successfully downloaded surface data to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download surface variables for {date}: {e}")
            return False
            
    def download_atmospheric_variables(self, date: datetime, output_path: Path) -> bool:
        """
        Download atmospheric variables at pressure levels.
        
        Args:
            date: Date to download
            output_path: Output file path
            
        Returns:
            True if successful
        """
        try:
            request = {
                'product_type': 'reanalysis',
                'variable': self.config['ecmwf']['atmospheric_variables'],
                'pressure_level': self.config['ecmwf']['pressure_levels'],
                'year': str(date.year),
                'month': f"{date.month:02d}",
                'day': f"{date.day:02d}",
                'time': f"{date.hour:02d}:00",
                'grid': self.config['ecmwf']['grid'],
                'area': self.config['ecmwf']['area'],
                'format': self.config['ecmwf']['format'],
            }
            
            self.logger.info(f"Downloading atmospheric variables for {date}")
            self.cds_client.retrieve(
                "reanalysis-era5-pressure-levels",  # Different dataset for pressure levels
                request,
                str(output_path)
            )
            
            self.logger.info(f"Successfully downloaded atmospheric data to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download atmospheric variables for {date}: {e}")
            return False
            
    def download_static_variables(self, output_path: Path) -> bool:
        """
        Download static variables (topography, land-sea mask, etc.).
        
        Args:
            output_path: Output file path
            
        Returns:
            True if successful
        """
        try:
            # Use a reference date for static variables
            ref_date = datetime(2023, 1, 1)
            
            request = {
                'product_type': 'reanalysis',
                'variable': self.config['ecmwf']['static_variables'],
                'year': str(ref_date.year),
                'month': f"{ref_date.month:02d}",
                'day': f"{ref_date.day:02d}",
                'time': "00:00",
                'grid': self.config['ecmwf']['grid'],
                'area': self.config['ecmwf']['area'],
                'format': self.config['ecmwf']['format'],
            }
            
            self.logger.info("Downloading static variables")
            self.cds_client.retrieve(
                self.config['ecmwf']['dataset'],
                request,
                str(output_path)
            )
            
            self.logger.info(f"Successfully downloaded static data to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download static variables: {e}")
            return False
            
    def download_hourly_data(self, target_date: datetime) -> Dict[str, bool]:
        """
        Download all required data for a specific date.
        
        Args:
            target_date: Date to download data for
            
        Returns:
            Dictionary with download status for each data type
        """
        raw_data_dir = Path(self.config['paths']['raw_data'])
        date_str = target_date.strftime('%Y%m%d_%H%M')
        
        results = {}
        
        # Download surface variables
        surface_file = raw_data_dir / f"{date_str}_surface.nc"
        if not self.check_data_exists(target_date, 'surface'):
            results['surface'] = self.download_surface_variables(target_date, surface_file)
        else:
            self.logger.info(f"Surface data already exists for {target_date}")
            results['surface'] = True
            
        # Download atmospheric variables  
        atmospheric_file = raw_data_dir / f"{date_str}_atmospheric.nc"
        if not self.check_data_exists(target_date, 'atmospheric'):
            results['atmospheric'] = self.download_atmospheric_variables(target_date, atmospheric_file)
        else:
            self.logger.info(f"Atmospheric data already exists for {target_date}")
            results['atmospheric'] = True
            
        # Download static variables (only once)
        static_file = Path(self.config['paths']['static_data']) / "static.nc"
        if not static_file.exists():
            results['static'] = self.download_static_variables(static_file)
        else:
            self.logger.info("Static data already exists")
            results['static'] = True
            
        return results
        
    def download_historical_data(self, start_date: datetime, end_date: datetime) -> bool:
        """
        Download historical data for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            True if all downloads successful
        """
        current_date = start_date
        all_successful = True
        
        while current_date <= end_date:
            self.logger.info(f"Processing date: {current_date}")
            
            results = self.download_hourly_data(current_date)
            
            if not all(results.values()):
                self.logger.error(f"Failed to download some data for {current_date}")
                all_successful = False
                
            # Move to next 6-hour interval
            current_date += timedelta(hours=6)
            
            # Rate limiting - wait between requests
            time.sleep(2)
            
        return all_successful
        
    def download_latest_data(self) -> bool:
        """
        Download the latest available data.
        
        Returns:
            True if successful
        """
        latest_time = self.get_latest_available_time()
        self.logger.info(f"Downloading latest data for: {latest_time}")
        
        results = self.download_hourly_data(latest_time)
        
        success = all(results.values())
        if success:
            self.logger.info("Successfully downloaded latest data")
        else:
            self.logger.error("Failed to download some latest data")
            
        return success
        
    def cleanup_old_data(self):
        """Remove old data files based on retention policy."""
        retention_days = self.config['ecmwf']['data_retention_days']
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        raw_data_dir = Path(self.config['paths']['raw_data'])
        deleted_count = 0
        
        for file_path in raw_data_dir.glob("*.nc"):
            try:
                # Extract date from filename (YYYYMMDD_HHMM format)
                date_str = file_path.stem.split('_')[0] + file_path.stem.split('_')[1]
                file_date = datetime.strptime(date_str, '%Y%m%d%H%M')
                
                if file_date < cutoff_date:
                    file_path.unlink()
                    deleted_count += 1
                    self.logger.debug(f"Deleted old file: {file_path}")
                    
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Could not parse date from filename {file_path}: {e}")
                
        self.logger.info(f"Cleaned up {deleted_count} old data files")
        
    def validate_downloaded_data(self, file_path: Path) -> bool:
        """
        Validate downloaded NetCDF file.
        
        Args:
            file_path: Path to NetCDF file
            
        Returns:
            True if data is valid
        """
        try:
            with xr.open_dataset(file_path) as ds:
                # Check if required variables exist
                required_vars = set()
                if 'surface' in str(file_path):
                    required_vars = set(self.config['ecmwf']['surface_variables'])
                elif 'atmospheric' in str(file_path):
                    required_vars = set(self.config['ecmwf']['atmospheric_variables'])
                    
                available_vars = set(ds.data_vars.keys())
                missing_vars = required_vars - available_vars
                
                if missing_vars:
                    self.logger.error(f"Missing variables in {file_path}: {missing_vars}")
                    return False
                    
                # Check for data quality
                for var in ds.data_vars:
                    data = ds[var].values
                    if np.isnan(data).all():
                        self.logger.error(f"All NaN values in variable {var} in {file_path}")
                        return False
                        
                self.logger.info(f"Data validation passed for {file_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to validate {file_path}: {e}")
            return False


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="ECMWF Data Downloader for Aurora-ERA5")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--mode", choices=['latest', 'historical'], default='latest',
                       help="Download mode: latest data or historical range")
    parser.add_argument("--start-date", type=str, help="Start date for historical download (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for historical download (YYYY-MM-DD)")
    parser.add_argument("--cleanup", action='store_true', help="Clean up old data files")
    
    args = parser.parse_args()
    
    try:
        downloader = ECMWFDownloader(args.config)
        
        if args.cleanup:
            downloader.cleanup_old_data()
            
        if args.mode == 'latest':
            success = downloader.download_latest_data()
        elif args.mode == 'historical':
            if not args.start_date or not args.end_date:
                parser.error("Historical mode requires --start-date and --end-date")
                
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
            success = downloader.download_historical_data(start_date, end_date)
            
        if success:
            print("Download completed successfully")
            sys.exit(0)
        else:
            print("Download failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()