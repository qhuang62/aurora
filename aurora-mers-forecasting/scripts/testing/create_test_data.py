#!/usr/bin/env python3
"""
Create Test Data for Aurora-MERS System

Generates synthetic test data to verify the forecasting pipeline
without requiring actual ECMWF downloads.
"""

import os
import sys
import yaml
import numpy as np
import xarray as xr
import torch
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from aurora import Batch, Metadata

def create_test_batch():
    """Create a realistic test batch for Aurora."""
    
    # Use realistic grid dimensions (smaller for testing)
    nlat, nlon = 181, 360  # 1-degree global grid
    nlevels = 4
    ntimes = 2
    
    # Create coordinate arrays
    lats = torch.linspace(90, -90, nlat)
    lons = torch.linspace(0, 360, nlon + 1)[:-1]  # 0 to 359.x degrees
    levels = (1000, 850, 500, 250)
    times = (datetime(2024, 1, 15, 6, 0), datetime(2024, 1, 15, 12, 0))
    
    # Generate realistic surface variables
    surf_vars = {}
    
    # Temperature (2m) - realistic global temperature pattern
    lat_grid = lats.unsqueeze(1).expand(nlat, nlon)
    temp_base = 288 - 0.6 * torch.abs(lat_grid) + 10 * torch.sin(lat_grid * np.pi / 180)
    temp_noise = torch.randn(ntimes, nlat, nlon) * 5
    surf_vars['2t'] = (temp_base.unsqueeze(0).expand(1, ntimes, nlat, nlon) + 
                       temp_noise.unsqueeze(0)).float()
    
    # Wind components (10m) - realistic wind patterns
    u_base = 5 * torch.sin(lat_grid * 2 * np.pi / 180)
    v_base = torch.randn(nlat, nlon) * 3
    u_noise = torch.randn(ntimes, nlat, nlon) * 2
    v_noise = torch.randn(ntimes, nlat, nlon) * 2
    
    surf_vars['10u'] = (u_base.unsqueeze(0).expand(1, ntimes, nlat, nlon) + 
                        u_noise.unsqueeze(0)).float()
    surf_vars['10v'] = (v_base.unsqueeze(0).expand(1, ntimes, nlat, nlon) + 
                        v_noise.unsqueeze(0)).float()
    
    # Mean sea level pressure
    pressure_base = 101325 + 2000 * torch.sin(lat_grid * np.pi / 180)
    pressure_noise = torch.randn(ntimes, nlat, nlon) * 500
    surf_vars['msl'] = (pressure_base.unsqueeze(0).expand(1, ntimes, nlat, nlon) + 
                        pressure_noise.unsqueeze(0)).float()
    
    # Generate static variables
    static_vars = {}
    
    # Land-sea mask (simplified)
    lsm = torch.where(torch.abs(lons.unsqueeze(0) - 180) < 90, 1.0, 0.0)
    lsm = lsm.expand(nlat, nlon).float()
    static_vars['lsm'] = lsm
    
    # Surface geopotential (simplified topography)
    topo = 1000 * torch.abs(torch.sin(lat_grid * 2 * np.pi / 180)) * 9.81
    static_vars['z'] = topo.float()
    
    # Soil type (simplified)
    static_vars['slt'] = torch.ones(nlat, nlon).float()
    
    # Generate atmospheric variables
    atmos_vars = {}
    
    for var_name in ['t', 'u', 'v', 'q', 'z']:
        data = torch.randn(1, ntimes, nlevels, nlat, nlon)
        
        if var_name == 't':  # Temperature
            # Decrease with height
            for i, level in enumerate(levels):
                temp_decrease = (1000 - level) * 0.0065 / 100  # Lapse rate
                data[:, :, i, :, :] = temp_base.unsqueeze(0).expand(1, ntimes, nlat, nlon) - temp_decrease
        elif var_name == 'z':  # Geopotential
            # Increase with height
            for i, level in enumerate(levels):
                data[:, :, i, :, :] = (1000 - level) * 10 * 9.81
        elif var_name == 'q':  # Specific humidity
            # Decrease with height, higher in tropics
            for i, level in enumerate(levels):
                humid_factor = level / 1000.0  # More humidity at lower levels
                tropical_factor = torch.exp(-torch.abs(lat_grid) / 30)
                data[:, :, i, :, :] = 0.01 * humid_factor * tropical_factor.unsqueeze(0).expand(1, ntimes, nlat, nlon)
        
        atmos_vars[var_name] = data.float()
    
    # Create metadata - Aurora expects time to be a tuple with the LAST time step
    metadata = Metadata(
        lat=lats.float(),
        lon=lons.float(),
        time=(times[1],),  # Only the last time step for metadata
        atmos_levels=levels
    )
    
    # Create batch
    batch = Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=metadata
    )
    
    return batch

def main():
    """Generate test data files."""
    
    # Load configuration
    config_path = Path(__file__).parent.parent.parent / "config" / "system_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Creating synthetic test data for Aurora-MERS system...")
    
    # Create test batch
    batch = create_test_batch()
    
    # Save to processed data directory
    processed_dir = Path(config['paths']['processed_data'])
    processed_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M')
    output_file = processed_dir / f"aurora_batch_{timestamp}.nc"
    
    print(f"Saving test batch to: {output_file}")
    batch.to_netcdf(output_file)
    
    print("Test data created successfully!")
    print(f"Surface variables: {list(batch.surf_vars.keys())}")
    print(f"Static variables: {list(batch.static_vars.keys())}")
    print(f"Atmospheric variables: {list(batch.atmos_vars.keys())}")
    print(f"Grid size: {batch.spatial_shape}")
    print(f"Time steps: {len(batch.metadata.time)}")
    
    return str(output_file)

if __name__ == "__main__":
    main()