#!/usr/bin/env python3
"""
Aurora Weather Forecaster for MERS-Aurora Forecasting System

Main forecasting engine that generates weather predictions using
the Aurora foundation model with ECMWF MERS reanalysis data.
"""

import os
import sys
import yaml
import logging
import argparse
import torch
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time
import psutil
import gc

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import Aurora components
from aurora import Aurora, AuroraSmallPretrained, Batch, Metadata, rollout

class AuroraForecaster:
    """
    Aurora Weather Forecasting System.
    
    Generates automated weather forecasts using the Aurora foundation model
    with configurable forecast horizons and regional focusing capabilities.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the Aurora forecaster.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "system_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_logging()
        self.setup_device()
        self.model = None
        self.forecast_metadata = {}
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = Path(self.config['paths']['logs'])
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(log_dir / 'aurora_forecaster.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_device(self):
        """Set up compute device (GPU/CPU)."""
        device_config = self.config['aurora']['device']
        
        if device_config == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU for computation")
            
    def load_model(self) -> bool:
        """
        Load the Aurora model and checkpoint.
        
        Returns:
            True if successful
        """
        try:
            model_name = self.config['aurora']['model_name']
            checkpoint = self.config['aurora']['checkpoint']
            checkpoint_file = self.config['aurora']['checkpoint_file']
            
            self.logger.info(f"Loading Aurora model: {model_name}")
            
            # Load appropriate model based on configuration
            if model_name == "AuroraSmallPretrained":
                self.model = AuroraSmallPretrained()
            else:
                self.model = Aurora()
                
            # Load checkpoint
            self.logger.info(f"Loading checkpoint: {checkpoint}/{checkpoint_file}")
            self.model.load_checkpoint(checkpoint, checkpoint_file)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Move to device
            self.model = self.model.to(self.device)
            
            self.logger.info("Aurora model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load Aurora model: {e}")
            return False
            
    def find_latest_processed_data(self) -> Optional[Path]:
        """
        Find the latest processed Aurora batch file.
        
        Returns:
            Path to latest batch file
        """
        processed_dir = Path(self.config['paths']['processed_data'])
        batch_files = list(processed_dir.glob("aurora_batch_*.nc"))
        
        if not batch_files:
            self.logger.error("No processed Aurora batch files found")
            return None
            
        # Sort by modification time and get the latest
        latest_file = max(batch_files, key=lambda x: x.stat().st_mtime)
        self.logger.info(f"Using latest batch file: {latest_file}")
        
        return latest_file
        
    def load_input_batch(self, batch_file: Path) -> Optional[Batch]:
        """
        Load Aurora batch from file.
        
        Args:
            batch_file: Path to batch file
            
        Returns:
            Loaded batch or None if failed
        """
        try:
            self.logger.info(f"Loading input batch from {batch_file}")
            batch = Batch.from_netcdf(batch_file)
            
            # Move to appropriate device and set precision
            batch = batch.to(self.device)
            
            precision = self.config['aurora']['precision']
            if precision == 'float16':
                batch = batch.type(torch.float16)
            elif precision == 'float32':
                batch = batch.type(torch.float32)
                
            # Crop to patch size if needed
            patch_size = self.config['aurora']['patch_size']
            batch = batch.crop(patch_size)
            
            self.logger.info(f"Loaded batch with spatial shape: {batch.spatial_shape}")
            return batch
            
        except Exception as e:
            self.logger.error(f"Failed to load input batch: {e}")
            return None
            
    def generate_forecast(self, input_batch: Batch) -> List[Batch]:
        """
        Generate weather forecast using Aurora model.
        
        Args:
            input_batch: Input data batch
            
        Returns:
            List of forecast batches for each time step
        """
        try:
            forecast_steps = self.config['aurora']['forecast_steps']
            memory_optimization = self.config['aurora']['memory_optimization']
            
            self.logger.info(f"Starting forecast generation for {forecast_steps} steps")
            start_time = time.time()
            
            # Clear GPU cache if using CUDA
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
            with torch.inference_mode():
                if memory_optimization:
                    # Memory-optimized rollout: move predictions to CPU immediately
                    predictions = []
                    step_count = 0
                    
                    for pred in rollout(self.model, input_batch, steps=forecast_steps):
                        step_count += 1
                        
                        # Move to CPU to save GPU memory
                        pred_cpu = pred.to('cpu')
                        predictions.append(pred_cpu)
                        
                        # Log progress
                        if step_count % 10 == 0:
                            elapsed = time.time() - start_time
                            self.logger.info(f"Completed {step_count}/{forecast_steps} steps in {elapsed:.1f}s")
                            
                        # Clear GPU cache periodically
                        if step_count % 5 == 0 and self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                            
                else:
                    # Standard rollout: keep all predictions on device
                    predictions = list(rollout(self.model, input_batch, steps=forecast_steps))
                    
            end_time = time.time()
            duration = end_time - start_time
            
            self.logger.info(f"Forecast generation completed in {duration:.1f}s ({duration/forecast_steps:.2f}s per step)")
            
            # Update forecast metadata
            self.forecast_metadata.update({
                'generation_time': datetime.utcnow(),
                'forecast_steps': forecast_steps,
                'duration_seconds': duration,
                'input_time': input_batch.metadata.time[0],
                'model_device': str(self.device),
                'memory_optimization': memory_optimization
            })
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Failed to generate forecast: {e}")
            if self.device.type == 'cuda':
                self.logger.error(f"GPU memory usage: {torch.cuda.memory_allocated()/1e9:.1f} GB")
            raise
            
    def save_forecast_outputs(self, predictions: List[Batch], base_time: datetime) -> List[Path]:
        """
        Save forecast outputs to files.
        
        Args:
            predictions: List of forecast batches
            base_time: Base time for forecast
            
        Returns:
            List of output file paths
        """
        try:
            output_dir = Path(self.config['paths']['forecasts'])
            output_dir.mkdir(exist_ok=True)
            
            base_time_str = base_time.strftime('%Y%m%d_%H%M')
            output_files = []
            
            # Save individual forecast steps
            output_frequency = self.config['aurora']['output_frequency']
            
            for i, pred in enumerate(predictions):
                # Only save at specified frequency (e.g., every 24 hours)
                if (i + 1) % output_frequency == 0:
                    forecast_hour = (i + 1) * 6  # 6-hour intervals
                    forecast_time = base_time + timedelta(hours=forecast_hour)
                    
                    output_file = output_dir / f"forecast_{base_time_str}_+{forecast_hour:03d}h.nc"
                    
                    # Ensure prediction is on CPU for saving
                    if pred.surf_vars[list(pred.surf_vars.keys())[0]].device.type != 'cpu':
                        pred = pred.to('cpu')
                        
                    pred.to_netcdf(output_file)
                    output_files.append(output_file)
                    
                    self.logger.debug(f"Saved forecast +{forecast_hour}h to {output_file}")
                    
            # Save complete forecast as single file
            complete_file = output_dir / f"complete_forecast_{base_time_str}.nc"
            self.save_complete_forecast(predictions, complete_file, base_time)
            output_files.append(complete_file)
            
            self.logger.info(f"Saved {len(output_files)} forecast output files")
            return output_files
            
        except Exception as e:
            self.logger.error(f"Failed to save forecast outputs: {e}")
            raise
            
    def save_complete_forecast(self, predictions: List[Batch], output_file: Path, base_time: datetime):
        """
        Save complete forecast as a single NetCDF file.
        
        Args:
            predictions: List of forecast batches
            output_file: Output file path
            base_time: Base forecast time
        """
        try:
            # Combine all predictions into a single dataset
            time_coords = []
            forecast_data = {}
            
            for i, pred in enumerate(predictions):
                forecast_hour = (i + 1) * 6
                forecast_time = base_time + timedelta(hours=forecast_hour)
                time_coords.append(forecast_time)
                
                # Ensure prediction is on CPU
                if pred.surf_vars[list(pred.surf_vars.keys())[0]].device.type != 'cpu':
                    pred = pred.to('cpu')
                
                # Extract surface variables
                for var_name, var_data in pred.surf_vars.items():
                    if var_name not in forecast_data:
                        forecast_data[var_name] = []
                    forecast_data[var_name].append(var_data[0, 0].numpy())  # Remove batch and time dims
                    
            # Create xarray dataset
            coords = {
                'time': time_coords,
                'latitude': predictions[0].metadata.lat.numpy(),
                'longitude': predictions[0].metadata.lon.numpy()
            }
            
            data_vars = {}
            for var_name, var_list in forecast_data.items():
                data_vars[var_name] = (['time', 'latitude', 'longitude'], np.stack(var_list))
                
            ds = xr.Dataset(data_vars, coords=coords)
            
            # Add metadata attributes
            ds.attrs.update({
                'title': 'Aurora Weather Forecast',
                'source': 'Aurora Foundation Model',
                'base_time': base_time.isoformat(),
                'forecast_steps': len(predictions),
                'model': self.config['aurora']['model_name'],
                'generation_time': self.forecast_metadata['generation_time'].isoformat()
            })
            
            # Save to file
            ds.to_netcdf(output_file)
            self.logger.info(f"Saved complete forecast to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save complete forecast: {e}")
            raise
            
    def generate_forecast_summary(self, predictions: List[Batch], base_time: datetime) -> Dict:
        """
        Generate summary statistics for the forecast.
        
        Args:
            predictions: List of forecast batches
            base_time: Base forecast time
            
        Returns:
            Dictionary with forecast summary
        """
        try:
            summary = {
                'base_time': base_time.isoformat(),
                'forecast_horizon_hours': len(predictions) * 6,
                'num_forecast_steps': len(predictions),
                'generation_time': self.forecast_metadata['generation_time'].isoformat(),
                'generation_duration_seconds': self.forecast_metadata['duration_seconds'],
                'variables': [],
                'statistics': {}
            }
            
            # Analyze surface variables
            for var_name in predictions[0].surf_vars.keys():
                summary['variables'].append(var_name)
                
                # Compute statistics across all forecast steps
                all_values = []
                for pred in predictions[:10]:  # First 10 steps to avoid memory issues
                    pred_cpu = pred.to('cpu') if pred.surf_vars[var_name].device.type != 'cpu' else pred
                    values = pred_cpu.surf_vars[var_name].numpy().flatten()
                    all_values.extend(values)
                    
                all_values = np.array(all_values)
                
                summary['statistics'][var_name] = {
                    'mean': float(np.mean(all_values)),
                    'std': float(np.std(all_values)),
                    'min': float(np.min(all_values)),
                    'max': float(np.max(all_values))
                }
                
            self.logger.info("Generated forecast summary statistics")
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate forecast summary: {e}")
            return {}
            
    def run_operational_forecast(self) -> bool:
        """
        Run a complete operational forecast cycle.
        
        Returns:
            True if successful
        """
        try:
            self.logger.info("Starting operational forecast run")
            
            # Step 1: Load model
            if not self.load_model():
                return False
                
            # Step 2: Find and load latest input data
            batch_file = self.find_latest_processed_data()
            if batch_file is None:
                return False
                
            input_batch = self.load_input_batch(batch_file)
            if input_batch is None:
                return False
                
            base_time = input_batch.metadata.time[0]
            
            # Step 3: Generate forecast
            predictions = self.generate_forecast(input_batch)
            
            # Step 4: Save outputs
            output_files = self.save_forecast_outputs(predictions, base_time)
            
            # Step 5: Generate summary
            summary = self.generate_forecast_summary(predictions, base_time)
            
            # Save summary
            summary_file = Path(self.config['paths']['forecasts']) / f"forecast_summary_{base_time.strftime('%Y%m%d_%H%M')}.yaml"
            with open(summary_file, 'w') as f:
                yaml.dump(summary, f)
                
            self.logger.info(f"Operational forecast completed successfully")
            self.logger.info(f"Generated {len(predictions)} forecast steps")
            self.logger.info(f"Output files: {len(output_files)}")
            
            # Clean up GPU memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Operational forecast failed: {e}")
            return False
            
    def run_test_forecast(self, steps: int = 10) -> bool:
        """
        Run a test forecast with reduced steps for validation.
        
        Args:
            steps: Number of forecast steps
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Starting test forecast run with {steps} steps")
            
            # Temporarily override forecast steps
            original_steps = self.config['aurora']['forecast_steps']
            self.config['aurora']['forecast_steps'] = steps
            
            success = self.run_operational_forecast()
            
            # Restore original configuration
            self.config['aurora']['forecast_steps'] = original_steps
            
            return success
            
        except Exception as e:
            self.logger.error(f"Test forecast failed: {e}")
            return False
            
    def get_system_status(self) -> Dict:
        """
        Get current system status and resource usage.
        
        Returns:
            Dictionary with system status
        """
        status = {
            'timestamp': datetime.utcnow().isoformat(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'available_memory_gb': psutil.virtual_memory().available / 1e9,
            'disk_usage_percent': psutil.disk_usage(self.config['paths']['data_root']).percent
        }
        
        if torch.cuda.is_available():
            status.update({
                'gpu_available': True,
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9
            })
        else:
            status['gpu_available'] = False
            
        return status


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Aurora Weather Forecaster")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--mode", choices=['operational', 'test'], default='operational',
                       help="Forecast mode: operational or test")
    parser.add_argument("--test-steps", type=int, default=10,
                       help="Number of steps for test mode")
    parser.add_argument("--status", action='store_true',
                       help="Show system status")
    
    args = parser.parse_args()
    
    try:
        forecaster = AuroraForecaster(args.config)
        
        if args.status:
            status = forecaster.get_system_status()
            print("System Status:")
            for key, value in status.items():
                print(f"  {key}: {value}")
            return 0
            
        if args.mode == 'operational':
            success = forecaster.run_operational_forecast()
        elif args.mode == 'test':
            success = forecaster.run_test_forecast(args.test_steps)
            
        if success:
            print("Forecast generation completed successfully")
            return 0
        else:
            print("Forecast generation failed")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())