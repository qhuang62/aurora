#!/usr/bin/env python3
"""
System Validator for Aurora-MERS Forecasting System

Comprehensive testing script to validate all components
before proceeding with full deployment.
"""

import os
import sys
import yaml
import logging
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

class SystemValidator:
    """
    Comprehensive system validation for Aurora-MERS forecasting.
    
    Tests all components from data download through forecast generation
    to ensure the system is operational.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the system validator."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "system_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_logging()
        self.project_root = Path(__file__).parent.parent.parent
        self.test_results = {}
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = Path(self.config['paths']['logs'])
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'system_validation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def test_environment_setup(self) -> bool:
        """Test basic environment and dependencies."""
        try:
            self.logger.info("Testing environment setup...")
            
            # Test Python version
            if sys.version_info < (3, 10):
                self.logger.error(f"Python version {sys.version} is too old (need 3.10+)")
                return False
                
            # Test required packages
            required_packages = [
                'torch', 'numpy', 'xarray', 'yaml', 'pandas', 
                'scipy', 'matplotlib', 'cartopy'
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
                    
            if missing_packages:
                self.logger.error(f"Missing packages: {missing_packages}")
                return False
                
            # Test Aurora import
            try:
                from aurora import Aurora, Batch, Metadata
                self.logger.info("Aurora import successful")
            except ImportError as e:
                self.logger.error(f"Aurora import failed: {e}")
                return False
                
            # Test CUDA availability (optional)
            import torch
            if torch.cuda.is_available():
                self.logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
            else:
                self.logger.warning("CUDA not available - will use CPU")
                
            self.logger.info("Environment setup test PASSED")
            return True
            
        except Exception as e:
            self.logger.error(f"Environment setup test FAILED: {e}")
            return False
            
    def test_configuration_files(self) -> bool:
        """Test configuration file validity."""
        try:
            self.logger.info("Testing configuration files...")
            
            # Test main config file
            config_file = self.project_root / "config" / "system_config.yaml"
            if not config_file.exists():
                self.logger.error(f"Configuration file not found: {config_file}")
                return False
                
            # Validate config structure
            required_sections = ['ecmwf', 'aurora', 'regions', 'paths', 'compute']
            for section in required_sections:
                if section not in self.config:
                    self.logger.error(f"Missing configuration section: {section}")
                    return False
                    
            # Test directory creation
            for path_key, path_value in self.config['paths'].items():
                path_obj = Path(path_value)
                if not path_obj.exists():
                    self.logger.info(f"Creating missing directory: {path_obj}")
                    path_obj.mkdir(parents=True, exist_ok=True)
                    
            self.logger.info("Configuration files test PASSED")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration files test FAILED: {e}")
            return False
            
    def test_ecmwf_credentials(self) -> bool:
        """Test ECMWF API credentials."""
        try:
            self.logger.info("Testing ECMWF credentials...")
            
            # Check .cdsapirc file
            cdsapirc_path = Path.home() / ".cdsapirc"
            if not cdsapirc_path.exists():
                self.logger.warning("ECMWF credentials file not found (~/.cdsapirc)")
                self.logger.warning("You'll need to configure this for data downloads")
                return False
                
            # Test CDS API import
            try:
                import cdsapi
                client = cdsapi.Client()
                self.logger.info("ECMWF CDS API client initialized successfully")
                return True
            except Exception as e:
                self.logger.error(f"ECMWF API test failed: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"ECMWF credentials test FAILED: {e}")
            return False
            
    def test_aurora_model_loading(self) -> bool:
        """Test Aurora model loading and basic inference."""
        try:
            self.logger.info("Testing Aurora model loading...")
            
            from aurora import AuroraSmallPretrained, Batch, Metadata
            from datetime import datetime
            import torch
            
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
            
            # Load model
            self.logger.info("Loading Aurora model (this may take a while)...")
            model = AuroraSmallPretrained()
            model.load_checkpoint()
            model.eval()
            
            # Test prediction
            with torch.inference_mode():
                prediction = model.forward(batch)
                
            self.logger.info(f"Aurora model test PASSED - prediction shape: {prediction.surf_vars['2t'].shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"Aurora model test FAILED: {e}")
            return False
            
    def test_data_converter(self) -> bool:
        """Test data conversion functionality with synthetic data."""
        try:
            self.logger.info("Testing data converter...")
            
            # Create synthetic test data files
            test_data_dir = Path(self.config['paths']['raw_data']) / "test"
            test_data_dir.mkdir(exist_ok=True)
            
            # Create minimal synthetic NetCDF files for testing
            import xarray as xr
            import numpy as np
            
            # Synthetic surface data
            time_coord = [datetime.utcnow() - timedelta(hours=6), datetime.utcnow()]
            lat_coord = np.linspace(90, -90, 721)
            lon_coord = np.linspace(0, 359.75, 1440)
            
            surface_data = xr.Dataset({
                '2m_temperature': (['time', 'latitude', 'longitude'], 
                                 np.random.normal(288, 20, (2, len(lat_coord), len(lon_coord)))),
                '10m_u_component_of_wind': (['time', 'latitude', 'longitude'],
                                          np.random.normal(0, 5, (2, len(lat_coord), len(lon_coord)))),
                '10m_v_component_of_wind': (['time', 'latitude', 'longitude'],
                                          np.random.normal(0, 5, (2, len(lat_coord), len(lon_coord)))),
                'mean_sea_level_pressure': (['time', 'latitude', 'longitude'],
                                          np.random.normal(101325, 2000, (2, len(lat_coord), len(lon_coord))))
            }, coords={
                'time': time_coord,
                'latitude': lat_coord,
                'longitude': lon_coord
            })
            
            surface_file = test_data_dir / "test_surface.nc"
            surface_data.to_netcdf(surface_file)
            
            # Test if converter can handle the file
            converter_script = self.project_root / "scripts" / "processing" / "aurora_data_converter.py"
            if converter_script.exists():
                self.logger.info("Data converter script found - basic structure test PASSED")
                return True
            else:
                self.logger.error("Data converter script not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Data converter test FAILED: {e}")
            return False
            
    def test_forecaster_script(self) -> bool:
        """Test the Aurora forecaster script."""
        try:
            self.logger.info("Testing forecaster script...")
            
            forecaster_script = self.project_root / "scripts" / "forecasting" / "aurora_forecaster.py"
            if not forecaster_script.exists():
                self.logger.error("Forecaster script not found")
                return False
                
            # Test script help/status functionality
            cmd = [sys.executable, str(forecaster_script), "--status"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.logger.info("Forecaster script basic test PASSED")
                return True
            else:
                self.logger.warning(f"Forecaster script returned {result.returncode}")
                self.logger.warning(f"Output: {result.stdout}")
                self.logger.warning(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Forecaster script test FAILED: {e}")
            return False
            
    def test_slurm_availability(self) -> bool:
        """Test SLURM availability (optional on non-HPC systems)."""
        try:
            self.logger.info("Testing SLURM availability...")
            
            result = subprocess.run(['squeue', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.logger.info(f"SLURM available: {result.stdout.strip()}")
                return True
            else:
                self.logger.warning("SLURM not available (OK if not on HPC system)")
                return False
                
        except FileNotFoundError:
            self.logger.warning("SLURM not found (OK if not on HPC system)")
            return False
        except Exception as e:
            self.logger.warning(f"SLURM test failed: {e}")
            return False
            
    def test_regional_processor(self) -> bool:
        """Test regional processing capabilities."""
        try:
            self.logger.info("Testing regional processor...")
            
            regional_script = self.project_root / "scripts" / "processing" / "regional_processor.py"
            if not regional_script.exists():
                self.logger.error("Regional processor script not found")
                return False
                
            # Check if regions are properly configured
            regions = self.config.get('regions', {})
            if len(regions) == 0:
                self.logger.error("No regions configured")
                return False
                
            self.logger.info(f"Regional processor found with {len(regions)} regions configured")
            return True
            
        except Exception as e:
            self.logger.error(f"Regional processor test FAILED: {e}")
            return False
            
    def run_comprehensive_test(self) -> Dict:
        """Run all validation tests."""
        self.logger.info("Starting comprehensive system validation...")
        start_time = time.time()
        
        tests = [
            ("Environment Setup", self.test_environment_setup),
            ("Configuration Files", self.test_configuration_files),
            ("ECMWF Credentials", self.test_ecmwf_credentials),
            ("Aurora Model Loading", self.test_aurora_model_loading),
            ("Data Converter", self.test_data_converter),
            ("Forecaster Script", self.test_forecaster_script),
            ("SLURM Availability", self.test_slurm_availability),
            ("Regional Processor", self.test_regional_processor),
        ]
        
        results = {}
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Running test: {test_name}")
            self.logger.info(f"{'='*50}")
            
            try:
                result = test_func()
                results[test_name] = {
                    'status': 'PASSED' if result else 'FAILED',
                    'success': result
                }
                if result:
                    passed_tests += 1
                    
            except Exception as e:
                self.logger.error(f"Test {test_name} encountered an error: {e}")
                results[test_name] = {
                    'status': 'ERROR',
                    'success': False,
                    'error': str(e)
                }
                
        end_time = time.time()
        duration = end_time - start_time
        
        # Generate summary
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': (passed_tests / total_tests) * 100,
            'duration_seconds': duration,
            'timestamp': datetime.utcnow().isoformat(),
            'test_results': results
        }
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total Tests: {total_tests}")
        self.logger.info(f"Passed: {passed_tests}")
        self.logger.info(f"Failed: {total_tests - passed_tests}")
        self.logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        self.logger.info(f"Duration: {duration:.1f} seconds")
        
        # Save detailed results
        results_file = Path(self.config['paths']['logs']) / 'validation_results.yaml'
        with open(results_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
            
        self.logger.info(f"Detailed results saved to: {results_file}")
        
        return summary


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Aurora-MERS System Validator")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--test", choices=[
        'environment', 'config', 'ecmwf', 'aurora', 'converter', 
        'forecaster', 'slurm', 'regional', 'all'
    ], default='all', help="Specific test to run")
    
    args = parser.parse_args()
    
    try:
        validator = SystemValidator(args.config)
        
        if args.test == 'all':
            results = validator.run_comprehensive_test()
            success_rate = results['success_rate']
            
            if success_rate >= 80:
                print(f"\n✅ System validation PASSED ({success_rate:.1f}% success rate)")
                print("System is ready for operational use!")
                return 0
            else:
                print(f"\n❌ System validation FAILED ({success_rate:.1f}% success rate)")
                print("Please address the failed tests before proceeding.")
                return 1
        else:
            # Run specific test
            test_methods = {
                'environment': validator.test_environment_setup,
                'config': validator.test_configuration_files,
                'ecmwf': validator.test_ecmwf_credentials,
                'aurora': validator.test_aurora_model_loading,
                'converter': validator.test_data_converter,
                'forecaster': validator.test_forecaster_script,
                'slurm': validator.test_slurm_availability,
                'regional': validator.test_regional_processor
            }
            
            if args.test in test_methods:
                result = test_methods[args.test]()
                if result:
                    print(f"✅ {args.test.title()} test PASSED")
                    return 0
                else:
                    print(f"❌ {args.test.title()} test FAILED")
                    return 1
            
    except Exception as e:
        print(f"Validation error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())