"""
Aurora Sensitivity Threshold Mapping

Systematic exploration of Aurora's sensitivity thresholds across multiple dimensions:
1. Variable types (temperature, winds, pressure, humidity, geopotential)
2. Perturbation magnitudes (realistic to extreme)
3. Spatial scales (local to global)
4. Temporal persistence (single time step to multi-day)
5. Atmospheric levels (surface to upper troposphere)

This script provides comprehensive mapping of Aurora's sensitivity landscape
to inform ensemble forecasting and Weather Jiu-Jitsu applications.
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import pickle
import time
from itertools import product
from datetime import datetime
import warnings

from aurora import Aurora, Tracker, rollout, Batch
from perturbation_utils import (
    add_gaussian_perturbation,
    compute_track_deviation,
    analyze_forecast_spread,
    evaluate_perturbation_sensitivity
)


class AuroraSensitivityMapper:
    """
    Comprehensive sensitivity threshold mapping for Aurora model.
    
    Maps Aurora's response across multiple perturbation dimensions to identify
    optimal parameter ranges for ensemble forecasting and intervention studies.
    """
    
    def __init__(
        self,
        model: Aurora,
        device: str = "cuda",
        save_intermediate: bool = True,
        output_directory: str = "sensitivity_mapping_results"
    ):
        """
        Initialize sensitivity mapping system.
        
        Args:
            model: Aurora model instance
            device: Device for model execution
            save_intermediate: Whether to save intermediate results
            output_directory: Directory for saving results
        """
        self.model = model
        self.device = device
        self.save_intermediate = save_intermediate
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment configuration
        self.baseline_batch = None
        self.baseline_forecast = None
        self.baseline_track = None
        
        # Results storage
        self.sensitivity_map = {}
        self.threshold_analysis = {}
        self.parameter_optimization = {}
        
    def load_baseline_case(
        self,
        case_name: str = "nanmadol_2022_09_17",
        download_path: Optional[str] = None
    ) -> None:
        """
        Load baseline case for sensitivity testing.
        
        Args:
            case_name: Name of the test case
            download_path: Path to downloaded data
        """
        
        print(f"Loading baseline case: {case_name}")
        
        if download_path is None:
            download_path = Path("~/downloads").expanduser()
        else:
            download_path = Path(download_path)
        
        # Load Typhoon Nanmadol case (reuse existing data preparation)
        day = "2022-09-17"
        
        import xarray as xr
        
        static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
        surf_vars_ds = xr.open_dataset(download_path / f"{day}-surface-level.nc", engine="netcdf4")
        atmos_vars_ds = xr.open_dataset(download_path / f"{day}-atmospheric.nc", engine="netcdf4")
        
        def _prepare(x: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(x[[1, 2]][None][..., ::-1, :].copy())
        
        # Create baseline batch
        self.baseline_batch = Batch(
            surf_vars={
                "2t": _prepare(surf_vars_ds["2m_temperature"].values),
                "10u": _prepare(surf_vars_ds["10m_u_component_of_wind"].values),
                "10v": _prepare(surf_vars_ds["10m_v_component_of_wind"].values),
                "msl": _prepare(surf_vars_ds["mean_sea_level_pressure"].values),
            },
            static_vars={
                "z": torch.from_numpy(static_vars_ds["z"].values[0]),
                "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
                "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
            },
            atmos_vars={
                "t": _prepare(atmos_vars_ds["temperature"].values),
                "u": _prepare(atmos_vars_ds["u_component_of_wind"].values),
                "v": _prepare(atmos_vars_ds["v_component_of_wind"].values),
                "q": _prepare(atmos_vars_ds["specific_humidity"].values),
                "z": _prepare(atmos_vars_ds["geopotential"].values),
            },
            metadata=Metadata(
                lat=torch.from_numpy(surf_vars_ds.latitude.values[::-1].copy()),
                lon=torch.from_numpy(surf_vars_ds.longitude.values),
                time=(surf_vars_ds.time.values.astype("datetime64[s]").tolist()[2],),
                atmos_levels=tuple(int(level) for level in atmos_vars_ds.level.values),
            ),
        )
        
        print("Baseline batch loaded successfully")
        
        # Run baseline forecast
        print("Running baseline forecast...")
        self._run_baseline_forecast()
        
    def _run_baseline_forecast(self, forecast_steps: int = 40) -> None:
        """Run baseline forecast for comparison."""
        
        # Initialize tracker
        tracker = Tracker(
            init_lat=27.50,
            init_lon=132,
            init_time=datetime(2022, 9, 17, 12, 0)
        )
        
        # Run forecast
        predictions = []
        with torch.inference_mode():
            for pred in rollout(self.model, self.baseline_batch, steps=forecast_steps):
                pred = pred.to("cpu")
                predictions.append(pred)
                tracker.step(pred)
        
        self.baseline_forecast = predictions
        self.baseline_track = tracker.results()
        
        print(f"Baseline forecast completed: {len(predictions)} time steps")
        
    def map_variable_sensitivity(
        self,
        variables: Optional[List[str]] = None,
        magnitude_ranges: Optional[Dict[str, List[float]]] = None,
        forecast_steps: int = 40
    ) -> Dict:
        """
        Map sensitivity across different variables and magnitude ranges.
        
        Args:
            variables: Variables to test (default: comprehensive set)
            magnitude_ranges: Magnitude ranges for each variable type
            forecast_steps: Number of forecast steps
            
        Returns:
            Dictionary with sensitivity mapping results
        """
        
        if variables is None:
            variables = ['temperature', 'winds', 'pressure', 'humidity', 'geopotential']
        
        if magnitude_ranges is None:
            magnitude_ranges = {
                'temperature': [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0],  # K
                'winds': [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],        # m/s
                'pressure': [20, 50, 100, 200, 500, 1000, 2000, 5000],        # Pa
                'humidity': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],     # kg/kg
                'geopotential': [10, 25, 50, 100, 200, 500, 1000]             # m²/s²
            }
        
        print(f"Mapping variable sensitivity for {len(variables)} variables...")
        
        sensitivity_results = {}
        
        for variable in variables:
            print(f"  Testing {variable} perturbations...")
            
            variable_results = []
            magnitudes = magnitude_ranges.get(variable, [1.0])
            
            for magnitude in magnitudes:
                print(f"    Magnitude: {magnitude}")
                
                # Run perturbation test
                result = self._test_single_perturbation(
                    variable, magnitude, forecast_steps
                )
                
                variable_results.append(result)
                
                # Save intermediate results
                if self.save_intermediate:
                    self._save_intermediate_result(variable, magnitude, result)
            
            sensitivity_results[variable] = variable_results
            
            # Analyze sensitivity for this variable
            sensitivity_analysis = evaluate_perturbation_sensitivity(
                variable_results, variable
            )
            
            print(f"    {variable} sensitivity: {sensitivity_analysis['sensitivity_level']}")
            print(f"    Threshold magnitude: {sensitivity_analysis['threshold_magnitude']}")
        
        self.sensitivity_map['variable_sensitivity'] = sensitivity_results
        return sensitivity_results
    
    def map_spatial_scale_sensitivity(
        self,
        base_variable: str = 'winds',
        base_magnitude: float = 2.0,
        spatial_scales: Optional[List[Tuple[int, int]]] = None,
        forecast_steps: int = 40
    ) -> Dict:
        """
        Map sensitivity to different spatial scales of perturbations.
        
        Args:
            base_variable: Variable type for spatial testing
            base_magnitude: Base perturbation magnitude
            spatial_scales: List of (lat_scale, lon_scale) tuples in grid points
            forecast_steps: Number of forecast steps
            
        Returns:
            Dictionary with spatial scale sensitivity results
        """
        
        if spatial_scales is None:
            # Grid scales from local to global
            spatial_scales = [
                (1, 1),     # Single grid point
                (3, 3),     # Small local region
                (5, 5),     # Local region
                (10, 10),   # Regional scale
                (20, 20),   # Large regional scale
                (50, 50),   # Continental scale
                (-1, -1)    # Global (all grid points)
            ]
        
        print(f"Mapping spatial scale sensitivity for {base_variable}...")
        
        spatial_results = []
        
        for lat_scale, lon_scale in spatial_scales:
            scale_name = f"{lat_scale}x{lon_scale}" if lat_scale > 0 else "global"
            print(f"  Testing spatial scale: {scale_name}")
            
            # Create spatially-constrained perturbation
            result = self._test_spatial_perturbation(
                base_variable, base_magnitude, lat_scale, lon_scale, forecast_steps
            )
            
            result['spatial_scale'] = (lat_scale, lon_scale)
            result['scale_name'] = scale_name
            spatial_results.append(result)
        
        self.sensitivity_map['spatial_sensitivity'] = spatial_results
        return spatial_results
    
    def map_atmospheric_level_sensitivity(
        self,
        base_variable: str = 'winds',
        base_magnitude: float = 2.0,
        target_levels: Optional[List[Union[str, int]]] = None,
        forecast_steps: int = 40
    ) -> Dict:
        """
        Map sensitivity to perturbations at different atmospheric levels.
        
        Args:
            base_variable: Variable type for level testing
            base_magnitude: Base perturbation magnitude
            target_levels: Atmospheric levels to test
            forecast_steps: Number of forecast steps
            
        Returns:
            Dictionary with level sensitivity results
        """
        
        if target_levels is None:
            target_levels = ['surface', 100, 250, 500, 850]  # hPa levels + surface
        
        print(f"Mapping atmospheric level sensitivity for {base_variable}...")
        
        level_results = []
        
        for level in target_levels:
            print(f"  Testing level: {level}")
            
            # Test perturbation at specific level
            result = self._test_level_perturbation(
                base_variable, base_magnitude, level, forecast_steps
            )
            
            result['target_level'] = level
            level_results.append(result)
        
        self.sensitivity_map['level_sensitivity'] = level_results
        return level_results
    
    def map_temporal_persistence_sensitivity(
        self,
        base_variable: str = 'winds',
        base_magnitude: float = 2.0,
        persistence_durations: Optional[List[int]] = None,
        forecast_steps: int = 40
    ) -> Dict:
        """
        Map sensitivity to temporal persistence of perturbations.
        
        Args:
            base_variable: Variable type for temporal testing
            base_magnitude: Base perturbation magnitude
            persistence_durations: Duration of perturbation persistence (time steps)
            forecast_steps: Number of forecast steps
            
        Returns:
            Dictionary with temporal sensitivity results
        """
        
        if persistence_durations is None:
            persistence_durations = [1, 2, 4, 8, 16, -1]  # -1 = permanent
        
        print(f"Mapping temporal persistence sensitivity for {base_variable}...")
        
        temporal_results = []
        
        for duration in persistence_durations:
            duration_name = f"{duration}_steps" if duration > 0 else "permanent"
            print(f"  Testing duration: {duration_name}")
            
            # Test temporally-persistent perturbation
            result = self._test_temporal_perturbation(
                base_variable, base_magnitude, duration, forecast_steps
            )
            
            result['persistence_duration'] = duration
            result['duration_name'] = duration_name
            temporal_results.append(result)
        
        self.sensitivity_map['temporal_sensitivity'] = temporal_results
        return temporal_results
    
    def _test_single_perturbation(
        self,
        variable_type: str,
        magnitude: float,
        forecast_steps: int
    ) -> Dict:
        """Test single perturbation configuration."""
        
        # Create perturbed batch
        perturbed_batch = add_gaussian_perturbation(
            self.baseline_batch,
            perturbation_type=variable_type,
            magnitude=magnitude
        )
        
        # Run perturbed forecast
        tracker = Tracker(
            init_lat=27.50, init_lon=132, init_time=datetime(2022, 9, 17, 12, 0)
        )
        
        predictions = []
        with torch.inference_mode():
            for pred in rollout(self.model, perturbed_batch, steps=forecast_steps):
                pred = pred.to("cpu")
                predictions.append(pred)
                tracker.step(pred)
        
        track_perturbed = tracker.results()
        
        # Compute deviations
        track_deviation = compute_track_deviation(self.baseline_track, track_perturbed)
        forecast_rmse = analyze_forecast_spread(self.baseline_forecast, predictions)
        
        return {
            'variable_type': variable_type,
            'magnitude': magnitude,
            'max_track_deviation_km': track_deviation['max_distance_km'],
            'final_track_deviation_km': track_deviation['final_distance_km'],
            'mean_track_deviation_km': track_deviation['mean_distance_km'],
            'track_deviation_time_series': track_deviation['distance_time_series'],
            'forecast_rmse': forecast_rmse,
            'track_baseline': self.baseline_track,
            'track_perturbed': track_perturbed
        }
    
    def _test_spatial_perturbation(
        self,
        variable_type: str,
        magnitude: float,
        lat_scale: int,
        lon_scale: int,
        forecast_steps: int
    ) -> Dict:
        """Test spatially-constrained perturbation."""
        
        # Create spatially-limited perturbation
        perturbed_batch = self._create_spatial_perturbation(
            self.baseline_batch, variable_type, magnitude, lat_scale, lon_scale
        )
        
        # Run forecast and analyze (same as single perturbation)
        result = self._run_perturbation_forecast(perturbed_batch, forecast_steps)
        result.update({
            'variable_type': variable_type,
            'magnitude': magnitude,
            'lat_scale': lat_scale,
            'lon_scale': lon_scale
        })
        
        return result
    
    def _test_level_perturbation(
        self,
        variable_type: str,
        magnitude: float,
        target_level: Union[str, int],
        forecast_steps: int
    ) -> Dict:
        """Test level-specific perturbation."""
        
        # Create level-specific perturbation
        perturbed_batch = self._create_level_perturbation(
            self.baseline_batch, variable_type, magnitude, target_level
        )
        
        # Run forecast and analyze
        result = self._run_perturbation_forecast(perturbed_batch, forecast_steps)
        result.update({
            'variable_type': variable_type,
            'magnitude': magnitude,
            'target_level': target_level
        })
        
        return result
    
    def _test_temporal_perturbation(
        self,
        variable_type: str,
        magnitude: float,
        duration: int,
        forecast_steps: int
    ) -> Dict:
        """Test temporally-persistent perturbation."""
        
        # For temporal persistence, we need to modify the rollout process
        # This is a simplified implementation - full implementation would require
        # modifying the Aurora rollout to maintain perturbations
        
        # Start with regular perturbation
        perturbed_batch = add_gaussian_perturbation(
            self.baseline_batch,
            perturbation_type=variable_type,
            magnitude=magnitude
        )
        
        # Run forecast (temporal persistence would require more complex implementation)
        result = self._run_perturbation_forecast(perturbed_batch, forecast_steps)
        result.update({
            'variable_type': variable_type,
            'magnitude': magnitude,
            'persistence_duration': duration
        })
        
        return result
    
    def _create_spatial_perturbation(
        self,
        batch: Batch,
        variable_type: str,
        magnitude: float,
        lat_scale: int,
        lon_scale: int
    ) -> Batch:
        """Create spatially-constrained perturbation."""
        
        import copy
        perturbed_batch = copy.deepcopy(batch)
        
        # Get spatial dimensions
        if variable_type in ['temperature', 'winds']:
            sample_field = batch.surf_vars['2t']
            H, W = sample_field.shape[-2:]
        else:
            sample_field = batch.surf_vars['msl']
            H, W = sample_field.shape[-2:]
        
        # Create spatial mask
        if lat_scale < 0 or lon_scale < 0:
            # Global perturbation
            spatial_mask = torch.ones(H, W)
        else:
            # Localized perturbation (center of domain)
            spatial_mask = torch.zeros(H, W)
            center_lat = H // 2
            center_lon = W // 2
            
            lat_start = max(0, center_lat - lat_scale // 2)
            lat_end = min(H, center_lat + lat_scale // 2)
            lon_start = max(0, center_lon - lon_scale // 2)
            lon_end = min(W, center_lon + lon_scale // 2)
            
            spatial_mask[lat_start:lat_end, lon_start:lon_end] = 1.0
        
        # Apply spatially-masked perturbation
        if variable_type == 'temperature':
            noise = torch.normal(0, magnitude, sample_field.shape)
            mask_expanded = spatial_mask[None, None, :, :]
            perturbed_batch.surf_vars['2t'] += noise * mask_expanded
            
            if 't' in perturbed_batch.atmos_vars:
                atmos_noise = torch.normal(0, magnitude, perturbed_batch.atmos_vars['t'].shape)
                atmos_mask = spatial_mask[None, None, None, :, :]
                perturbed_batch.atmos_vars['t'] += atmos_noise * atmos_mask
        
        elif variable_type == 'winds':
            # Apply to all wind components
            for wind_var in ['10u', '10v']:
                if wind_var in perturbed_batch.surf_vars:
                    noise = torch.normal(0, magnitude, perturbed_batch.surf_vars[wind_var].shape)
                    mask_expanded = spatial_mask[None, None, :, :]
                    perturbed_batch.surf_vars[wind_var] += noise * mask_expanded
            
            for wind_var in ['u', 'v']:
                if wind_var in perturbed_batch.atmos_vars:
                    noise = torch.normal(0, magnitude, perturbed_batch.atmos_vars[wind_var].shape)
                    mask_expanded = spatial_mask[None, None, None, :, :]
                    perturbed_batch.atmos_vars[wind_var] += noise * mask_expanded
        
        return perturbed_batch
    
    def _create_level_perturbation(
        self,
        batch: Batch,
        variable_type: str,
        magnitude: float,
        target_level: Union[str, int]
    ) -> Batch:
        """Create level-specific perturbation."""
        
        import copy
        perturbed_batch = copy.deepcopy(batch)
        
        if target_level == 'surface':
            # Surface-only perturbation
            if variable_type == 'temperature':
                noise = torch.normal(0, magnitude, perturbed_batch.surf_vars['2t'].shape)
                perturbed_batch.surf_vars['2t'] += noise
            elif variable_type == 'winds':
                for var in ['10u', '10v']:
                    noise = torch.normal(0, magnitude, perturbed_batch.surf_vars[var].shape)
                    perturbed_batch.surf_vars[var] += noise
        
        else:
            # Specific atmospheric level
            if isinstance(target_level, int) and target_level in batch.metadata.atmos_levels:
                level_idx = list(batch.metadata.atmos_levels).index(target_level)
                
                if variable_type == 'temperature' and 't' in perturbed_batch.atmos_vars:
                    noise_shape = list(perturbed_batch.atmos_vars['t'].shape)
                    noise = torch.normal(0, magnitude, noise_shape)
                    perturbed_batch.atmos_vars['t'][:, :, level_idx] += noise[:, :, level_idx]
                
                elif variable_type == 'winds':
                    for var in ['u', 'v']:
                        if var in perturbed_batch.atmos_vars:
                            noise_shape = list(perturbed_batch.atmos_vars[var].shape)
                            noise = torch.normal(0, magnitude, noise_shape)
                            perturbed_batch.atmos_vars[var][:, :, level_idx] += noise[:, :, level_idx]
        
        return perturbed_batch
    
    def _run_perturbation_forecast(self, perturbed_batch: Batch, forecast_steps: int) -> Dict:
        """Run forecast for perturbed batch and compute metrics."""
        
        tracker = Tracker(
            init_lat=27.50, init_lon=132, init_time=datetime(2022, 9, 17, 12, 0)
        )
        
        predictions = []
        with torch.inference_mode():
            for pred in rollout(self.model, perturbed_batch, steps=forecast_steps):
                pred = pred.to("cpu")
                predictions.append(pred)
                tracker.step(pred)
        
        track_perturbed = tracker.results()
        
        # Compute deviations
        track_deviation = compute_track_deviation(self.baseline_track, track_perturbed)
        forecast_rmse = analyze_forecast_spread(self.baseline_forecast, predictions)
        
        return {
            'max_track_deviation_km': track_deviation['max_distance_km'],
            'final_track_deviation_km': track_deviation['final_distance_km'],
            'mean_track_deviation_km': track_deviation['mean_distance_km'],
            'track_deviation_time_series': track_deviation['distance_time_series'],
            'forecast_rmse': forecast_rmse,
            'track_perturbed': track_perturbed
        }
    
    def analyze_sensitivity_thresholds(self) -> Dict:
        """
        Analyze sensitivity thresholds across all tested dimensions.
        
        Returns:
            Dictionary with threshold analysis results
        """
        
        print("Analyzing sensitivity thresholds...")
        
        threshold_analysis = {}
        
        # Variable sensitivity thresholds
        if 'variable_sensitivity' in self.sensitivity_map:
            var_thresholds = {}
            
            for variable, results in self.sensitivity_map['variable_sensitivity'].items():
                # Find minimum magnitude that produces >1 km deviation
                threshold_mag = None
                for result in results:
                    if result['max_track_deviation_km'] > 1.0:
                        threshold_mag = result['magnitude']
                        break
                
                # Find saturation magnitude (where response plateaus)
                saturation_mag = None
                if len(results) > 3:
                    deviations = [r['max_track_deviation_km'] for r in results]
                    for i in range(len(deviations) - 2):
                        # Look for plateau (small increase despite magnitude increase)
                        slope = (deviations[i+2] - deviations[i]) / 2
                        if slope < 0.1:  # Less than 0.1 km increase per magnitude unit
                            saturation_mag = results[i]['magnitude']
                            break
                
                var_thresholds[variable] = {
                    'response_threshold': threshold_mag,
                    'saturation_threshold': saturation_mag,
                    'max_response': max([r['max_track_deviation_km'] for r in results]),
                    'sensitivity_evaluation': evaluate_perturbation_sensitivity(results, variable)
                }
            
            threshold_analysis['variable_thresholds'] = var_thresholds
        
        # Spatial scale thresholds
        if 'spatial_sensitivity' in self.sensitivity_map:
            spatial_results = self.sensitivity_map['spatial_sensitivity']
            
            # Find optimal spatial scale (maximum response)
            best_spatial = max(spatial_results, key=lambda x: x['max_track_deviation_km'])
            
            threshold_analysis['optimal_spatial_scale'] = {
                'scale': best_spatial['spatial_scale'],
                'scale_name': best_spatial['scale_name'],
                'max_response': best_spatial['max_track_deviation_km']
            }
        
        # Level sensitivity analysis
        if 'level_sensitivity' in self.sensitivity_map:
            level_results = self.sensitivity_map['level_sensitivity']
            
            # Find most sensitive level
            best_level = max(level_results, key=lambda x: x['max_track_deviation_km'])
            
            threshold_analysis['optimal_level'] = {
                'level': best_level['target_level'],
                'max_response': best_level['max_track_deviation_km']
            }
        
        self.threshold_analysis = threshold_analysis
        return threshold_analysis
    
    def optimize_perturbation_parameters(self) -> Dict:
        """
        Optimize perturbation parameters for maximum Aurora sensitivity.
        
        Returns:
            Optimized parameter recommendations
        """
        
        if not self.threshold_analysis:
            self.analyze_sensitivity_thresholds()
        
        print("Optimizing perturbation parameters...")
        
        # Find best variable type
        var_thresholds = self.threshold_analysis.get('variable_thresholds', {})
        
        if var_thresholds:
            # Rank variables by maximum response
            variable_ranking = sorted(
                var_thresholds.items(),
                key=lambda x: x[1]['max_response'],
                reverse=True
            )
            
            best_variable = variable_ranking[0][0]
            best_var_response = variable_ranking[0][1]['max_response']
        else:
            best_variable = 'winds'  # default
            best_var_response = 0.0
        
        # Find optimal magnitude (just above response threshold)
        optimal_magnitude = None
        if best_variable in var_thresholds:
            threshold_mag = var_thresholds[best_variable]['response_threshold']
            if threshold_mag:
                optimal_magnitude = threshold_mag * 1.5  # 50% above threshold
        
        if optimal_magnitude is None:
            optimal_magnitude = 2.0  # default
        
        # Get optimal spatial scale
        optimal_spatial = self.threshold_analysis.get('optimal_spatial_scale', {})
        optimal_level = self.threshold_analysis.get('optimal_level', {})
        
        optimization_results = {
            'recommended_variable': best_variable,
            'recommended_magnitude': optimal_magnitude,
            'expected_response_km': best_var_response,
            'optimal_spatial_scale': optimal_spatial.get('scale', (10, 10)),
            'optimal_level': optimal_level.get('level', 'surface'),
            'variable_ranking': variable_ranking,
            'optimization_summary': f\"Best configuration: {best_variable} at {optimal_magnitude} magnitude\"
        }
        
        self.parameter_optimization = optimization_results
        return optimization_results
    
    def generate_visualization_suite(self) -> None:
        """Generate comprehensive visualization suite of sensitivity results."""
        
        print("Generating visualization suite...")
        
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        # Variable sensitivity heatmap
        if 'variable_sensitivity' in self.sensitivity_map:
            self._plot_variable_sensitivity_heatmap(plot_dir)
        
        # Spatial sensitivity plot
        if 'spatial_sensitivity' in self.sensitivity_map:
            self._plot_spatial_sensitivity(plot_dir)
        
        # Level sensitivity plot
        if 'level_sensitivity' in self.sensitivity_map:
            self._plot_level_sensitivity(plot_dir)
        
        # Threshold analysis summary
        if self.threshold_analysis:
            self._plot_threshold_summary(plot_dir)
        
        print(f"Visualizations saved to {plot_dir}")
    
    def _plot_variable_sensitivity_heatmap(self, plot_dir: Path) -> None:
        """Plot variable sensitivity heatmap."""
        
        # Prepare data for heatmap
        variables = list(self.sensitivity_map['variable_sensitivity'].keys())
        all_magnitudes = set()
        
        for var_results in self.sensitivity_map['variable_sensitivity'].values():
            for result in var_results:
                all_magnitudes.add(result['magnitude'])
        
        all_magnitudes = sorted(list(all_magnitudes))
        
        # Create response matrix
        response_matrix = np.zeros((len(variables), len(all_magnitudes)))
        
        for i, variable in enumerate(variables):
            var_results = self.sensitivity_map['variable_sensitivity'][variable]
            for result in var_results:
                j = all_magnitudes.index(result['magnitude'])
                response_matrix[i, j] = result['max_track_deviation_km']
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            response_matrix,
            xticklabels=[f"{m:.1f}" for m in all_magnitudes],
            yticklabels=variables,
            annot=True,
            fmt='.1f',
            cmap='viridis',
            cbar_kws={'label': 'Max Track Deviation (km)'}
        )
        plt.title('Aurora Variable Sensitivity Heatmap')
        plt.xlabel('Perturbation Magnitude')
        plt.ylabel('Variable Type')
        plt.tight_layout()
        plt.savefig(plot_dir / 'variable_sensitivity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_spatial_sensitivity(self, plot_dir: Path) -> None:
        """Plot spatial scale sensitivity."""
        
        spatial_results = self.sensitivity_map['spatial_sensitivity']
        
        scale_names = [r['scale_name'] for r in spatial_results]
        responses = [r['max_track_deviation_km'] for r in spatial_results]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(scale_names, responses, color='skyblue', edgecolor='navy')
        plt.title('Aurora Spatial Scale Sensitivity')
        plt.xlabel('Spatial Scale')
        plt.ylabel('Max Track Deviation (km)')
        plt.xticks(rotation=45)
        
        # Highlight best scale
        max_idx = np.argmax(responses)
        bars[max_idx].set_color('orange')
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'spatial_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_level_sensitivity(self, plot_dir: Path) -> None:
        """Plot atmospheric level sensitivity."""
        
        level_results = self.sensitivity_map['level_sensitivity']
        
        levels = [str(r['target_level']) for r in level_results]
        responses = [r['max_track_deviation_km'] for r in level_results]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(levels, responses, color='lightcoral', edgecolor='darkred')
        plt.title('Aurora Atmospheric Level Sensitivity')
        plt.xlabel('Atmospheric Level')
        plt.ylabel('Max Track Deviation (km)')
        
        # Highlight best level
        max_idx = np.argmax(responses)
        bars[max_idx].set_color('red')
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'level_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_threshold_summary(self, plot_dir: Path) -> None:
        """Plot threshold analysis summary."""
        
        var_thresholds = self.threshold_analysis.get('variable_thresholds', {})
        
        if not var_thresholds:
            return
        
        variables = list(var_thresholds.keys())
        thresholds = [var_thresholds[v]['response_threshold'] or 0 for v in variables]
        max_responses = [var_thresholds[v]['max_response'] for v in variables]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Response thresholds
        ax1.bar(variables, thresholds, color='lightgreen', edgecolor='darkgreen')
        ax1.set_title('Response Thresholds by Variable')
        ax1.set_ylabel('Threshold Magnitude')
        ax1.tick_params(axis='x', rotation=45)
        
        # Maximum responses
        ax2.bar(variables, max_responses, color='lightblue', edgecolor='darkblue')
        ax2.set_title('Maximum Response by Variable')
        ax2.set_ylabel('Max Track Deviation (km)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'threshold_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_comprehensive_results(self) -> None:
        """Save all sensitivity mapping results."""
        
        print("Saving comprehensive results...")
        
        # Save sensitivity maps
        with open(self.output_dir / 'sensitivity_map.pkl', 'wb') as f:
            pickle.dump(self.sensitivity_map, f)
        
        # Save threshold analysis
        if self.threshold_analysis:
            with open(self.output_dir / 'threshold_analysis.pkl', 'wb') as f:
                pickle.dump(self.threshold_analysis, f)
        
        # Save parameter optimization
        if self.parameter_optimization:
            with open(self.output_dir / 'parameter_optimization.pkl', 'wb') as f:
                pickle.dump(self.parameter_optimization, f)
        
        # Generate and save comprehensive report
        report = self.generate_comprehensive_report()
        with open(self.output_dir / 'sensitivity_mapping_report.txt', 'w') as f:
            f.write(report)
        
        print(f"Results saved to {self.output_dir}")
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive sensitivity mapping report."""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("AURORA COMPREHENSIVE SENSITIVITY MAPPING REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 40)
        
        total_tests = 0
        max_response = 0.0
        
        for mapping_type, results in self.sensitivity_map.items():
            if mapping_type == 'variable_sensitivity':
                for var_results in results.values():
                    total_tests += len(var_results)
                    max_response = max(max_response, max([r['max_track_deviation_km'] for r in var_results]))
            else:
                total_tests += len(results)
                max_response = max(max_response, max([r['max_track_deviation_km'] for r in results]))
        
        report_lines.append(f"Total perturbation tests conducted: {total_tests}")
        report_lines.append(f"Maximum observed response: {max_response:.2f} km")
        report_lines.append("")
        
        # Parameter optimization results
        if self.parameter_optimization:
            opt = self.parameter_optimization
            report_lines.append("OPTIMAL PARAMETER CONFIGURATION")
            report_lines.append("-" * 40)
            report_lines.append(f"Best variable type: {opt['recommended_variable']}")
            report_lines.append(f"Optimal magnitude: {opt['recommended_magnitude']:.2f}")
            report_lines.append(f"Expected response: {opt['expected_response_km']:.2f} km")
            report_lines.append(f"Optimal spatial scale: {opt['optimal_spatial_scale']}")
            report_lines.append(f"Optimal atmospheric level: {opt['optimal_level']}")
            report_lines.append("")
        
        # Threshold analysis
        if self.threshold_analysis and 'variable_thresholds' in self.threshold_analysis:
            report_lines.append("VARIABLE SENSITIVITY ANALYSIS")
            report_lines.append("-" * 40)
            
            var_thresholds = self.threshold_analysis['variable_thresholds']
            for variable, analysis in var_thresholds.items():
                report_lines.append(f"{variable.upper()}:")
                report_lines.append(f"  Response threshold: {analysis['response_threshold']}")
                report_lines.append(f"  Maximum response: {analysis['max_response']:.2f} km")
                report_lines.append(f"  Sensitivity level: {analysis['sensitivity_evaluation']['sensitivity_level']}")
                report_lines.append("")
        
        # Ensemble forecasting assessment
        report_lines.append("ENSEMBLE FORECASTING ASSESSMENT")
        report_lines.append("-" * 40)
        
        if max_response < 1.0:
            assessment = "UNSUITABLE - Aurora shows insufficient sensitivity for ensemble forecasting"
            recommendations = [
                "- Consider alternative models for ensemble applications",
                "- Investigate non-perturbation based uncertainty methods",
                "- Focus on deterministic forecasting applications"
            ]
        elif max_response < 10.0:
            assessment = "LIMITED - Aurora may support ensemble with enhanced perturbations"
            recommendations = [
                "- Use optimal parameter configuration from this analysis",
                "- Consider larger perturbation magnitudes",
                "- Test compound perturbation strategies"
            ]
        else:
            assessment = "SUITABLE - Aurora can support ensemble forecasting"
            recommendations = [
                "- Implement ensemble system with optimal parameters",
                "- Develop ensemble post-processing and calibration",
                "- Test operational ensemble deployment"
            ]
        
        report_lines.append(f"Assessment: {assessment}")
        report_lines.append("")
        report_lines.append("Recommendations:")
        for rec in recommendations:
            report_lines.append(rec)
        
        return "\n".join(report_lines)
    
    def _save_intermediate_result(self, variable: str, magnitude: float, result: Dict) -> None:
        """Save intermediate result to prevent data loss."""
        
        intermediate_dir = self.output_dir / "intermediate"
        intermediate_dir.mkdir(exist_ok=True)
        
        filename = f"{variable}_{magnitude:.3f}_result.pkl"
        
        with open(intermediate_dir / filename, 'wb') as f:
            pickle.dump(result, f)


def run_full_sensitivity_mapping(
    model: Aurora,
    output_directory: str = "aurora_sensitivity_mapping",
    quick_mode: bool = False
) -> AuroraSensitivityMapper:
    """
    Run complete sensitivity mapping experiment.
    
    Args:
        model: Aurora model instance
        output_directory: Directory for saving results
        quick_mode: Whether to run in quick mode (fewer tests)
        
    Returns:
        AuroraSensitivityMapper instance with results
    """
    
    print("Starting Aurora comprehensive sensitivity mapping...")
    
    # Initialize mapper
    mapper = AuroraSensitivityMapper(model, output_directory=output_directory)
    
    # Load baseline case
    mapper.load_baseline_case()
    
    # Configure test parameters
    if quick_mode:
        variables = ['temperature', 'winds', 'pressure']
        magnitude_ranges = {
            'temperature': [0.5, 1.0, 2.0, 5.0],
            'winds': [1.0, 2.0, 5.0, 10.0],
            'pressure': [100, 200, 500, 1000]
        }
        forecast_steps = 20
    else:
        variables = ['temperature', 'winds', 'pressure', 'humidity', 'geopotential']
        magnitude_ranges = None  # Use defaults
        forecast_steps = 40
    
    # Run sensitivity mapping
    print("\n1. Variable sensitivity mapping...")
    mapper.map_variable_sensitivity(variables, magnitude_ranges, forecast_steps)
    
    print("\n2. Spatial scale sensitivity mapping...")
    mapper.map_spatial_scale_sensitivity(forecast_steps=forecast_steps)
    
    print("\n3. Atmospheric level sensitivity mapping...")
    mapper.map_atmospheric_level_sensitivity(forecast_steps=forecast_steps)
    
    if not quick_mode:
        print("\n4. Temporal persistence sensitivity mapping...")
        mapper.map_temporal_persistence_sensitivity(forecast_steps=forecast_steps)
    
    # Analysis and optimization
    print("\n5. Analyzing sensitivity thresholds...")
    mapper.analyze_sensitivity_thresholds()
    
    print("\n6. Optimizing perturbation parameters...")
    mapper.optimize_perturbation_parameters()
    
    # Generate visualizations
    print("\n7. Generating visualization suite...")
    mapper.generate_visualization_suite()
    
    # Save comprehensive results
    print("\n8. Saving comprehensive results...")
    mapper.save_comprehensive_results()
    
    # Display summary
    print("\n" + "="*60)
    print("SENSITIVITY MAPPING COMPLETED")
    print("="*60)
    
    if mapper.parameter_optimization:
        opt = mapper.parameter_optimization
        print(f"Optimal configuration: {opt['optimization_summary']}")
        print(f"Expected response: {opt['expected_response_km']:.2f} km")
    
    print(f"Full results saved to: {mapper.output_dir}")
    print(f"Summary report: {mapper.output_dir}/sensitivity_mapping_report.txt")
    
    return mapper


if __name__ == "__main__":
    # Example usage
    print("Aurora Sensitivity Threshold Mapping")
    print("This script requires Aurora model and data to be available")
    print("Run run_full_sensitivity_mapping() with loaded Aurora model")