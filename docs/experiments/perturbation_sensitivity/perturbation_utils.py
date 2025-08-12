"""
Utility functions for Aurora perturbation sensitivity experiments.

This module provides tools for:
1. Adding realistic perturbations to Aurora input batches
2. Computing forecast deviations and sensitivity metrics
3. Analyzing ensemble spread and forecast uncertainty
4. Comparing with operational NWP ensemble statistics
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from scipy.spatial.distance import euclidean
from aurora.batch import Batch
import copy


def add_gaussian_perturbation(
    batch: Batch, 
    perturbation_type: str, 
    magnitude: Union[float, Dict], 
    variables: Optional[List[str]] = None,
    spatial_correlation: float = 0.0,
    random_seed: Optional[int] = None
) -> Batch:
    """
    Add Gaussian perturbations to Aurora input batch.
    
    Args:
        batch: Aurora Batch object to perturb
        perturbation_type: Type of perturbation ('temperature', 'winds', 'pressure', 'combined')
        magnitude: Perturbation magnitude (units depend on type)
        variables: Optional list of specific variables to perturb
        spatial_correlation: Spatial correlation length (0 = uncorrelated noise)
        random_seed: Random seed for reproducibility
        
    Returns:
        Perturbed batch object
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # Create deep copy to avoid modifying original
    perturbed_batch = copy.deepcopy(batch)
    
    if perturbation_type == 'temperature':
        perturbed_batch = _add_temperature_perturbation(perturbed_batch, magnitude, variables)
    elif perturbation_type == 'winds':
        perturbed_batch = _add_wind_perturbation(perturbed_batch, magnitude, variables)
    elif perturbation_type == 'pressure':
        perturbed_batch = _add_pressure_perturbation(perturbed_batch, magnitude, variables)
    elif perturbation_type == 'combined':
        perturbed_batch = _add_combined_perturbation(perturbed_batch, magnitude, variables)
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")
    
    return perturbed_batch


def _add_temperature_perturbation(batch: Batch, magnitude: float, variables: Optional[List[str]]) -> Batch:
    """Add Gaussian noise to temperature fields."""
    
    # Surface temperature
    if variables is None or '2t' in variables:
        noise_shape = batch.surf_vars['2t'].shape
        temp_noise = torch.normal(0, magnitude, noise_shape)
        batch.surf_vars['2t'] = batch.surf_vars['2t'] + temp_noise
    
    # Atmospheric temperature
    if variables is None or 't' in variables:
        noise_shape = batch.atmos_vars['t'].shape
        temp_noise = torch.normal(0, magnitude, noise_shape)
        batch.atmos_vars['t'] = batch.atmos_vars['t'] + temp_noise
    
    return batch


def _add_wind_perturbation(batch: Batch, magnitude: float, variables: Optional[List[str]]) -> Batch:
    """Add Gaussian noise to wind fields."""
    
    # Surface winds
    if variables is None or '10u' in variables:
        noise_shape = batch.surf_vars['10u'].shape
        u_noise = torch.normal(0, magnitude, noise_shape)
        batch.surf_vars['10u'] = batch.surf_vars['10u'] + u_noise
    
    if variables is None or '10v' in variables:
        noise_shape = batch.surf_vars['10v'].shape
        v_noise = torch.normal(0, magnitude, noise_shape)
        batch.surf_vars['10v'] = batch.surf_vars['10v'] + v_noise
    
    # Atmospheric winds
    if variables is None or 'u' in variables:
        noise_shape = batch.atmos_vars['u'].shape
        u_noise = torch.normal(0, magnitude, noise_shape)
        batch.atmos_vars['u'] = batch.atmos_vars['u'] + u_noise
    
    if variables is None or 'v' in variables:
        noise_shape = batch.atmos_vars['v'].shape
        v_noise = torch.normal(0, magnitude, noise_shape)
        batch.atmos_vars['v'] = batch.atmos_vars['v'] + v_noise
    
    return batch


def _add_pressure_perturbation(batch: Batch, magnitude: float, variables: Optional[List[str]]) -> Batch:
    """Add Gaussian noise to pressure fields."""
    
    # Mean sea level pressure
    if variables is None or 'msl' in variables:
        noise_shape = batch.surf_vars['msl'].shape
        pressure_noise = torch.normal(0, magnitude, noise_shape)
        batch.surf_vars['msl'] = batch.surf_vars['msl'] + pressure_noise
    
    return batch


def _add_combined_perturbation(batch: Batch, magnitude_dict: Dict, variables: Optional[List[str]]) -> Batch:
    """Add combined perturbations to multiple variable types."""
    
    if 'temp' in magnitude_dict:
        batch = _add_temperature_perturbation(batch, magnitude_dict['temp'], variables)
    
    if 'wind' in magnitude_dict:
        batch = _add_wind_perturbation(batch, magnitude_dict['wind'], variables)
    
    if 'pressure' in magnitude_dict:
        batch = _add_pressure_perturbation(batch, magnitude_dict['pressure'], variables)
    
    return batch


def compute_track_deviation(
    track_baseline: pd.DataFrame, 
    track_perturbed: pd.DataFrame
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute track deviation metrics between baseline and perturbed forecasts.
    
    Args:
        track_baseline: Baseline track DataFrame from Aurora Tracker
        track_perturbed: Perturbed track DataFrame from Aurora Tracker
        
    Returns:
        Dictionary with deviation metrics
    """
    
    # Ensure same number of time points
    min_length = min(len(track_baseline), len(track_perturbed))
    track_base = track_baseline.iloc[:min_length]
    track_pert = track_perturbed.iloc[:min_length]
    
    # Calculate great circle distances
    distances_km = []
    for i in range(min_length):
        lat1, lon1 = track_base.iloc[i]['lat'], track_base.iloc[i]['lon']
        lat2, lon2 = track_pert.iloc[i]['lat'], track_pert.iloc[i]['lon']
        
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        distances_km.append(distance)
    
    distances_km = np.array(distances_km)
    
    return {
        'distance_time_series': distances_km,
        'max_distance_km': np.max(distances_km),
        'final_distance_km': distances_km[-1] if len(distances_km) > 0 else 0.0,
        'mean_distance_km': np.mean(distances_km),
        'std_distance_km': np.std(distances_km)
    }


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points on Earth.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
        
    Returns:
        Distance in kilometers
    """
    R = 6371.0  # Earth radius in km
    
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def analyze_forecast_spread(
    predictions_baseline: List, 
    predictions_perturbed: List, 
    variables: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Analyze forecast differences between baseline and perturbed predictions.
    
    Args:
        predictions_baseline: List of baseline Aurora predictions
        predictions_perturbed: List of perturbed Aurora predictions
        variables: Optional list of variables to analyze
        
    Returns:
        Dictionary with RMSE values for each variable
    """
    
    if variables is None:
        # Default variables to analyze
        variables = ['2t', '10u', '10v', 'msl']
    
    rmse_results = {}
    
    for var in variables:
        if var in predictions_baseline[0].surf_vars:
            var_type = 'surf_vars'
        elif var in predictions_baseline[0].atmos_vars:
            var_type = 'atmos_vars'
        else:
            continue
        
        # Compute RMSE across all time steps
        rmse_values = []
        for pred_base, pred_pert in zip(predictions_baseline, predictions_perturbed):
            base_field = getattr(pred_base, var_type)[var]
            pert_field = getattr(pred_pert, var_type)[var]
            
            # Compute RMSE
            diff = base_field - pert_field
            rmse = torch.sqrt(torch.mean(diff**2)).item()
            rmse_values.append(rmse)
        
        rmse_results[var] = np.mean(rmse_values)
    
    return rmse_results


def generate_ensemble_perturbations(
    batch_baseline: Batch,
    n_members: int,
    perturbation_config: Dict,
    correlation_length: float = 0.0
) -> List[Batch]:
    """
    Generate ensemble of perturbed initial conditions.
    
    Args:
        batch_baseline: Baseline initial conditions
        n_members: Number of ensemble members
        perturbation_config: Configuration for perturbation types and magnitudes
        correlation_length: Spatial correlation length for perturbations
        
    Returns:
        List of perturbed batch objects
    """
    
    ensemble_members = []
    
    for member in range(n_members):
        # Use different random seed for each member
        perturbed_batch = add_gaussian_perturbation(
            batch_baseline,
            perturbation_type=perturbation_config.get('type', 'combined'),
            magnitude=perturbation_config.get('magnitude', 1.0),
            variables=perturbation_config.get('variables', None),
            random_seed=member
        )
        
        ensemble_members.append(perturbed_batch)
    
    return ensemble_members


def analyze_ensemble_statistics(
    ensemble_forecasts: List[List],
    variables: Optional[List[str]] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Analyze ensemble forecast statistics (mean, spread, etc.).
    
    Args:
        ensemble_forecasts: List of forecast sequences from ensemble members
        variables: Variables to analyze
        
    Returns:
        Dictionary with ensemble statistics for each variable
    """
    
    if variables is None:
        variables = ['2t', '10u', '10v', 'msl']
    
    n_members = len(ensemble_forecasts)
    n_steps = len(ensemble_forecasts[0])
    
    ensemble_stats = {}
    
    for var in variables:
        # Determine variable type
        if var in ensemble_forecasts[0][0].surf_vars:
            var_type = 'surf_vars'
        elif var in ensemble_forecasts[0][0].atmos_vars:
            var_type = 'atmos_vars'
        else:
            continue
        
        # Collect ensemble data
        ensemble_data = []
        for member_forecasts in ensemble_forecasts:
            member_data = []
            for forecast in member_forecasts:
                field = getattr(forecast, var_type)[var]
                member_data.append(field.numpy())
            ensemble_data.append(member_data)
        
        # Convert to numpy array: (members, time_steps, ...)
        ensemble_array = np.array(ensemble_data)
        
        # Compute statistics
        ensemble_mean = np.mean(ensemble_array, axis=0)
        ensemble_std = np.std(ensemble_array, axis=0)
        ensemble_min = np.min(ensemble_array, axis=0)
        ensemble_max = np.max(ensemble_array, axis=0)
        
        # Compute spread metrics
        spread_time_series = np.mean(ensemble_std, axis=(2, 3))  # Spatial average of std
        spread_growth = spread_time_series[-1] / spread_time_series[0] if spread_time_series[0] > 0 else 0
        
        ensemble_stats[var] = {
            'mean': ensemble_mean,
            'std': ensemble_std,
            'min': ensemble_min,
            'max': ensemble_max,
            'spread_time_series': spread_time_series,
            'spread_growth_factor': spread_growth
        }
    
    return ensemble_stats


def compare_with_gcm_ensemble(
    aurora_ensemble_stats: Dict,
    gcm_reference_spread: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Compare Aurora ensemble spread with typical GCM ensemble characteristics.
    
    Args:
        aurora_ensemble_stats: Ensemble statistics from Aurora
        gcm_reference_spread: Reference GCM ensemble spread values
        
    Returns:
        Comparison metrics
    """
    
    if gcm_reference_spread is None:
        # Typical GCM ensemble spread for key variables (rough estimates)
        gcm_reference_spread = {
            '2t': 2.0,      # 2K temperature spread at 5 days
            '10u': 5.0,     # 5 m/s wind spread at 5 days  
            '10v': 5.0,     # 5 m/s wind spread at 5 days
            'msl': 500.0    # 500 Pa pressure spread at 5 days
        }
    
    comparison_ratios = {}
    
    for var in aurora_ensemble_stats:
        if var in gcm_reference_spread:
            aurora_spread = aurora_ensemble_stats[var]['spread_time_series'][-1]
            gcm_spread = gcm_reference_spread[var]
            
            ratio = aurora_spread / gcm_spread
            comparison_ratios[var] = ratio
    
    return comparison_ratios


def evaluate_perturbation_sensitivity(
    sensitivity_results: List[Dict],
    perturbation_type: str
) -> Dict[str, Union[float, str]]:
    """
    Evaluate overall sensitivity characteristics from perturbation test results.
    
    Args:
        sensitivity_results: List of results from sensitivity tests
        perturbation_type: Type of perturbation tested
        
    Returns:
        Sensitivity evaluation metrics
    """
    
    magnitudes = [r['magnitude'] for r in sensitivity_results]
    deviations = [r['max_track_deviation_km'] for r in sensitivity_results]
    
    # Linear regression to find sensitivity slope
    if len(magnitudes) > 1:
        coeffs = np.polyfit(magnitudes, deviations, 1)
        sensitivity_slope = coeffs[0]
        correlation = np.corrcoef(magnitudes, deviations)[0, 1]
    else:
        sensitivity_slope = 0.0
        correlation = 0.0
    
    # Find threshold for meaningful response (>1 km deviation)
    threshold_magnitude = None
    for result in sensitivity_results:
        if result['max_track_deviation_km'] > 1.0:
            threshold_magnitude = result['magnitude']
            break
    
    # Assess sensitivity level
    if sensitivity_slope < 0.1:
        sensitivity_level = "Very Low"
    elif sensitivity_slope < 0.5:
        sensitivity_level = "Low"
    elif sensitivity_slope < 2.0:
        sensitivity_level = "Moderate"
    else:
        sensitivity_level = "High"
    
    return {
        'sensitivity_slope': sensitivity_slope,
        'correlation': correlation,
        'threshold_magnitude': threshold_magnitude,
        'sensitivity_level': sensitivity_level,
        'max_deviation_tested': max(deviations),
        'perturbation_type': perturbation_type
    }


def create_perturbation_report(
    all_results: Dict,
    output_file: str = "perturbation_sensitivity_report.txt"
) -> str:
    """
    Create comprehensive report of perturbation sensitivity experiments.
    
    Args:
        all_results: Dictionary containing all experimental results
        output_file: Output file name for the report
        
    Returns:
        Report text content
    """
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("AURORA PERTURBATION SENSITIVITY ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Executive summary
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("-" * 40)
    
    for pert_type in ['temperature', 'winds', 'pressure']:
        if f'{pert_type}_results' in all_results:
            results = all_results[f'{pert_type}_results']
            evaluation = evaluate_perturbation_sensitivity(results, pert_type)
            
            report_lines.append(f"{pert_type.upper()} PERTURBATIONS:")
            report_lines.append(f"  Sensitivity Level: {evaluation['sensitivity_level']}")
            report_lines.append(f"  Maximum Deviation: {evaluation['max_deviation_tested']:.2f} km")
            
            if evaluation['threshold_magnitude']:
                report_lines.append(f"  1km Threshold: {evaluation['threshold_magnitude']}")
            else:
                report_lines.append(f"  1km Threshold: Not reached in tested range")
            
            report_lines.append("")
    
    # Detailed results
    report_lines.append("DETAILED RESULTS")
    report_lines.append("-" * 40)
    
    if 'summary_dataframe' in all_results:
        df = all_results['summary_dataframe']
        report_lines.append("Summary Statistics:")
        report_lines.append(df.to_string(index=False))
        report_lines.append("")
    
    # Recommendations
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 40)
    report_lines.append("1. Based on sensitivity analysis:")
    
    # Determine overall assessment
    max_deviation = 0
    for pert_type in ['temperature', 'winds', 'pressure']:
        if f'{pert_type}_results' in all_results:
            results = all_results[f'{pert_type}_results']
            type_max = max([r['max_track_deviation_km'] for r in results])
            max_deviation = max(max_deviation, type_max)
    
    if max_deviation < 1.0:
        report_lines.append("   - Aurora shows very low sensitivity to realistic perturbations")
        report_lines.append("   - Ensemble forecasting may not be feasible with current approach")
        report_lines.append("   - Consider larger perturbations or alternative strategies")
    elif max_deviation < 10.0:
        report_lines.append("   - Aurora shows limited sensitivity to realistic perturbations")
        report_lines.append("   - Ensemble forecasting may be possible with enhanced perturbations")
        report_lines.append("   - Focus on most sensitive variable types")
    else:
        report_lines.append("   - Aurora shows reasonable sensitivity to perturbations")
        report_lines.append("   - Ensemble forecasting appears feasible")
        report_lines.append("   - Proceed with full ensemble framework development")
    
    report_lines.append("")
    report_lines.append("2. Next Steps:")
    report_lines.append("   - Implement systematic ensemble generation framework")
    report_lines.append("   - Test spatial perturbation patterns")
    report_lines.append("   - Compare with operational NWP ensemble spread")
    report_lines.append("   - Investigate alternative perturbation strategies")
    
    report_text = "\n".join(report_lines)
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    return report_text