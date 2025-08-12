"""
Aurora Ensemble Forecasting Framework

This module provides a comprehensive framework for generating and analyzing
ensemble forecasts with Aurora, including:
1. Ensemble member generation with various perturbation strategies
2. Parallel forecast execution and management
3. Ensemble statistics computation and analysis
4. Comparison with operational NWP ensemble characteristics
5. Uncertainty quantification and visualization tools
"""

import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import pickle
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from aurora import Aurora, Tracker, rollout, Batch
from perturbation_utils import (
    add_gaussian_perturbation,
    compute_track_deviation,
    analyze_ensemble_statistics,
    compare_with_gcm_ensemble
)


class AuroraEnsemble:
    """
    Aurora Ensemble Forecasting System
    
    Provides functionality for generating ensemble forecasts with Aurora
    using various perturbation strategies and uncertainty quantification methods.
    """
    
    def __init__(
        self,
        model: Aurora,
        n_members: int = 20,
        device: str = "cuda",
        parallel_execution: bool = True,
        max_workers: Optional[int] = None
    ):
        """
        Initialize Aurora ensemble forecasting system.
        
        Args:
            model: Aurora model instance
            n_members: Number of ensemble members
            device: Device for model execution
            parallel_execution: Whether to run members in parallel
            max_workers: Maximum number of parallel workers
        """
        self.model = model
        self.n_members = n_members
        self.device = device
        self.parallel_execution = parallel_execution
        self.max_workers = max_workers or min(4, n_members)
        
        # Ensemble configuration
        self.perturbation_config = None
        self.ensemble_members = []
        self.forecast_results = []
        
        # Statistics and analysis
        self.ensemble_stats = {}
        self.uncertainty_metrics = {}
        
    def configure_perturbations(
        self,
        strategy: str = "realistic_obs",
        custom_config: Optional[Dict] = None
    ) -> None:
        """
        Configure perturbation strategy for ensemble generation.
        
        Args:
            strategy: Predefined strategy name or 'custom'
            custom_config: Custom perturbation configuration
        """
        
        if strategy == "realistic_obs":
            # Realistic observational uncertainty
            self.perturbation_config = {
                'temperature': {
                    'magnitude': 0.5,  # K
                    'variables': ['2t', 't']
                },
                'winds': {
                    'magnitude': 1.5,  # m/s
                    'variables': ['10u', '10v', 'u', 'v']
                },
                'pressure': {
                    'magnitude': 100,  # Pa
                    'variables': ['msl']
                }
            }
            
        elif strategy == "enhanced_uncertainty":
            # Enhanced uncertainty for better ensemble spread
            self.perturbation_config = {
                'temperature': {
                    'magnitude': 2.0,  # K
                    'variables': ['2t', 't']
                },
                'winds': {
                    'magnitude': 5.0,  # m/s
                    'variables': ['10u', '10v', 'u', 'v']
                },
                'pressure': {
                    'magnitude': 300,  # Pa
                    'variables': ['msl']
                }
            }
            
        elif strategy == "analysis_uncertainty":
            # Typical analysis uncertainty estimates
            self.perturbation_config = {
                'temperature': {
                    'magnitude': 1.0,  # K
                    'variables': ['2t', 't']
                },
                'winds': {
                    'magnitude': 3.0,  # m/s
                    'variables': ['10u', '10v', 'u', 'v']
                },
                'pressure': {
                    'magnitude': 200,  # Pa
                    'variables': ['msl']
                }
            }
            
        elif strategy == "custom":
            if custom_config is None:
                raise ValueError("Custom configuration required for 'custom' strategy")
            self.perturbation_config = custom_config
            
        else:
            raise ValueError(f"Unknown perturbation strategy: {strategy}")
    
    def generate_ensemble_members(
        self,
        initial_conditions: Batch,
        perturbation_strategy: str = "realistic_obs",
        custom_config: Optional[Dict] = None
    ) -> List[Batch]:
        """
        Generate ensemble of perturbed initial conditions.
        
        Args:
            initial_conditions: Baseline initial conditions
            perturbation_strategy: Strategy for perturbation generation
            custom_config: Custom perturbation configuration
            
        Returns:
            List of perturbed initial condition batches
        """
        
        # Configure perturbations
        self.configure_perturbations(perturbation_strategy, custom_config)
        
        print(f"Generating {self.n_members} ensemble members...")
        print(f"Perturbation strategy: {perturbation_strategy}")
        
        self.ensemble_members = []
        
        for member_id in range(self.n_members):
            # Create perturbed initial conditions
            perturbed_ic = self._create_perturbed_member(
                initial_conditions, 
                member_id
            )
            
            self.ensemble_members.append({
                'member_id': member_id,
                'initial_conditions': perturbed_ic,
                'perturbation_config': self.perturbation_config.copy()
            })
        
        print(f"Ensemble member generation completed: {len(self.ensemble_members)} members")
        return self.ensemble_members
    
    def _create_perturbed_member(
        self,
        initial_conditions: Batch,
        member_id: int
    ) -> Batch:
        """
        Create single perturbed ensemble member.
        
        Args:
            initial_conditions: Baseline initial conditions
            member_id: Ensemble member identifier
            
        Returns:
            Perturbed initial conditions
        """
        
        perturbed_batch = initial_conditions
        
        # Apply perturbations for each variable type
        for pert_type, config in self.perturbation_config.items():
            perturbed_batch = add_gaussian_perturbation(
                perturbed_batch,
                perturbation_type=pert_type,
                magnitude=config['magnitude'],
                variables=config.get('variables', None),
                random_seed=member_id * 100 + hash(pert_type) % 1000
            )
        
        return perturbed_batch
    
    def run_ensemble_forecast(
        self,
        forecast_steps: int = 40,
        save_all_steps: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Run ensemble forecast for all members.
        
        Args:
            forecast_steps: Number of forecast time steps
            save_all_steps: Whether to save all intermediate steps
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing ensemble forecast results
        """
        
        if not self.ensemble_members:
            raise ValueError("No ensemble members generated. Call generate_ensemble_members() first.")
        
        print(f"Running ensemble forecast for {len(self.ensemble_members)} members...")
        print(f"Forecast steps: {forecast_steps}")
        print(f"Parallel execution: {self.parallel_execution}")
        
        start_time = time.time()
        
        if self.parallel_execution:
            results = self._run_parallel_forecast(forecast_steps, save_all_steps, progress_callback)
        else:
            results = self._run_sequential_forecast(forecast_steps, save_all_steps, progress_callback)
        
        execution_time = time.time() - start_time
        
        # Store results
        self.forecast_results = results
        
        print(f"Ensemble forecast completed in {execution_time:.1f} seconds")
        print(f"Average time per member: {execution_time/len(self.ensemble_members):.1f} seconds")
        
        return {
            'ensemble_forecasts': results,
            'execution_time': execution_time,
            'forecast_steps': forecast_steps,
            'n_members': len(self.ensemble_members)
        }
    
    def _run_parallel_forecast(
        self,
        forecast_steps: int,
        save_all_steps: bool,
        progress_callback: Optional[Callable]
    ) -> List[Dict]:
        """Run ensemble forecasts in parallel."""
        
        results = [None] * len(self.ensemble_members)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all forecast jobs
            future_to_member = {
                executor.submit(
                    self._run_single_member_forecast,
                    member['member_id'],
                    member['initial_conditions'],
                    forecast_steps,
                    save_all_steps
                ): member['member_id']
                for member in self.ensemble_members
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_member):
                member_id = future_to_member[future]
                try:
                    result = future.result()
                    results[member_id] = result
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, len(self.ensemble_members))
                    else:
                        print(f"  Completed member {member_id + 1}/{len(self.ensemble_members)}")
                        
                except Exception as e:
                    print(f"Member {member_id} failed: {e}")
                    results[member_id] = None
        
        return [r for r in results if r is not None]
    
    def _run_sequential_forecast(
        self,
        forecast_steps: int,
        save_all_steps: bool,
        progress_callback: Optional[Callable]
    ) -> List[Dict]:
        """Run ensemble forecasts sequentially."""
        
        results = []
        
        for i, member in enumerate(self.ensemble_members):
            print(f"  Running member {i + 1}/{len(self.ensemble_members)}")
            
            try:
                result = self._run_single_member_forecast(
                    member['member_id'],
                    member['initial_conditions'],
                    forecast_steps,
                    save_all_steps
                )
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(self.ensemble_members))
                    
            except Exception as e:
                print(f"Member {i} failed: {e}")
                continue
        
        return results
    
    def _run_single_member_forecast(
        self,
        member_id: int,
        initial_conditions: Batch,
        forecast_steps: int,
        save_all_steps: bool
    ) -> Dict:
        """
        Run forecast for single ensemble member.
        
        Args:
            member_id: Ensemble member identifier
            initial_conditions: Initial conditions for this member
            forecast_steps: Number of forecast steps
            save_all_steps: Whether to save all time steps
            
        Returns:
            Dictionary with member forecast results
        """
        
        # Initialize tracker
        tracker = Tracker(
            init_lat=27.50,  # Typhoon Nanmadol initial position
            init_lon=132,
            init_time=datetime(2022, 9, 17, 12, 0)
        )
        
        # Run forecast
        predictions = []
        
        with torch.inference_mode():
            for step, pred in enumerate(rollout(self.model, initial_conditions, steps=forecast_steps)):
                pred = pred.to("cpu")  # Free GPU memory
                
                if save_all_steps:
                    predictions.append(pred)
                
                # Update tracker
                tracker.step(pred)
        
        # Get track results
        track = tracker.results()
        
        return {
            'member_id': member_id,
            'track': track,
            'predictions': predictions if save_all_steps else None,
            'final_prediction': predictions[-1] if predictions else None
        }
    
    def analyze_ensemble_uncertainty(
        self,
        variables: Optional[List[str]] = None,
        compute_correlations: bool = True
    ) -> Dict:
        """
        Analyze ensemble uncertainty and spread characteristics.
        
        Args:
            variables: Variables to analyze (default: key variables)
            compute_correlations: Whether to compute inter-variable correlations
            
        Returns:
            Dictionary with uncertainty analysis results
        """
        
        if not self.forecast_results:
            raise ValueError("No forecast results available. Run ensemble forecast first.")
        
        if variables is None:
            variables = ['2t', '10u', '10v', 'msl']
        
        print(f"Analyzing ensemble uncertainty for {len(variables)} variables...")
        
        # Track-based uncertainty analysis
        track_uncertainty = self._analyze_track_uncertainty()
        
        # Field-based uncertainty analysis
        if any(r['predictions'] for r in self.forecast_results):
            field_uncertainty = self._analyze_field_uncertainty(variables)
        else:
            field_uncertainty = {}
        
        # Ensemble spread growth analysis
        spread_analysis = self._analyze_spread_growth()
        
        # Store results
        self.uncertainty_metrics = {
            'track_uncertainty': track_uncertainty,
            'field_uncertainty': field_uncertainty,
            'spread_analysis': spread_analysis,
            'ensemble_size': len(self.forecast_results)
        }
        
        return self.uncertainty_metrics
    
    def _analyze_track_uncertainty(self) -> Dict:
        """Analyze tropical cyclone track uncertainty."""
        
        # Extract all tracks
        tracks = [result['track'] for result in self.forecast_results]
        
        # Compute track spread at each time
        track_spreads = []
        track_means = []
        
        max_length = max(len(track) for track in tracks)
        
        for t in range(max_length):
            lats = []
            lons = []
            
            for track in tracks:
                if t < len(track):
                    lats.append(track.iloc[t]['lat'])
                    lons.append(track.iloc[t]['lon'])
            
            if len(lats) > 1:
                lat_std = np.std(lats)
                lon_std = np.std(lons)
                
                # Convert to approximate km (rough estimate)
                lat_spread_km = lat_std * 111  # 1 degree lat ≈ 111 km
                lon_spread_km = lon_std * 111 * np.cos(np.radians(np.mean(lats)))
                
                total_spread_km = np.sqrt(lat_spread_km**2 + lon_spread_km**2)
                
                track_spreads.append(total_spread_km)
                track_means.append({'lat': np.mean(lats), 'lon': np.mean(lons)})
            else:
                track_spreads.append(0.0)
                track_means.append({'lat': lats[0] if lats else 0, 'lon': lons[0] if lons else 0})
        
        return {
            'track_spread_km': np.array(track_spreads),
            'track_means': track_means,
            'initial_spread_km': track_spreads[0] if track_spreads else 0,
            'final_spread_km': track_spreads[-1] if track_spreads else 0,
            'max_spread_km': max(track_spreads) if track_spreads else 0,
            'spread_growth_factor': track_spreads[-1] / track_spreads[0] if track_spreads and track_spreads[0] > 0 else 0
        }
    
    def _analyze_field_uncertainty(self, variables: List[str]) -> Dict:
        """Analyze atmospheric field uncertainty."""
        
        # Extract predictions for all members
        all_predictions = []
        for result in self.forecast_results:
            if result['predictions']:
                all_predictions.append(result['predictions'])
        
        if not all_predictions:
            return {}
        
        # Analyze ensemble statistics
        ensemble_stats = analyze_ensemble_statistics(all_predictions, variables)
        
        return ensemble_stats
    
    def _analyze_spread_growth(self) -> Dict:
        """Analyze ensemble spread growth characteristics."""
        
        if 'track_uncertainty' not in self.uncertainty_metrics:
            track_uncertainty = self._analyze_track_uncertainty()
        else:
            track_uncertainty = self.uncertainty_metrics['track_uncertainty']
        
        spread_time_series = track_uncertainty['track_spread_km']
        
        # Compute growth metrics
        if len(spread_time_series) > 1:
            # Linear growth rate
            time_hours = np.arange(len(spread_time_series)) * 6  # 6-hour intervals
            
            # Fit exponential growth model: spread = a * exp(b * t)
            if np.all(spread_time_series > 0):
                log_spread = np.log(spread_time_series)
                growth_coeffs = np.polyfit(time_hours, log_spread, 1)
                growth_rate = growth_coeffs[0]  # exponential growth rate per hour
                doubling_time = np.log(2) / growth_rate if growth_rate > 0 else np.inf
            else:
                growth_rate = 0.0
                doubling_time = np.inf
            
            # Linear growth rate
            linear_coeffs = np.polyfit(time_hours, spread_time_series, 1)
            linear_growth_rate = linear_coeffs[0]  # km per hour
            
        else:
            growth_rate = 0.0
            doubling_time = np.inf
            linear_growth_rate = 0.0
        
        return {
            'exponential_growth_rate_per_hour': growth_rate,
            'doubling_time_hours': doubling_time,
            'linear_growth_rate_km_per_hour': linear_growth_rate,
            'spread_time_series': spread_time_series
        }
    
    def compare_with_operational_ensembles(
        self,
        reference_ensemble_stats: Optional[Dict] = None
    ) -> Dict:
        """
        Compare Aurora ensemble characteristics with operational NWP ensembles.
        
        Args:
            reference_ensemble_stats: Reference ensemble statistics for comparison
            
        Returns:
            Comparison metrics and assessment
        """
        
        if not self.uncertainty_metrics:
            self.analyze_ensemble_uncertainty()
        
        # Default reference values (rough estimates for tropical cyclone forecasts)
        if reference_ensemble_stats is None:
            reference_ensemble_stats = {
                'track_spread_24h': 100,   # km
                'track_spread_48h': 200,   # km  
                'track_spread_72h': 350,   # km
                'track_spread_96h': 500,   # km
                'track_spread_120h': 700,  # km
                'growth_rate_typical': 0.02,  # exponential growth rate per hour
                'doubling_time_typical': 35    # hours
            }
        
        # Extract Aurora ensemble metrics
        aurora_track_spread = self.uncertainty_metrics['track_uncertainty']['track_spread_km']
        aurora_growth_rate = self.uncertainty_metrics['spread_analysis']['exponential_growth_rate_per_hour']
        aurora_doubling_time = self.uncertainty_metrics['spread_analysis']['doubling_time_hours']
        
        # Compare spread at key forecast times
        comparison_ratios = {}
        forecast_hours = [24, 48, 72, 96, 120]
        
        for hours in forecast_hours:
            time_index = min(hours // 6, len(aurora_track_spread) - 1)  # 6-hour intervals
            if time_index < len(aurora_track_spread):
                aurora_spread = aurora_track_spread[time_index]
                ref_spread = reference_ensemble_stats.get(f'track_spread_{hours}h', 0)
                
                if ref_spread > 0:
                    ratio = aurora_spread / ref_spread
                    comparison_ratios[f'{hours}h'] = ratio
        
        # Compare growth characteristics
        growth_ratio = aurora_growth_rate / reference_ensemble_stats['growth_rate_typical']
        doubling_time_ratio = aurora_doubling_time / reference_ensemble_stats['doubling_time_typical']
        
        # Overall assessment
        avg_spread_ratio = np.mean(list(comparison_ratios.values())) if comparison_ratios else 0.0
        
        if avg_spread_ratio < 0.1:
            assessment = "Much too low - Aurora ensemble spread is insufficient"
        elif avg_spread_ratio < 0.3:
            assessment = "Too low - Aurora ensemble spread is below operational standards"
        elif avg_spread_ratio < 0.7:
            assessment = "Somewhat low - Aurora ensemble spread could be enhanced"
        elif avg_spread_ratio < 1.3:
            assessment = "Good - Aurora ensemble spread is comparable to operational ensembles"
        else:
            assessment = "High - Aurora ensemble spread exceeds typical operational values"
        
        return {
            'spread_ratios': comparison_ratios,
            'average_spread_ratio': avg_spread_ratio,
            'growth_rate_ratio': growth_ratio,
            'doubling_time_ratio': doubling_time_ratio,
            'assessment': assessment,
            'reference_stats': reference_ensemble_stats
        }
    
    def save_ensemble_results(
        self,
        output_directory: Union[str, Path],
        include_full_forecasts: bool = False
    ) -> None:
        """
        Save ensemble results to disk.
        
        Args:
            output_directory: Directory to save results
            include_full_forecasts: Whether to save full forecast fields (large files)
        """
        
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble configuration
        config_data = {
            'n_members': self.n_members,
            'perturbation_config': self.perturbation_config,
            'ensemble_size': len(self.forecast_results)
        }
        
        with open(output_dir / 'ensemble_config.pkl', 'wb') as f:
            pickle.dump(config_data, f)
        
        # Save track results
        track_data = []
        for result in self.forecast_results:
            track_df = result['track'].copy()
            track_df['member_id'] = result['member_id']
            track_data.append(track_df)
        
        if track_data:
            combined_tracks = pd.concat(track_data, ignore_index=True)
            combined_tracks.to_csv(output_dir / 'ensemble_tracks.csv', index=False)
        
        # Save uncertainty metrics
        if self.uncertainty_metrics:
            with open(output_dir / 'uncertainty_metrics.pkl', 'wb') as f:
                pickle.dump(self.uncertainty_metrics, f)
        
        # Save full forecasts if requested
        if include_full_forecasts and any(r['predictions'] for r in self.forecast_results):
            print("Saving full forecast fields (this may take a while)...")
            forecast_data = {
                'forecasts': [r['predictions'] for r in self.forecast_results if r['predictions']],
                'member_ids': [r['member_id'] for r in self.forecast_results if r['predictions']]
            }
            
            with open(output_dir / 'ensemble_forecasts.pkl', 'wb') as f:
                pickle.dump(forecast_data, f)
        
        print(f"Ensemble results saved to {output_dir}")
    
    def generate_summary_report(self) -> str:
        """
        Generate summary report of ensemble forecast experiment.
        
        Returns:
            Formatted text report
        """
        
        if not self.forecast_results:
            return "No ensemble forecast results available."
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("AURORA ENSEMBLE FORECAST SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Ensemble configuration
        report_lines.append("ENSEMBLE CONFIGURATION")
        report_lines.append("-" * 40)
        report_lines.append(f"Number of ensemble members: {len(self.forecast_results)}")
        report_lines.append(f"Perturbation strategy: {self.perturbation_config}")
        report_lines.append("")
        
        # Track uncertainty summary
        if self.uncertainty_metrics and 'track_uncertainty' in self.uncertainty_metrics:
            track_uncertainty = self.uncertainty_metrics['track_uncertainty']
            
            report_lines.append("TRACK UNCERTAINTY ANALYSIS")
            report_lines.append("-" * 40)
            report_lines.append(f"Initial spread: {track_uncertainty['initial_spread_km']:.1f} km")
            report_lines.append(f"Final spread: {track_uncertainty['final_spread_km']:.1f} km")
            report_lines.append(f"Maximum spread: {track_uncertainty['max_spread_km']:.1f} km")
            report_lines.append(f"Spread growth factor: {track_uncertainty['spread_growth_factor']:.2f}")
            report_lines.append("")
        
        # Operational comparison
        operational_comparison = self.compare_with_operational_ensembles()
        
        report_lines.append("COMPARISON WITH OPERATIONAL ENSEMBLES")
        report_lines.append("-" * 40)
        report_lines.append(f"Average spread ratio: {operational_comparison['average_spread_ratio']:.3f}")
        report_lines.append(f"Assessment: {operational_comparison['assessment']}")
        report_lines.append("")
        
        report_lines.append("SPREAD RATIOS BY FORECAST TIME:")
        for time, ratio in operational_comparison['spread_ratios'].items():
            report_lines.append(f"  {time}: {ratio:.3f}")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        avg_ratio = operational_comparison['average_spread_ratio']
        if avg_ratio < 0.1:
            report_lines.append("- Aurora ensemble spread is much too low for practical use")
            report_lines.append("- Consider significantly larger perturbations (5-10x current)")
            report_lines.append("- Investigate alternative perturbation strategies")
            report_lines.append("- Aurora may not be suitable for ensemble forecasting")
        elif avg_ratio < 0.3:
            report_lines.append("- Aurora ensemble spread is below operational standards")
            report_lines.append("- Increase perturbation magnitudes by 2-3x")
            report_lines.append("- Test enhanced uncertainty perturbation strategy")
            report_lines.append("- Consider compound perturbation methods")
        elif avg_ratio < 0.7:
            report_lines.append("- Aurora ensemble spread could be enhanced")
            report_lines.append("- Modest increase in perturbation magnitudes recommended")
            report_lines.append("- Test spatial correlation in perturbations")
            report_lines.append("- Generally promising for ensemble applications")
        else:
            report_lines.append("- Aurora ensemble spread is adequate for operational use")
            report_lines.append("- Proceed with operational ensemble implementation")
            report_lines.append("- Focus on ensemble post-processing and calibration")
            report_lines.append("- Investigate ensemble-based uncertainty communication")
        
        return "\n".join(report_lines)


def run_systematic_ensemble_experiment(
    model: Aurora,
    initial_conditions: Batch,
    experiment_config: Dict,
    output_directory: str
) -> Dict:
    """
    Run systematic ensemble experiment with multiple perturbation strategies.
    
    Args:
        model: Aurora model instance
        initial_conditions: Baseline initial conditions
        experiment_config: Experiment configuration
        output_directory: Directory for saving results
        
    Returns:
        Dictionary with all experiment results
    """
    
    strategies = experiment_config.get('strategies', ['realistic_obs', 'enhanced_uncertainty'])
    n_members = experiment_config.get('n_members', 20)
    forecast_steps = experiment_config.get('forecast_steps', 40)
    
    all_results = {}
    
    for strategy in strategies:
        print(f"\n" + "="*60)
        print(f"Running ensemble experiment with strategy: {strategy}")
        print(f"="*60)
        
        # Create ensemble system
        ensemble = AuroraEnsemble(model, n_members=n_members)
        
        # Generate ensemble members
        ensemble.generate_ensemble_members(initial_conditions, strategy)
        
        # Run forecast
        forecast_results = ensemble.run_ensemble_forecast(forecast_steps)
        
        # Analyze uncertainty
        uncertainty_analysis = ensemble.analyze_ensemble_uncertainty()
        
        # Compare with operational ensembles
        operational_comparison = ensemble.compare_with_operational_ensembles()
        
        # Save results
        strategy_output_dir = Path(output_directory) / f"strategy_{strategy}"
        ensemble.save_ensemble_results(strategy_output_dir)
        
        # Generate report
        report = ensemble.generate_summary_report()
        with open(strategy_output_dir / "summary_report.txt", 'w') as f:
            f.write(report)
        
        # Store results
        all_results[strategy] = {
            'ensemble_system': ensemble,
            'forecast_results': forecast_results,
            'uncertainty_analysis': uncertainty_analysis,
            'operational_comparison': operational_comparison,
            'summary_report': report
        }
    
    # Generate comparative analysis
    comparative_report = generate_strategy_comparison_report(all_results)
    with open(Path(output_directory) / "strategy_comparison.txt", 'w') as f:
        f.write(comparative_report)
    
    return all_results


def generate_strategy_comparison_report(strategy_results: Dict) -> str:
    """Generate comparative analysis of different perturbation strategies."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("AURORA ENSEMBLE STRATEGY COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Summary table
    report_lines.append("STRATEGY COMPARISON SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"{'Strategy':<20} {'Avg Spread Ratio':<15} {'Max Spread (km)':<15} {'Assessment'}")
    report_lines.append("-" * 80)
    
    for strategy, results in strategy_results.items():
        operational_comp = results['operational_comparison']
        uncertainty = results['uncertainty_analysis']['track_uncertainty']
        
        avg_ratio = operational_comp['average_spread_ratio']
        max_spread = uncertainty['max_spread_km']
        assessment = operational_comp['assessment'].split(' - ')[0]  # First part only
        
        report_lines.append(f"{strategy:<20} {avg_ratio:<15.3f} {max_spread:<15.1f} {assessment}")
    
    report_lines.append("")
    
    # Best strategy recommendation
    best_strategy = max(
        strategy_results.keys(),
        key=lambda s: strategy_results[s]['operational_comparison']['average_spread_ratio']
    )
    
    report_lines.append("RECOMMENDATION")
    report_lines.append("-" * 40)
    report_lines.append(f"Best performing strategy: {best_strategy}")
    report_lines.append(f"Recommended for operational ensemble implementation")
    
    return "\n".join(report_lines)