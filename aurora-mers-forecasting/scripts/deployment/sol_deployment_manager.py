#!/usr/bin/env python3
"""
Sol Supercomputer Deployment Manager for Aurora-MERS Forecasting System

Handles job submission, monitoring, and management on ASU's Sol supercomputer
for automated weather forecasting operations.
"""

import os
import sys
import yaml
import logging
import argparse
import subprocess
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

class SolDeploymentManager:
    """
    Deployment manager for Sol supercomputer operations.
    
    Handles SLURM job submission, monitoring, and automated scheduling
    for the Aurora-MERS weather forecasting system.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the deployment manager.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "system_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_logging()
        self.project_root = Path(__file__).parent.parent.parent
        self.job_template = self.project_root / "scripts" / "deployment" / "sol_job_template.slurm"
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = Path(self.config['paths']['logs'])
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(log_dir / 'sol_deployment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def check_slurm_availability(self) -> bool:
        """
        Check if SLURM is available and accessible.
        
        Returns:
            True if SLURM is available
        """
        try:
            result = subprocess.run(['squeue', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.logger.info(f"SLURM available: {result.stdout.strip()}")
                return True
            else:
                self.logger.error("SLURM not available or not accessible")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.error(f"SLURM check failed: {e}")
            return False
            
    def get_account_info(self) -> Optional[str]:
        """
        Get the user's SLURM account information.
        
        Returns:
            Account name if available
        """
        try:
            result = subprocess.run(['sacctmgr', 'show', 'user', os.getenv('USER'), '-s'],
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Parse account information from output
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if '|' in line and 'Account' not in line:
                        parts = line.split('|')
                        if len(parts) > 1:
                            account = parts[1].strip()
                            self.logger.info(f"Found SLURM account: {account}")
                            return account
                            
            self.logger.warning("Could not determine SLURM account")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            return None
            
    def create_job_script(self, job_mode: str = "operational", 
                         custom_params: Optional[Dict] = None) -> Path:
        """
        Create a customized job script for submission.
        
        Args:
            job_mode: Job execution mode
            custom_params: Custom SLURM parameters
            
        Returns:
            Path to created job script
        """
        try:
            # Read template
            with open(self.job_template, 'r') as f:
                template_content = f.read()
                
            # Apply customizations
            compute_config = self.config['compute']
            
            # Replace placeholders
            replacements = {
                'ASU_PROJECT_ACCOUNT': custom_params.get('account', 'default') if custom_params else 'default',
                '--partition=gpu': f"--partition={compute_config['partition']}",
                '--nodes=1': f"--nodes={compute_config['nodes']}",
                '--ntasks-per-node=1': f"--ntasks-per-node={compute_config['ntasks_per_node']}",
                '--cpus-per-task=16': f"--cpus-per-task={compute_config['cpus_per_task']}",
                '--mem=64GB': f"--mem={compute_config['memory']}",
                '--gres=gpu:1': f"--gres={compute_config['gres']}",
                '--time=4:00:00': f"--time={compute_config['time']}"
            }
            
            # Apply additional custom parameters
            if custom_params:
                for key, value in custom_params.items():
                    if key.startswith('--'):
                        replacements[key] = f"{key}={value}"
                        
            # Perform replacements
            customized_content = template_content
            for old, new in replacements.items():
                customized_content = customized_content.replace(old, new)
                
            # Create temporary job script
            job_script = tempfile.NamedTemporaryFile(mode='w', suffix='.slurm', 
                                                   delete=False, prefix='aurora_job_')
            job_script.write(customized_content)
            job_script.close()
            
            # Make executable
            os.chmod(job_script.name, 0o755)
            
            self.logger.info(f"Created job script: {job_script.name}")
            return Path(job_script.name)
            
        except Exception as e:
            self.logger.error(f"Failed to create job script: {e}")
            raise
            
    def submit_job(self, job_mode: str = "operational", 
                   custom_params: Optional[Dict] = None) -> Optional[str]:
        """
        Submit a job to SLURM queue.
        
        Args:
            job_mode: Job execution mode
            custom_params: Custom SLURM parameters
            
        Returns:
            Job ID if successful
        """
        try:
            # Check SLURM availability
            if not self.check_slurm_availability():
                self.logger.error("SLURM not available, cannot submit job")
                return None
                
            # Create job script
            job_script = self.create_job_script(job_mode, custom_params)
            
            # Submit job
            cmd = ['sbatch', str(job_script), job_mode]
            
            self.logger.info(f"Submitting job with mode: {job_mode}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Extract job ID from output
                output = result.stdout.strip()
                match = re.search(r'Submitted batch job (\d+)', output)
                
                if match:
                    job_id = match.group(1)
                    self.logger.info(f"Job submitted successfully: {job_id}")
                    
                    # Save job info
                    self.save_job_info(job_id, job_mode, job_script)
                    
                    # Clean up temporary script
                    try:
                        os.unlink(job_script)
                    except:
                        pass
                        
                    return job_id
                else:
                    self.logger.error(f"Could not extract job ID from output: {output}")
                    return None
            else:
                self.logger.error(f"Job submission failed: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to submit job: {e}")
            return None
            
    def save_job_info(self, job_id: str, job_mode: str, job_script: Path):
        """
        Save job information for tracking.
        
        Args:
            job_id: SLURM job ID
            job_mode: Job execution mode
            job_script: Path to job script
        """
        try:
            job_info = {
                'job_id': job_id,
                'job_mode': job_mode,
                'submission_time': datetime.utcnow().isoformat(),
                'job_script': str(job_script),
                'status': 'SUBMITTED'
            }
            
            jobs_file = Path(self.config['paths']['logs']) / 'submitted_jobs.yaml'
            
            # Load existing jobs
            if jobs_file.exists():
                with open(jobs_file, 'r') as f:
                    jobs_data = yaml.safe_load(f) or {}
            else:
                jobs_data = {}
                
            # Add new job
            jobs_data[job_id] = job_info
            
            # Save updated jobs
            with open(jobs_file, 'w') as f:
                yaml.dump(jobs_data, f)
                
            self.logger.debug(f"Saved job info for {job_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save job info: {e}")
            
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """
        Get status of a SLURM job.
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            Job status information
        """
        try:
            cmd = ['scontrol', 'show', 'job', job_id]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Parse job information
                output = result.stdout
                status_info = {}
                
                # Extract key information
                patterns = {
                    'JobState': r'JobState=(\w+)',
                    'StartTime': r'StartTime=([^\s]+)',
                    'EndTime': r'EndTime=([^\s]+)',
                    'RunTime': r'RunTime=([^\s]+)',
                    'NodeList': r'NodeList=([^\s]+)',
                    'ExitCode': r'ExitCode=([^\s]+)'
                }
                
                for key, pattern in patterns.items():
                    match = re.search(pattern, output)
                    if match:
                        status_info[key] = match.group(1)
                        
                return status_info
            else:
                self.logger.warning(f"Could not get status for job {job_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get job status: {e}")
            return None
            
    def list_active_jobs(self) -> List[Dict]:
        """
        List all active Aurora forecasting jobs.
        
        Returns:
            List of active job information
        """
        try:
            cmd = ['squeue', '-u', os.getenv('USER'), '--format=%i,%j,%T,%M,%N', '--noheader']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                jobs = []
                lines = result.stdout.strip().split('\n')
                
                for line in lines:
                    if line.strip() and 'aurora' in line.lower():
                        parts = line.split(',')
                        if len(parts) >= 5:
                            job_info = {
                                'job_id': parts[0].strip(),
                                'job_name': parts[1].strip(),
                                'status': parts[2].strip(),
                                'time': parts[3].strip(),
                                'nodes': parts[4].strip()
                            }
                            jobs.append(job_info)
                            
                return jobs
            else:
                self.logger.error(f"Failed to list jobs: {result.stderr}")
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to list active jobs: {e}")
            return []
            
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a SLURM job.
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            True if successful
        """
        try:
            cmd = ['scancel', job_id]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully cancelled job {job_id}")
                return True
            else:
                self.logger.error(f"Failed to cancel job {job_id}: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
            
    def schedule_recurring_jobs(self, frequency_hours: int = 12) -> bool:
        """
        Set up recurring job scheduling using cron.
        
        Args:
            frequency_hours: Frequency of job submission in hours
            
        Returns:
            True if successful
        """
        try:
            # Create cron job script
            cron_script = self.project_root / "scripts" / "deployment" / "cron_job_submitter.sh"
            
            script_content = f"""#!/bin/bash
# Aurora-MERS automated job submission
cd {self.project_root}
python scripts/deployment/sol_deployment_manager.py submit --mode operational >> logs/cron_job_submission.log 2>&1
"""
            
            with open(cron_script, 'w') as f:
                f.write(script_content)
                
            os.chmod(cron_script, 0o755)
            
            # Calculate cron schedule
            if frequency_hours == 12:
                cron_schedule = "0 0,12 * * *"  # 00:00 and 12:00 daily
            elif frequency_hours == 6:
                cron_schedule = "0 0,6,12,18 * * *"  # Every 6 hours
            elif frequency_hours == 24:
                cron_schedule = "0 0 * * *"  # Daily at midnight
            else:
                cron_schedule = f"0 */{frequency_hours} * * *"  # Custom frequency
                
            self.logger.info(f"Cron script created: {cron_script}")
            self.logger.info(f"To enable automated scheduling, add this to your crontab:")
            self.logger.info(f"{cron_schedule} {cron_script}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set up recurring jobs: {e}")
            return False
            
    def monitor_job_performance(self, job_id: str) -> Dict:
        """
        Monitor job performance and resource usage.
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            Performance metrics
        """
        try:
            # Get job accounting information
            cmd = ['sacct', '-j', job_id, '--format=JobID,State,ExitCode,Start,End,Elapsed,MaxRSS,MaxVMSize,CPUTime', '--noheader']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0].strip():
                    parts = lines[0].split()
                    
                    metrics = {
                        'job_id': job_id,
                        'state': parts[1] if len(parts) > 1 else 'UNKNOWN',
                        'exit_code': parts[2] if len(parts) > 2 else 'N/A',
                        'start_time': parts[3] if len(parts) > 3 else 'N/A',
                        'end_time': parts[4] if len(parts) > 4 else 'N/A',
                        'elapsed_time': parts[5] if len(parts) > 5 else 'N/A',
                        'max_memory': parts[6] if len(parts) > 6 else 'N/A',
                        'max_vmem': parts[7] if len(parts) > 7 else 'N/A',
                        'cpu_time': parts[8] if len(parts) > 8 else 'N/A'
                    }
                    
                    return metrics
                    
            self.logger.warning(f"No accounting data available for job {job_id}")
            return {}
            
        except Exception as e:
            self.logger.error(f"Failed to get job performance metrics: {e}")
            return {}


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Sol Deployment Manager for Aurora-MERS")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit a job')
    submit_parser.add_argument('--mode', choices=['operational', 'test', 'forecast-only', 'data-only'],
                              default='operational', help='Job execution mode')
    submit_parser.add_argument('--account', type=str, help='SLURM account to use')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check job status')
    status_parser.add_argument('job_id', help='SLURM job ID')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List active jobs')
    
    # Cancel command
    cancel_parser = subparsers.add_parser('cancel', help='Cancel a job')
    cancel_parser.add_argument('job_id', help='SLURM job ID')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor job performance')
    monitor_parser.add_argument('job_id', help='SLURM job ID')
    
    # Schedule command
    schedule_parser = subparsers.add_parser('schedule', help='Set up recurring jobs')
    schedule_parser.add_argument('--frequency', type=int, default=12,
                                help='Frequency in hours (default: 12)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    try:
        manager = SolDeploymentManager(args.config)
        
        if args.command == 'submit':
            custom_params = {}
            if args.account:
                custom_params['account'] = args.account
                
            job_id = manager.submit_job(args.mode, custom_params)
            if job_id:
                print(f"Job submitted successfully: {job_id}")
                return 0
            else:
                print("Job submission failed")
                return 1
                
        elif args.command == 'status':
            status = manager.get_job_status(args.job_id)
            if status:
                print(f"Job Status for {args.job_id}:")
                for key, value in status.items():
                    print(f"  {key}: {value}")
                return 0
            else:
                print(f"Could not get status for job {args.job_id}")
                return 1
                
        elif args.command == 'list':
            jobs = manager.list_active_jobs()
            if jobs:
                print("Active Aurora Jobs:")
                for job in jobs:
                    print(f"  {job['job_id']}: {job['job_name']} ({job['status']}) - {job['time']}")
            else:
                print("No active Aurora jobs found")
            return 0
            
        elif args.command == 'cancel':
            success = manager.cancel_job(args.job_id)
            if success:
                print(f"Job {args.job_id} cancelled successfully")
                return 0
            else:
                print(f"Failed to cancel job {args.job_id}")
                return 1
                
        elif args.command == 'monitor':
            metrics = manager.monitor_job_performance(args.job_id)
            if metrics:
                print(f"Performance Metrics for Job {args.job_id}:")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
            else:
                print(f"No performance data available for job {args.job_id}")
            return 0
            
        elif args.command == 'schedule':
            success = manager.schedule_recurring_jobs(args.frequency)
            if success:
                print(f"Recurring job schedule configured for every {args.frequency} hours")
                return 0
            else:
                print("Failed to configure recurring jobs")
                return 1
                
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())