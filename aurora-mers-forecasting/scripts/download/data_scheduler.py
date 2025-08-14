#!/usr/bin/env python3
"""
Data Download Scheduler for Aurora-MERS Forecasting System

Handles automated scheduling of ECMWF data downloads every 12 hours
and manages the data processing pipeline.
"""

import os
import sys
import yaml
import logging
import schedule
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import signal
import threading
import queue

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

class DataScheduler:
    """
    Automated scheduler for ECMWF data downloads and processing.
    
    Manages the complete data pipeline from ECMWF download to
    Aurora-ready format conversion.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the data scheduler.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "system_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_logging()
        self.running = True
        self.job_queue = queue.Queue()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = Path(self.config['paths']['logs'])
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(log_dir / 'data_scheduler.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        
    def run_download_script(self) -> bool:
        """
        Execute the ECMWF download script.
        
        Returns:
            True if successful
        """
        try:
            script_path = Path(__file__).parent / "ecmwf_downloader.py"
            config_path = Path(__file__).parent.parent.parent / "config" / "system_config.yaml"
            
            cmd = [
                sys.executable,
                str(script_path),
                "--config", str(config_path),
                "--mode", "latest",
                "--cleanup"
            ]
            
            self.logger.info("Starting ECMWF data download...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                self.logger.info("ECMWF download completed successfully")
                self.logger.debug(f"Download output: {result.stdout}")
                return True
            else:
                self.logger.error(f"ECMWF download failed with return code {result.returncode}")
                self.logger.error(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("ECMWF download timed out after 1 hour")
            return False
        except Exception as e:
            self.logger.error(f"Failed to run ECMWF download script: {e}")
            return False
            
    def run_data_processing(self) -> bool:
        """
        Execute the data processing script to convert to Aurora format.
        
        Returns:
            True if successful
        """
        try:
            script_path = Path(__file__).parent.parent / "processing" / "aurora_data_converter.py"
            config_path = Path(__file__).parent.parent.parent / "config" / "system_config.yaml"
            
            cmd = [
                sys.executable,
                str(script_path),
                "--config", str(config_path),
                "--mode", "latest"
            ]
            
            self.logger.info("Starting data processing for Aurora...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            if result.returncode == 0:
                self.logger.info("Data processing completed successfully")
                self.logger.debug(f"Processing output: {result.stdout}")
                return True
            else:
                self.logger.error(f"Data processing failed with return code {result.returncode}")
                self.logger.error(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Data processing timed out after 30 minutes")
            return False
        except Exception as e:
            self.logger.error(f"Failed to run data processing script: {e}")
            return False
            
    def run_forecast_generation(self) -> bool:
        """
        Execute the Aurora forecast generation script.
        
        Returns:
            True if successful
        """
        try:
            script_path = Path(__file__).parent.parent / "forecasting" / "aurora_forecaster.py"
            config_path = Path(__file__).parent.parent.parent / "config" / "system_config.yaml"
            
            cmd = [
                sys.executable,
                str(script_path),
                "--config", str(config_path),
                "--mode", "operational"
            ]
            
            self.logger.info("Starting Aurora forecast generation...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            if result.returncode == 0:
                self.logger.info("Forecast generation completed successfully")
                self.logger.debug(f"Forecast output: {result.stdout}")
                return True
            else:
                self.logger.error(f"Forecast generation failed with return code {result.returncode}")
                self.logger.error(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Forecast generation timed out after 2 hours")
            return False
        except Exception as e:
            self.logger.error(f"Failed to run forecast generation script: {e}")
            return False
            
    def execute_full_pipeline(self):
        """Execute the complete data pipeline."""
        pipeline_start = datetime.utcnow()
        self.logger.info(f"Starting full data pipeline at {pipeline_start}")
        
        # Step 1: Download latest ECMWF data
        download_success = self.run_download_script()
        if not download_success:
            self.logger.error("Pipeline failed at download stage")
            return False
            
        # Step 2: Process data for Aurora
        processing_success = self.run_data_processing()
        if not processing_success:
            self.logger.error("Pipeline failed at processing stage")
            return False
            
        # Step 3: Generate forecasts
        forecast_success = self.run_forecast_generation()
        if not forecast_success:
            self.logger.error("Pipeline failed at forecast generation stage")
            return False
            
        pipeline_end = datetime.utcnow()
        duration = pipeline_end - pipeline_start
        
        self.logger.info(f"Full pipeline completed successfully in {duration}")
        
        # Update status file
        self.update_status_file(pipeline_end, True, duration)
        
        return True
        
    def update_status_file(self, timestamp: datetime, success: bool, duration: timedelta):
        """
        Update the system status file with latest pipeline execution info.
        
        Args:
            timestamp: Completion timestamp
            success: Whether pipeline succeeded
            duration: Pipeline execution duration
        """
        status_file = Path(self.config['paths']['logs']) / "pipeline_status.yaml"
        
        status = {
            'last_execution': timestamp.isoformat(),
            'success': success,
            'duration_seconds': duration.total_seconds(),
            'next_scheduled': self.get_next_scheduled_time().isoformat()
        }
        
        try:
            with open(status_file, 'w') as f:
                yaml.dump(status, f)
        except Exception as e:
            self.logger.error(f"Failed to update status file: {e}")
            
    def get_next_scheduled_time(self) -> datetime:
        """Get the next scheduled execution time."""
        now = datetime.utcnow()
        
        # Schedule for 00:00 and 12:00 UTC
        if now.hour < 12:
            next_run = now.replace(hour=12, minute=0, second=0, microsecond=0)
        else:
            next_run = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            
        return next_run
        
    def schedule_jobs(self):
        """Set up the job schedule."""
        # Schedule jobs every 12 hours at 00:00 and 12:00 UTC
        schedule.every().day.at("00:00").do(self.job_queue.put, "pipeline")
        schedule.every().day.at("12:00").do(self.job_queue.put, "pipeline")
        
        self.logger.info("Jobs scheduled for 00:00 and 12:00 UTC daily")
        
    def worker_thread(self):
        """Worker thread to process scheduled jobs."""
        while self.running:
            try:
                if not self.job_queue.empty():
                    job = self.job_queue.get(timeout=1)
                    
                    if job == "pipeline":
                        self.logger.info("Executing scheduled pipeline job")
                        self.execute_full_pipeline()
                        
                    self.job_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in worker thread: {e}")
                
    def run_scheduler(self):
        """Run the main scheduler loop."""
        self.logger.info("Starting data scheduler...")
        
        # Schedule jobs
        self.schedule_jobs()
        
        # Start worker thread
        worker = threading.Thread(target=self.worker_thread, daemon=True)
        worker.start()
        
        # Run an initial pipeline execution if requested
        initial_run = os.getenv('AURORA_INITIAL_RUN', 'false').lower() == 'true'
        if initial_run:
            self.logger.info("Running initial pipeline execution...")
            self.execute_full_pipeline()
        
        # Main scheduler loop
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
                
        self.logger.info("Scheduler stopped")
        
    def run_manual_pipeline(self):
        """Run a manual pipeline execution (for testing/debugging)."""
        self.logger.info("Running manual pipeline execution...")
        success = self.execute_full_pipeline()
        
        if success:
            print("Pipeline executed successfully")
            return 0
        else:
            print("Pipeline execution failed")
            return 1


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Scheduler for Aurora-MERS")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--mode", choices=['scheduler', 'manual'], default='scheduler',
                       help="Run mode: continuous scheduler or manual execution")
    
    args = parser.parse_args()
    
    try:
        scheduler = DataScheduler(args.config)
        
        if args.mode == 'scheduler':
            scheduler.run_scheduler()
            return 0
        elif args.mode == 'manual':
            return scheduler.run_manual_pipeline()
            
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())