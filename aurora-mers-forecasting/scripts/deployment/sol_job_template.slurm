#!/bin/bash
#SBATCH --job-name=aurora-forecast
# #SBATCH --account=ASU_PROJECT_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/home/qhuang62/aurora/aurora-mers-forecasting/logs/aurora_forecast_%j.out
#SBATCH --error=/home/qhuang62/aurora/aurora-mers-forecasting/logs/aurora_forecast_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qhuang62@asu.edu

# Job configuration for Aurora-ERA5 Weather Forecasting System
# This script runs on ASU's Sol supercomputer to generate automated weather forecasts

echo "Aurora-ERA5 Forecasting Job Starting at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Set environment variables
export AURORA_PROJECT_ROOT="/home/qhuang62/aurora/aurora-mers-forecasting"
export AURORA_CONFIG="$AURORA_PROJECT_ROOT/config/system_config.yaml"
export PYTHONPATH="$AURORA_PROJECT_ROOT:$PYTHONPATH"

# Load required modules
module load python/3.10
module load cuda/11.8
module load miniconda3

# Activate conda environment (create if doesn't exist)
ENV_NAME="aurora-forecasting"
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Creating conda environment: $ENV_NAME"
    conda create -y -n $ENV_NAME python=3.10
    source activate $ENV_NAME
    
    # Install required packages
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install microsoft-aurora
    pip install xarray netcdf4 pyyaml schedule psutil
    pip install cdsapi eccodes cfgrib
    pip install numpy scipy pandas matplotlib cartopy
else
    source activate $ENV_NAME
fi

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi

# Verify Python environment
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Change to project directory
cd $AURORA_PROJECT_ROOT

# Set job mode based on parameter (default: operational)
JOB_MODE=${1:-operational}
echo "Job mode: $JOB_MODE"

# Function to run data pipeline
run_data_pipeline() {
    echo "Starting data download and processing pipeline..."
    
    # Step 1: Download latest ECMWF data
    echo "Step 1: Downloading ECMWF ERA5 data..."
    python scripts/download/ecmwf_downloader.py \
        --config "$AURORA_CONFIG" \
        --mode latest \
        --cleanup
    
    if [ $? -ne 0 ]; then
        echo "ERROR: ECMWF ERA5 data download failed"
        exit 1
    fi
    
    # Step 2: Convert to Aurora format
    echo "Step 2: Converting data to Aurora format..."
    python scripts/processing/aurora_data_converter.py \
        --config "$AURORA_CONFIG" \
        --mode latest \
        --validate
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Data conversion failed"
        exit 1
    fi
    
    echo "Data pipeline completed successfully"
}

# Function to run Aurora forecast
run_aurora_forecast() {
    echo "Starting Aurora forecast generation..."
    
    python scripts/forecasting/aurora_forecaster.py \
        --config "$AURORA_CONFIG" \
        --mode "$JOB_MODE"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Aurora forecast generation failed"
        exit 1
    fi
    
    echo "Aurora forecast completed successfully"
}

# Function to run post-processing
run_post_processing() {
    echo "Starting post-processing and visualization..."
    
    # Generate regional forecasts (if script exists)
    if [ -f "scripts/processing/regional_processor.py" ]; then
        python scripts/processing/regional_processor.py \
            --config "$AURORA_CONFIG" \
            --mode latest
    fi
    
    # Generate visualizations (if script exists)
    if [ -f "scripts/processing/forecast_visualizer.py" ]; then
        python scripts/processing/forecast_visualizer.py \
            --config "$AURORA_CONFIG" \
            --mode latest
    fi
    
    echo "Post-processing completed"
}

# Main execution based on job mode
case $JOB_MODE in
    "operational")
        echo "Running operational forecast pipeline..."
        run_data_pipeline
        run_aurora_forecast
        run_post_processing
        ;;
    "forecast-only")
        echo "Running forecast generation only..."
        run_aurora_forecast
        ;;
    "test")
        echo "Running test forecast..."
        python scripts/forecasting/aurora_forecaster.py \
            --config "$AURORA_CONFIG" \
            --mode test \
            --test-steps 5
        ;;
    "data-only")
        echo "Running data pipeline only..."
        run_data_pipeline
        ;;
    *)
        echo "Unknown job mode: $JOB_MODE"
        echo "Valid modes: operational, forecast-only, test, data-only"
        exit 1
        ;;
esac

# Cleanup
echo "Cleaning up temporary files..."
find /tmp -name "aurora_*" -user $USER -mtime +1 -delete 2>/dev/null || true

# Generate job summary
echo "Job Summary:"
echo "  Start time: $SLURM_JOB_START_TIME"
echo "  End time: $(date)"
echo "  Duration: $SECONDS seconds"
echo "  Peak memory usage: $(sacct -j $SLURM_JOB_ID --format=MaxRSS --noheader | head -1)"
echo "  GPU utilization: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)"

echo "Aurora-ERA5 Forecasting Job Completed Successfully at $(date)"