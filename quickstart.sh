#!/bin/bash
# Quick start script - Run this after cloning the repository

set -e  # Exit on error

echo "=========================================="
echo "RepE on CLUTRR - Quick Start"
echo "=========================================="
echo ""

# Check if running on Compute Canada or locally
if [[ -n "$CC_CLUSTER" ]] || [[ -n "$SLURM_CLUSTER_NAME" ]]; then
    echo "Detected Compute Canada environment"
    IS_CC=true
else
    echo "Detected local environment"
    IS_CC=false
fi
echo ""

# Step 1: Create directories
echo "Step 1: Creating directories..."
mkdir -p data/clutrr
mkdir -p outputs
mkdir -p logs
echo "  ✓ Directories created"
echo ""

# Step 2: Download dataset
echo "Step 2: Downloading CLUTRR dataset..."
if [ -f "data/clutrr/train.json" ] && [ -f "data/clutrr/val.json" ] && [ -f "data/clutrr/test.json" ]; then
    echo "  ⚠ Dataset already exists, skipping download"
else
    python download_data.py
    if [ $? -eq 0 ]; then
        echo "  ✓ Dataset downloaded"
    else
        echo "  ✗ Dataset download failed"
        exit 1
    fi
fi
echo ""

# Step 3: Test setup
echo "Step 3: Testing setup..."
python test_setup.py
if [ $? -eq 0 ]; then
    echo "  ✓ Setup test passed"
else
    echo "  ✗ Setup test failed"
    echo ""
    echo "Please install missing dependencies:"
    echo "  pip install -r requirements.txt"
    exit 1
fi
echo ""

# Step 4: Next steps
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""

if [ "$IS_CC" = true ]; then
    echo "Next steps on Compute Canada:"
    echo ""
    echo "1. Edit train_job.sh with your account:"
    echo "   nano train_job.sh"
    echo "   # Update: #SBATCH --account=def-<your-account>"
    echo ""
    echo "2. Submit the job:"
    echo "   sbatch train_job.sh"
    echo ""
    echo "3. Monitor the job:"
    echo "   sq                               # Check status"
    echo "   tail -f logs/slurm-<job-id>.out  # View logs"
    echo ""
else
    echo "Next steps for local/Colab:"
    echo ""
    echo "Option 1: Train locally"
    echo "   python train_repe.py --batch_size 8 --num_epochs 5"
    echo ""
    echo "Option 2: Use Colab"
    echo "   Open colab_notebook.ipynb in Google Colab"
    echo ""
    echo "Option 3: Setup Compute Canada"
    echo "   See COMPUTE_CANADA_GUIDE.md for detailed instructions"
    echo ""
fi

echo "Documentation:"
echo "  - README.md                   : Project overview"
echo "  - COMPUTE_CANADA_GUIDE.md    : Detailed CC setup guide"
echo "  - colab_notebook.ipynb       : Colab notebook"
echo ""
