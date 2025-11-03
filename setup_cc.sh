#!/bin/bash
# Setup script for Compute Canada

echo "=========================================="
echo "Compute Canada Setup for RepE on CLUTRR"
echo "=========================================="
echo ""

# Variables (update these)
CC_USERNAME="your_username"
CC_CLUSTER="beluga"  # or cedar, graham, narval
PROJECT_DIR="~/projects/def-group/username/T244"
ENV_NAME="repe_env"

echo "Configuration:"
echo "  Username: $CC_USERNAME"
echo "  Cluster: $CC_CLUSTER"
echo "  Project Dir: $PROJECT_DIR"
echo "  Environment: $ENV_NAME"
echo ""

# Step 1: Load Python module
echo "Step 1: Loading Python module..."
module load python/3.9
python --version
echo ""

# Step 2: Create virtual environment
echo "Step 2: Creating virtual environment..."
mkdir -p ~/envs
virtualenv --no-download ~/envs/$ENV_NAME
source ~/envs/$ENV_NAME/bin/activate
echo "Virtual environment created and activated"
echo ""

# Step 3: Upgrade pip
echo "Step 3: Upgrading pip..."
pip install --no-index --upgrade pip
echo ""

# Step 4: Install PyTorch with CUDA
echo "Step 4: Installing PyTorch with CUDA..."
pip install --no-index torch torchvision torchaudio
echo ""

# Step 5: Install other packages
echo "Step 5: Installing other packages..."
pip install --no-index transformers numpy pandas scikit-learn tqdm
echo ""

# Step 6: Install packages not available with --no-index
echo "Step 6: Installing additional packages..."
pip install datasets pyyaml tensorboard
echo ""

# Step 7: Create project directories
echo "Step 7: Creating project directories..."
mkdir -p $PROJECT_DIR/data/clutrr
mkdir -p $PROJECT_DIR/outputs
mkdir -p $PROJECT_DIR/logs
echo "Directories created"
echo ""

# Step 8: Generate requirements file
echo "Step 8: Generating requirements.txt..."
pip freeze > cc_requirements.txt
echo "Requirements saved to cc_requirements.txt"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Git clone your repository:"
echo "   cd $PROJECT_DIR"
echo "   git clone <your-repo-url> ."
echo ""
echo "2. Download the dataset:"
echo "   python download_data.py"
echo ""
echo "3. Edit train_job.sh with your account name:"
echo "   #SBATCH --account=def-<your-account>"
echo ""
echo "4. Submit the job:"
echo "   sbatch train_job.sh"
echo ""
echo "5. Monitor the job:"
echo "   sq"
echo "   cat logs/slurm-<job-id>.out"
echo ""
