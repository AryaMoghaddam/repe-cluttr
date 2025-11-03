#!/bin/bash
# Personalized Compute Canada Setup Script for ajavadim
# CCI: pgy-481

echo "=========================================="
echo "RepE on CLUTRR - Compute Canada Setup"
echo "User: ajavadim"
echo "CCI: pgy-481"
echo "=========================================="
echo ""

# Step 1: Navigate to workspace
echo "Step 1: Setting up workspace directory..."
cd ~/projects/pgy-481/ajavadim || cd ~/projects/def-pgy-481/ajavadim || mkdir -p ~/projects/pgy-481/ajavadim && cd ~/projects/pgy-481/ajavadim
echo "Current directory: $(pwd)"
echo ""

# Step 2: Clone repository
echo "Step 2: Cloning repository..."
if [ -d "repe-cluttr" ]; then
    echo "  ⚠ Repository already exists, pulling latest changes..."
    cd repe-cluttr
    git pull
else
    git clone git@github.com:AryaMoghaddam/repe-cluttr.git
    cd repe-cluttr
fi
echo "Repository ready!"
echo ""

# Step 3: Load Python module
echo "Step 3: Loading Python module..."
module load python/3.9
python --version
echo ""

# Step 4: Create virtual environment
echo "Step 4: Creating virtual environment..."
if [ -d "~/envs/repe_env" ]; then
    echo "  ⚠ Virtual environment already exists"
else
    mkdir -p ~/envs
    virtualenv --no-download ~/envs/repe_env
fi
echo ""

# Step 5: Activate environment
echo "Step 5: Activating virtual environment..."
source ~/envs/repe_env/bin/activate
echo "Virtual environment activated!"
echo ""

# Step 6: Upgrade pip
echo "Step 6: Upgrading pip..."
pip install --no-index --upgrade pip
echo ""

# Step 7: Install PyTorch with CUDA
echo "Step 7: Installing PyTorch with CUDA support..."
pip install --no-index torch torchvision torchaudio
echo ""

# Step 8: Install other packages
echo "Step 8: Installing other dependencies..."
pip install --no-index transformers numpy pandas scikit-learn tqdm
echo ""

# Step 9: Install remaining packages
echo "Step 9: Installing additional packages..."
pip install datasets pyyaml tensorboard
echo ""

# Step 10: Create directories
echo "Step 10: Creating project directories..."
mkdir -p data/clutrr
mkdir -p outputs
mkdir -p logs
echo ""

# Step 11: Download dataset
echo "Step 11: Downloading CLUTRR dataset..."
python download_data.py
echo ""

# Step 12: Test setup
echo "Step 12: Testing setup..."
python test_setup.py
echo ""

# Step 13: Update SLURM script
echo "Step 13: Updating SLURM job script..."
sed -i "s/#SBATCH --account=def-<your-account>/#SBATCH --account=pgy-481/" train_job.sh
echo "  ✓ Updated account to: pgy-481"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit your email in train_job.sh:"
echo "   nano train_job.sh"
echo "   # Update: #SBATCH --mail-user=your.email@example.com"
echo ""
echo "2. Submit the job:"
echo "   sbatch train_job.sh"
echo ""
echo "3. Monitor the job:"
echo "   sq"
echo "   tail -f logs/slurm-<job-id>.out"
echo ""
