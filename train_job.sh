#!/bin/bash
#SBATCH --job-name=repe_clutrr           # Job name
#SBATCH --account=pgy-481                # Resource allocation project
#SBATCH --time=03:00:00                  # Time limit (3 hours)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=6                # CPU cores for data loading
#SBATCH --mem=32G                        # Memory per node
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --output=logs/slurm-%j.out       # Standard output log
#SBATCH --error=logs/slurm-%j.err        # Standard error log
#SBATCH --mail-type=END,FAIL             # Email on job end/fail
#SBATCH --mail-user=your.email@example.com  # Your email

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Load required modules
module load python/3.9
module load cuda/11.7  # Adjust based on available CUDA version

# Activate virtual environment
source ~/envs/repe_env/bin/activate

# Print environment info
echo "Python version:"
python --version
echo ""

echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo ""

echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
echo ""

echo "GPU info:"
nvidia-smi
echo ""

# Navigate to project directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Copy data to local scratch for faster I/O (optional but recommended)
echo "Copying data to scratch..."
mkdir -p $SLURM_TMPDIR/data
cp -r data/clutrr $SLURM_TMPDIR/data/
echo "Data copied to $SLURM_TMPDIR/data/clutrr"
echo ""

# Run training script
echo "=========================================="
echo "Starting training..."
echo "=========================================="
echo ""

python train_repe.py \
    --model_name gpt2 \
    --layer_idx -1 \
    --num_directions 10 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --data_dir $SLURM_TMPDIR/data/clutrr \
    --output_dir $SCRATCH/repe_outputs \
    --exp_name "repe_clutrr_$(date +%Y%m%d_%H%M%S)" \
    --eval_every 100 \
    --save_every 500

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "End Time: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Training failed with exit code $?"
    echo "End Time: $(date)"
    echo "=========================================="
    exit 1
fi

# Copy results from scratch to project directory (optional)
echo ""
echo "Copying results back to project directory..."
cp -r $SCRATCH/repe_outputs/* outputs/ 2>/dev/null || echo "No outputs to copy"

echo ""
echo "Job finished!"
