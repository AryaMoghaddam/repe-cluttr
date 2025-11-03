# Compute Canada Setup Guide for RepE on CLUTRR

## Prerequisites

✅ You already have an SSH key added to Compute Canada:
- Key: `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBDA0/o0n9WTIqfxQuyW04gNh5BXPV1tKje7k1OXxzu5 computecanada`

## Quick Start

### 1. Connect to Compute Canada

```bash
# SSH into your cluster (replace with your username and cluster)
ssh your_username@beluga.computecanada.ca

# Or if you've configured an alias in ~/.ssh/config:
ssh beluga
```

### 2. Setup SSH Config (Optional but Recommended)

On your **local machine**, add this to `~/.ssh/config`:

```bash
Host beluga
    HostName beluga.computecanada.ca
    User your_username
    IdentityFile ~/.ssh/id_ed25519

Host cedar
    HostName cedar.computecanada.ca
    User your_username
    IdentityFile ~/.ssh/id_ed25519

Host graham
    HostName graham.computecanada.ca
    User your_username
    IdentityFile ~/.ssh/id_ed25519
```

Now you can simply: `ssh beluga`

### 3. Setup GitHub SSH on Compute Canada

If you haven't already, add your Compute Canada SSH key to GitHub:

```bash
# On Compute Canada, generate SSH key if needed
ssh-keygen -t ed25519 -C "your_email@example.com"

# Display your public key
cat ~/.ssh/id_ed25519.pub

# Copy the output and add it to GitHub:
# Go to: https://github.com/settings/keys
# Click "New SSH key" and paste your public key
```

### 4. Navigate to Project Directory

```bash
# Navigate to your workspace
# Replace 'def-group' and 'username' with your actual values
cd ~/projects/def-group/username/

# Or create a symbolic link for easier access
ln -s ~/projects/def-group/username ~/workspace
cd ~/workspace
```

### 5. Clone Repository

```bash
# Clone your T244 repository
git clone git@github.com:your-username/T244.git
cd T244
```

### 6. Setup Virtual Environment

```bash
# Load Python module
module load python/3.9

# Create virtual environment
mkdir -p ~/envs
virtualenv --no-download ~/envs/repe_env

# Activate environment
source ~/envs/repe_env/bin/activate

# Upgrade pip
pip install --no-index --upgrade pip
```

### 7. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install --no-index torch torchvision torchaudio

# Install other packages available on Compute Canada
pip install --no-index transformers numpy pandas scikit-learn tqdm

# Check what's available
avail_wheels

# Install packages not available with --no-index
pip install datasets pyyaml tensorboard

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 8. Download Dataset

```bash
# Run on login node (quick task < 10 CPU-minutes)
python download_data.py

# Verify download
ls -lh data/clutrr/
```

### 9. Edit SLURM Job Script

Edit `train_job.sh` and update these lines:

```bash
#SBATCH --account=def-<your-account>     # YOUR RESOURCE ALLOCATION PROJECT
#SBATCH --mail-user=your.email@example.com  # YOUR EMAIL
```

To find your account:
1. Log into [CCDB](https://ccdb.computecanada.ca/)
2. Look under "Resource Allocation Projects"
3. Use the "Group Name" value

### 10. Submit Job

```bash
# Create logs directory
mkdir -p logs

# Submit the job
sbatch train_job.sh

# Note the job ID returned
```

### 11. Monitor Job

```bash
# Check job status (all jobs)
squeue -u $USER

# Or use the shorthand
sq

# View job details
scontrol show job <job-id>

# Watch job status in real-time
watch -n 10 sq

# View output logs (while job is running or after completion)
tail -f logs/slurm-<job-id>.out

# View error logs
tail -f logs/slurm-<job-id>.err

# Check GPU utilization (if job is running)
ssh <compute-node> nvidia-smi
```

### 12. TensorBoard Monitoring (Optional)

To view TensorBoard locally:

```bash
# On your local machine, create SSH tunnel
# Replace <compute-node> with actual node (find with: squeue -u $USER)
ssh -N -f -L localhost:6006:<compute-node>:6006 username@beluga.computecanada.ca

# Then open in browser: http://localhost:6006
```

### 13. Cancel Job (if needed)

```bash
# Cancel a specific job
scancel <job-id>

# Cancel all your jobs
scancel -u $USER
```

## File Locations

### Important Directories

- **Home**: `~/` - 50 GB, backed up, for code and scripts
- **Project**: `~/projects/def-group/username/` - Shared space, backed up
- **Scratch**: `$SCRATCH` or `~/scratch/` - 20 TB, files older than 60 days purged, for temporary outputs
- **SLURM_TMPDIR**: Available during job, on local SSD, fast I/O for small files

### Where to Put What

- **Code**: `~/projects/def-group/username/T244/` (backed up)
- **Dataset**: `~/scratch/data/clutrr/` (can be large)
- **Outputs**: `$SCRATCH/repe_outputs/` (temporary results)
- **Important results**: Copy back to project directory before 60 days

## Common Issues and Solutions

### Issue 1: Job Pending Forever

**Cause**: Requesting too many resources or wrong account

**Solution**:
```bash
# Check job priority
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.10T %.10M %.6D %.20S %.20R"

# Reduce time or resources in train_job.sh
#SBATCH --time=01:00:00  # Try shorter time
#SBATCH --mem=16G        # Try less memory
```

### Issue 2: CUDA Out of Memory

**Solution**: Reduce batch size in train_job.sh:
```bash
python train_repe.py \
    --batch_size 8 \  # Reduce from 16 to 8
    ...
```

### Issue 3: Module Not Found

**Solution**:
```bash
# Make sure virtual environment is activated
source ~/envs/repe_env/bin/activate

# Reinstall package
pip install <package-name>
```

### Issue 4: Dataset Download Fails

**Solution**:
```bash
# Compute Canada has limited internet on compute nodes
# Always download data on login node before submitting job

# Or use wget if Python script fails
wget https://raw.githubusercontent.com/fc2869/lo-fit/main/dataset/clutrr/train.json -O data/clutrr/train.json
wget https://raw.githubusercontent.com/fc2869/lo-fit/main/dataset/clutrr/val.json -O data/clutrr/val.json
wget https://raw.githubusercontent.com/fc2869/lo-fit/main/dataset/clutrr/test.json -O data/clutrr/test.json
```

## Useful Commands

```bash
# Check disk usage
diskusage_report

# Check available GPU types
sinfo -o "%20N %10c %10m %25f %10G"

# List available software modules
module avail

# Check your allocation
sshare -U

# Interactive session for testing (max 3 hours)
salloc --account=def-group --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=1:00:00

# Copy files from Compute Canada to local
scp -r username@beluga.computecanada.ca:~/path/to/outputs ./local_directory/
```

## Next Steps After Job Completes

1. **Check results**:
   ```bash
   ls -lh $SCRATCH/repe_outputs/
   cat logs/slurm-<job-id>.out
   ```

2. **Copy important results**:
   ```bash
   cp -r $SCRATCH/repe_outputs/best_model.pt ~/projects/def-group/username/T244/outputs/
   ```

3. **Analyze with Jupyter** (if available):
   ```bash
   # Request interactive session
   salloc --account=def-group --gres=gpu:1 --mem=16G --time=2:00:00
   
   # Start Jupyter
   jupyter notebook --no-browser --port=8888
   
   # On local machine, create tunnel
   ssh -N -f -L localhost:8888:localhost:8888 username@beluga.computecanada.ca
   ```

## Resources

- [Compute Canada Documentation](https://docs.alliancecan.ca/wiki/)
- [Running Jobs](https://docs.alliancecan.ca/wiki/Running_jobs)
- [Using GPUs](https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm)
- [Python Guide](https://docs.alliancecan.ca/wiki/Python)
- [Storage Guide](https://docs.alliancecan.ca/wiki/Storage_and_file_management)

## Getting Help

If you encounter issues with Compute Canada setup, schedule a troubleshooting meeting to work through them together!
