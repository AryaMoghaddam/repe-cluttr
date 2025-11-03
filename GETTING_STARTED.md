# RepE on CLUTRR - Getting Started Summary

## 🎯 What You Have Now

A complete repository for running Representation Engineering (RepE) experiments on the CLUTRR dataset with support for:
- ✅ Local development
- ✅ Google Colab
- ✅ Compute Canada HPC

## 📁 Repository Structure

```
T244/
├── README.md                    # Project overview
├── COMPUTE_CANADA_GUIDE.md      # Detailed CC setup guide
├── GETTING_STARTED.md           # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
│
├── quickstart.sh                # Automated setup script
├── setup_cc.sh                  # Compute Canada setup
├── test_setup.py                # Test your setup
│
├── download_data.py             # Download CLUTRR dataset
├── repe_clutrr.py              # RepE model implementation
├── train_repe.py               # Training script
├── train_job.sh                # SLURM job script
├── colab_notebook.ipynb        # Colab notebook
│
├── data/clutrr/                # Dataset (created by download_data.py)
├── outputs/                    # Training outputs
└── logs/                       # SLURM logs
```

## 🚀 Quick Start Options

### Option 1: Local Development (Mac/Linux)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run quickstart script
./quickstart.sh

# 3. Train model
python train_repe.py --batch_size 8 --num_epochs 5
```

### Option 2: Google Colab

1. Upload `colab_notebook.ipynb` to Google Colab
2. Run all cells
3. That's it! 🎉

### Option 3: Compute Canada (Recommended for Serious Training)

#### First Time Setup:

```bash
# 1. SSH into Compute Canada
ssh your_username@beluga.computecanada.ca

# 2. Navigate to workspace
cd ~/projects/def-group/username/

# 3. Clone repository
git clone git@github.com:your-username/T244.git
cd T244

# 4. Setup environment
module load python/3.9
virtualenv --no-download ~/envs/repe_env
source ~/envs/repe_env/bin/activate
pip install --no-index --upgrade pip

# 5. Install dependencies
pip install --no-index torch torchvision torchaudio transformers numpy pandas scikit-learn tqdm
pip install datasets pyyaml tensorboard

# 6. Download data
python download_data.py

# 7. Test setup
python test_setup.py
```

#### Submit Training Job:

```bash
# 1. Edit train_job.sh - update these lines:
#    #SBATCH --account=def-<your-account>
#    #SBATCH --mail-user=your.email@example.com

# 2. Submit job
sbatch train_job.sh

# 3. Monitor
sq                                # Check status
tail -f logs/slurm-<job-id>.out   # View logs
```

## 📊 About the CLUTRR Dataset

CLUTRR tests compositional generalization and relational reasoning through family relationship problems.

**Example:**
- **Story**: "Alice is Bob's mother. Bob is Charlie's father."
- **Question**: "What is Alice's relationship to Charlie?"
- **Answer**: "grandmother"

**Dataset splits** (from https://github.com/fc2869/lo-fit/tree/main/dataset/clutrr):
- `train.json`: Training examples
- `val.json`: Validation examples
- `test.json`: Test examples

## 🧠 About RepE (Representation Engineering)

RepE learns to identify and manipulate task-relevant directions in a language model's representation space:

1. **Extract representations** from a specific layer
2. **Learn concept directions** that capture task-relevant features
3. **Apply interventions** to enhance task performance

## ⚙️ Configuration

Edit these files to customize:

- **`train_repe.py`**: Model hyperparameters, training settings
- **`train_job.sh`**: Compute Canada job resources (GPUs, memory, time)
- **`repe_clutrr.py`**: RepE model architecture

## 🐛 Troubleshooting

### Import errors?
```bash
python test_setup.py  # Check what's missing
pip install -r requirements.txt
```

### Data not downloaded?
```bash
python download_data.py
```

### SLURM job pending forever?
- Check your account allocation
- Reduce time/memory in `train_job.sh`
- Try: `squeue -u $USER -o "%.18i %.10T %.10M %.20R"`

### CUDA out of memory?
- Reduce batch size in training command
- Use gradient accumulation

## 📚 Documentation

- **README.md**: Project overview and setup
- **COMPUTE_CANADA_GUIDE.md**: Comprehensive CC guide with troubleshooting
- **Compute Canada Docs**: https://docs.alliancecan.ca/wiki/

## 🔑 Your SSH Key

You already have an SSH key configured for Compute Canada:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBDA0/o0n9WTIqfxQuyW04gNh5BXPV1tKje7k1OXxzu5 computecanada
```

Make sure to also:
1. Add your Compute Canada SSH key to GitHub (for cloning repos)
2. Configure SSH aliases in `~/.ssh/config` for easy access

## 🆘 Need Help?

If you encounter issues with Compute Canada setup, schedule a troubleshooting meeting to work through them together!

## 📝 Next Steps

1. **Test locally first**: Run `python test_setup.py` to verify everything works
2. **Small experiment**: Try training for 1-2 epochs locally/Colab
3. **Scale up on CC**: Once confident, submit full training job
4. **Monitor progress**: Use TensorBoard to track training
5. **Iterate**: Adjust hyperparameters based on results

## 🎓 Research Tips

- Start with a smaller model (GPT-2) before scaling up
- Use validation set to tune hyperparameters
- Save checkpoints frequently (configured in `train_repe.py`)
- Document your experiments (git commits, experiment notes)
- Compare against baselines

Good luck with your research! 🚀
