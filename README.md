# RepE on CLUTRR Dataset

This repository contains code for running Representation Engineering (RepE) experiments on the CLUTRR (Compositional Language Understanding with Text-based Relational Reasoning) dataset.

## Project Structure

```
T244/
├── README.md
├── requirements.txt
├── download_data.py          # Download CLUTRR dataset
├── repe_clutrr.py           # Main RepE implementation
├── train_repe.py            # Training script
├── evaluate_repe.py         # Evaluation script
├── train_job.sh             # SLURM job script for Compute Canada
├── data/                    # Dataset directory
│   └── clutrr/
│       ├── train.json
│       ├── val.json
│       └── test.json
└── outputs/                 # Results and checkpoints
```

## Setup

### Local / Colab Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd T244
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the CLUTRR dataset:
```bash
python download_data.py
```

4. Run training:
```bash
python train_repe.py --config configs/default.yaml
```

### Compute Canada Setup

1. **SSH into Compute Canada**:
```bash
ssh beluga  # or your configured alias
```

2. **Clone the repository**:
```bash
cd ~/projects/def-<group>/username/
git clone <your-repo-url>
cd T244
```

3. **Setup virtual environment**:
```bash
module load python/3.9
virtualenv --no-download ~/envs/repe_env
source ~/envs/repe_env/bin/activate
pip install --no-index --upgrade pip
```

4. **Install dependencies**:
```bash
# Install PyTorch with CUDA
pip install --no-index torch torchvision torchaudio

# Install other packages
pip install --no-index transformers numpy pandas scikit-learn tqdm

# If packages not available with --no-index, install without it:
pip install datasets
```

5. **Download data** (on login node):
```bash
python download_data.py
```

6. **Submit job**:
```bash
sbatch train_job.sh
```

7. **Monitor job**:
```bash
sq  # Check job status
cat slurm-<job-id>.out  # View output logs
```

## CLUTRR Dataset

CLUTRR is a benchmark for testing compositional generalization and relational reasoning. The dataset consists of short narratives describing family relationships, with the task being to infer unstated relationships.

Example:
- Story: "Alice is Bob's mother. Bob is Charlie's father."
- Question: "What is Alice's relationship to Charlie?"
- Answer: "grandmother"

## RepE (Representation Engineering)

RepE learns to control and manipulate internal representations in language models by:
1. Identifying task-relevant directions in activation space
2. Learning linear transformations to enhance or suppress these directions
3. Applying interventions during inference to improve task performance

## Compute Canada Resources

- **Storage**: Use `$SCRATCH` for temporary files (20TB, 60-day retention)
- **Monitoring**: Use TensorBoard for training visualization
- **Documentation**: https://docs.alliancecan.ca/wiki/

## Troubleshooting

If you encounter issues with Compute Canada setup, please reach out to schedule a troubleshooting session.
