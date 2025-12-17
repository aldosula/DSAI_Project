# DCASE 2024 Task 5 Setup Guide

## 1. Install Dependencies

First, activate your virtual environment and install the required packages:

```bash
# Activate the shared virtual environment
source /data/msc-proj/sppc18_venv/bin/activate

# Navigate to the project directory
cd /export/home/4sula/DSAI_Final_Project/DSAI_Project

# Install requirements
pip install -r requirements.txt
```

## 2. Update Configuration

The main config file is at `configs/train.yaml`. Key lines to modify:

**Line 70-74**: Update data paths to point to your dataset:

```yaml
path:
  root_dir: /data/msc-proj
  train_dir: ${path.root_dir}/Training_Set
  eval_dir: ${path.root_dir}/Validation_Set_DSAI_2025_2026
  test_dir: null
```

## 3. Run Training

### Simple Training (Quick Test):
```bash

python3 -m src.train --epochs 5 --n_way 5
```
