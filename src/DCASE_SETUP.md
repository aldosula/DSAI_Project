# DCASE 2024 Task 5 Setup Guide

## 1. Install Dependencies

First, activate your virtual environment and install the required packages:

```bash
# Activate the virtual environment
source /export/home/4sula/DSAI_Final_Project/venv/bin/activate

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
python train.py trainer.max_epochs=10 train_param.num_episodes=100
```

### Full Training (Official Baseline):
```bash
python train.py
```

### With Different Features:
```bash
# Try PCEN features
python train.py features.feature_types="pcen" exp_name="pcen_experiment"

# Try combined features
python train.py features.feature_types="logmel@mfcc" exp_name="logmel_mfcc"
```

## 4. Key Configuration Parameters

- `train_param.seg_len`: Segment length (0.2s = 200ms)
- `train_param.n_shot`: Few-shot setting (5-shot)
- `train_param.k_way`: Number of classes per episode (10)
- `train_param.num_episodes`: Training episodes (2000)
- `trainer.max_epochs`: Maximum training epochs

## 5. Expected Output

Training outputs will be saved to:
- `logs/` - Training logs
- Checkpoints will be saved during training
- WandB logs (if configured)

## 6. Common Issues

**Issue**: CUDA out of memory
**Solution**: Reduce batch size or segment length

**Issue**: Missing dependencies
**Solution**: Run `pip install python-dotenv hydra-core pytorch-lightning`

## 7. Inference/Evaluation

After training, use your best checkpoint:

```bash
python test.py ckpt_path=/path/to/checkpoint.ckpt
```

## Notes

- The baseline uses **Prototypical Networks** (same as your implementation!)
- It includes advanced features like:
  - Multi-scale feature extraction
  - Negative contrastive learning
  - Adaptive segment length
  - Various audio features (PCEN, MFCC, LogMel)
