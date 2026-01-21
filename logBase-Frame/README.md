# FSBED: Few-Shot Bioacoustic Event Detection


## Installation

```bash
source /data/msc-proj/sppc18_venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### Training
```bash
python scripts/train_baseline.py --config configs/default.yaml
```


### Batch Inference
```bash
python scripts/infer_baseline.py \
    --input_dir /data/msc-proj/Evaluation_Set_DSAI_2025_2026 \
    --checkpoint /export/home/4sula/DSAI_Final_Project/DSAI_Project/logBase-Frame/checkpoints/best.pt \
    --output predictions/
```

### Evaluation
```bash
python scripts/evaluate.py \
    --pred_dir predictions/ \
    --gt_dir /data/msc-proj/Evaluation_Set_DSAI_2025_2026
```

## Project Structure

```
logBase-Frame/
├── fsbed/                 # Core package
│   ├── audio/             # Audio processing (Mel, PCEN)
│   ├── models/            # Embedding networks
│   ├── adapt/             # Pseudo-labeling logic
│   └── postprocess/       # NMS, smoothing
├── scripts/               # CLI entry points
└── configs/               # YAML configurations
```

## Configuration

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `sample_rate` | Audio sample rate | 22050 |
| `n_mels` | Mel frequency bins | 128 |
| `embed_dim` | Embedding dimension | 256 |
| `n_support` | Support examples | 5 |
| `adapt_iterations` | Pseudo-label rounds | 3 |


## Data Paths

- **Training:** `/data/msc-proj/Training_Set`
- **Validation:** `/data/msc-proj/Validation_Set_DSAI_2025_2026`

- **Evaluation:** `/data/msc-proj/Evaluation_Set_DSAI_2025_2026`

**Checkpoint:** `/export/home/4sula/DSAI_Final_Project/DSAI_Project/logBase-Frame/checkpoints/best.pt`


