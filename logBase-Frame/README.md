# FSBED: Few-Shot Bioacoustic Event Detection

Production-quality implementation for DCASE 2024 Task 5.

## Installation

```bash
source /data/msc-proj/sppc18_venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### Training
```bash
python scripts/train.py --config configs/default.yaml
```


### Batch Inference
```bash
python scripts/infer.py \
    --input_dir /data/msc-proj/Validation_Set_DSAI_2025_2026 \
    --checkpoint checkpoints/best.pt \
    --output predictions/
```

### Evaluation
```bash
python scripts/evaluate.py \
    --pred_dir predictions/ \
    --gt_dir /data/msc-proj/Validation_Set_DSAI_2025_2026
```

## Project Structure

```
fsbed/
├── fsbed/                 # Main package
│   ├── config.py          # Configuration dataclasses
│   ├── seed.py            # Reproducibility
│   ├── logging_utils.py   # Logging setup
│   ├── audio/             # Audio processing
│   ├── data/              # Dataset utilities
│   ├── models/            # Neural network models
│   ├── adapt/             # Per-file adaptation
│   ├── postprocess/       # Post-processing
│   ├── eval/              # Evaluation metrics
│   └── pipelines/         # Training/inference pipelines
├── scripts/               # CLI entry points
├── configs/               # YAML configurations
└── tests/                 # Unit tests
```

## Configuration

Key parameters in `configs/default.yaml`:
- `sample_rate`: 22050 Hz
- `n_mels`: 128 mel bins
- `embed_dim`: 256 embedding dimension
- `n_support`: 5 support examples
- `adapt_iterations`: 3 pseudo-labeling rounds

## Data Paths

- **Training:** `/data/msc-proj/Training_Set`
- **Validation:** `/data/msc-proj/Validation_Set_DSAI_2025_2026`

## License

MIT
