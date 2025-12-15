# Few-Shot Bioacoustic Event Detection - DSAI Final Project

Python 3.11 compatible implementation of Prototypical Networks for few-shot bioacoustic sound event detection.

## Project Structure

```
DSAI_Project/
├── src/                      # Source code
│   ├── dataset.py           # Data loading and episodic sampling
│   ├── features.py          # Feature extraction (Log-Mel, PCEN)
│   ├── model.py             # Prototypical Network model
│   ├── train.py             # Training script
│   └── inference.py         # Inference pipeline
├── run_inference.py         # Inference runner script
├── evaluate_model.ipynb     # Model evaluation notebook
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Navigate to project root
cd /export/home/4sula/DSAI_Final_Project

# Activate shared virtual environment (located in group storage)
source /data/msc-proj/sppc18_venv/bin/activate

# Install dependencies (if not already installed)
cd DSAI_Project
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Quick training (5 epochs)
python -m src.train --epochs 5 --n_way 5 --n_support 5 --n_query 5

# Full training (50 epochs)
python -m src.train --epochs 50 --n_way 5 --n_support 5 --n_query 5

# With custom parameters
python -m src.train \
    --epochs 50 \
    --n_way 10 \
    --n_support 5 \
    --n_query 5 \
    --lr 0.001 \
    --segment_len 0.2
```

### 3. Run Inference

```bash
# Run inference on validation set
python run_inference.py \
    --model_path best_model.pth \
    --val_dir /data/msc-proj/Validation_Set_DSAI_2025_2026 \
    --output_dir output_verify
```

### 4. Evaluate Results

Open and run the Jupyter notebook:

```bash
jupyter notebook evaluate_model.ipynb
```

## 📊 Data Paths

**Training Data:** `/data/msc-proj/Training_Set`
- 174 audio files with annotations
- Multiple bioacoustic sources (BV, HT, JD, MT, WMW)

**Validation Data:** `/data/msc-proj/Validation_Set_DSAI_2025_2026`
- 12 audio files for validation
- DCASE-compliant format

## 🔧 Configuration

### Training Arguments

```bash
--epochs          # Number of training epochs (default: 50)
--n_way           # Number of classes per episode (default: 5)
--n_support       # Support examples per class (default: 5)
--n_query         # Query examples per class (default: 5)
--lr              # Learning rate (default: 0.001)
--batch_size      # Batch size (default: 4)
--segment_len     # Audio segment length in seconds (default: 0.2)
--sample_rate     # Audio sample rate (default: 22050)
```

### Inference Arguments

```bash
--model_path      # Path to trained model checkpoint
--val_dir         # Directory containing validation audio files
--output_dir      # Directory for output CSV files
--threshold       # Detection threshold (default: 0.5)
--hop_len         # Sliding window hop length (default: 0.05s)
```

## 📈 Model Architecture

```
Audio Input → Log-Mel Spectrogram → MobileNetV3 → Embeddings → Prototypes → Predictions
               (128 mel bins)        (Backbone)     (512-dim)    (Distance)
```

**Key Components:**
- **Feature Extraction:** Log-Mel spectrograms (22.05 kHz, 128 mel bins)
- **Backbone:** MobileNetV3-Small for efficient embeddings
- **Meta-Learning:** Prototypical Networks with episodic training
- **Inference:** Sliding window with post-processing

## 📝 Output Format

Predictions are saved as DCASE-compliant CSV files:

```csv
Audiofilename,Starttime,Endtime
audio.wav,1.5,2.3
audio.wav,5.2,6.1
```

## 🔍 Evaluation Metrics

The evaluation notebook (`evaluate_model.ipynb`) calculates:
- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1-Score:** 2 × (Precision × Recall) / (Precision + Recall)

Using IoU (Intersection over Union) matching with threshold 0.5.

## 📂 Related Files

**Parent Directory:** `/export/home/4sula/DSAI_Final_Project/`

Additional resources:
- `simplified_dcase/` - Simplified DCASE baseline implementation
- `dcase-few-shot-bioacoustic/` - Original DCASE baseline reference
- `DCASE_SETUP.md` - DCASE baseline setup guide
- `DCASE_PY311_GUIDE.md` - Python 3.11 compatibility guide

## ⚙️ System Requirements

- **Python:** 3.11.2
- **PyTorch:** ≥2.0 (CUDA compatible)
- **Torchaudio:** ≥2.0
- **GPU:** CUDA-enabled (recommended)
- **Memory:** 8GB+ RAM, 4GB+ VRAM

## 🐛 Troubleshooting

**Issue:** `torchaudio.info` attribute error  
**Solution:** Code is updated to use `torchaudio.backend.soundfile_backend.info()` for compatibility

**Issue:** CUDA out of memory  
**Solution:** Reduce `--batch_size` or `--n_way` parameters

**Issue:** Import errors  
**Solution:** Ensure you're using the correct venv:
```bash
source /data/msc-proj/sppc18_venv/bin/activate
```

## 📚 References

- **DCASE 2024 Task 5:** Few-Shot Bioacoustic Event Detection
- **Prototypical Networks:** Snell et al., 2017
- **Dataset:** Multi-source bioacoustic recordings

## 🎯 Performance

With proper training (50 epochs):
- **Baseline F1:** 0-15% (1 epoch)
- **Expected F1:** 15-30% (50 epochs)
- **Optimized F1:** 30-50% (with tuning)

## 📞 Contact

For questions or issues, refer to the comprehensive guides in the parent directory.
