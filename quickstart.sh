#!/bin/bash
# Quick Start Script for DSAI Project

echo "================================================"
echo "DSAI Few-Shot Bioacoustic Detection - Quick Start"
echo "================================================"

# Navigate to project
cd /export/home/4sula/DSAI_Final_Project/DSAI_Project

# Activate environment
echo ""
echo "1. Activating virtual environment..."
source ../venv/bin/activate

# Show environment info
echo "   ✓ Virtual environment activated"
echo "   Python: $(python --version)"
echo "   PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "   CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Training options
echo ""
echo "================================================"
echo "2. Training Options:"
echo "================================================"
echo ""
echo "Quick Test (5 epochs):"
echo "  python -m src.train --epochs 5 --n_way 5"
echo ""
echo "Full Training (50 epochs):"
echo "  python -m src.train --epochs 50"
echo ""
echo "Custom Training:"
echo "  python -m src.train --epochs 50 --n_way 10 --lr 0.001"
echo ""

# Inference info
echo "================================================"
echo "3. Inference:"
echo "================================================"
echo ""
echo "After training, run:"
echo "  python run_inference.py --model_path best_model.pth --output_dir output"
echo ""

# Evaluation info
echo "================================================"
echo "4. Evaluation:"
echo "================================================"
echo ""
echo "Open Jupyter notebook:"
echo "  jupyter notebook evaluate_model.ipynb"
echo ""

read -p "Start quick test training now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Starting quick test training..."
    python -m src.train --epochs 5 --n_way 5 --n_support 5 --n_query 5
fi
