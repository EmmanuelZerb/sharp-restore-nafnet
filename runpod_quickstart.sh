#!/bin/bash
# ===========================================
# RUNPOD QUICKSTART SCRIPT
# Execute ce script directement sur RunPod
# ===========================================

set -e

echo "=========================================="
echo "  Sharp Restore - Setup RunPod"
echo "=========================================="

cd /workspace

# Clone ou cree le projet
if [ ! -d "sharp-restore-project" ]; then
    echo "Creating project structure..."
    mkdir -p sharp-restore-project/scripts
    mkdir -p sharp-restore-project/data
    mkdir -p sharp-restore-project/output
fi

cd sharp-restore-project

# Install dependencies
echo "Installing dependencies..."
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install --quiet numpy opencv-python pillow tqdm tensorboard einops timm scikit-image lpips gdown albumentations PyYAML

echo "Dependencies installed!"

# Check GPU
echo ""
echo "GPU Info:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB' if torch.cuda.is_available() else '')"

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload your scripts to /workspace/sharp-restore-project/scripts/"
echo "2. Run: python scripts/prepare_dataset.py --output ./data --download-gopro"
echo "3. Run: python scripts/train.py --train-blur ./data/processed/train/blur --train-sharp ./data/processed/train/sharp --output ./output"
echo ""
