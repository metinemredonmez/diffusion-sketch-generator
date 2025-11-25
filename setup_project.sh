#!/bin/bash

# MYTH Sketch Diffusion - Project Setup Script
# Bu script'i proje klasörünün içinde çalıştır

echo "=========================================="
echo "MYTH Sketch Diffusion - Project Setup"
echo "=========================================="
echo ""

# 1. Conda environment oluştur
echo "Step 1: Creating conda environment..."
conda create -n sketch python=3.9 -y

# 2. Environment'ı aktifleştir
echo ""
echo "Step 2: Activating environment..."
eval "$(conda shell.bash hook)"
conda activate sketch

# 3. PyTorch ve dependencies kur
echo ""
echo "Step 3: Installing PyTorch and dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Diğer kütüphaneleri kur
echo ""
echo "Step 4: Installing other packages..."
pip install ndjson matplotlib pillow numpy tqdm scikit-learn scipy
pip install clean-fid einops jupyter

# 5. Data klasörü oluştur
echo ""
echo "Step 5: Creating data directory..."
mkdir -p data

# 6. Subset.zip'i extract et (eğer varsa)
echo ""
echo "Step 6: Extracting subset.zip..."
if [ -f "subset.zip" ]; then
    unzip -q subset.zip
    echo "✓ subset.zip extracted"
else
    echo "⚠ subset.zip not found - please download from assignment"
fi

# 7. Quick Draw dataset'i indir (UZUN SÜRER!)
echo ""
echo "Step 7: Downloading Quick Draw dataset..."
echo "⚠ WARNING: This will download ~500MB of data"
read -p "Do you want to download now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading cat.ndjson..."
    gsutil -m cp gs://quickdraw_dataset/full/simplified/cat.ndjson ./data/ 2>/dev/null || \
        echo "⚠ gsutil not found. Install with: pip install gsutil"
    
    echo "Downloading bus.ndjson..."
    gsutil -m cp gs://quickdraw_dataset/full/simplified/bus.ndjson ./data/ 2>/dev/null || \
        echo "⚠ gsutil not found. Install with: pip install gsutil"
    
    echo "Downloading rabbit.ndjson..."
    gsutil -m cp gs://quickdraw_dataset/full/simplified/rabbit.ndjson ./data/ 2>/dev/null || \
        echo "⚠ gsutil not found. Install with: pip install gsutil"
else
    echo "Skipping download. You can download later with:"
    echo "  gsutil -m cp gs://quickdraw_dataset/full/simplified/cat.ndjson ./data/"
    echo "  gsutil -m cp gs://quickdraw_dataset/full/simplified/bus.ndjson ./data/"
    echo "  gsutil -m cp gs://quickdraw_dataset/full/simplified/rabbit.ndjson ./data/"
fi

# 8. Gerekli klasörleri oluştur
echo ""
echo "Step 8: Creating project directories..."
mkdir -p checkpoints
mkdir -p models
mkdir -p results
mkdir -p fid_eval

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. conda activate sketch"
echo "  2. jupyter notebook sketch_diffusion_solution.ipynb"
echo ""
echo "Project structure:"
echo "  data/           - Quick Draw NDJSON files"
echo "  subset/         - Train/test indices"
echo "  checkpoints/    - Training checkpoints"
echo "  models/         - Final trained models"
echo "  results/        - Generated outputs"
echo ""
