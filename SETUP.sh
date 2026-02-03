#!/bin/bash
# Setup script for nvdiffrec with intrinsic loss support
# Run this on a system with conda and CUDA 11.3+ installed

set -e

echo "============================================"
echo "NVDIFFREC Setup with Intrinsic Loss Support"
echo "============================================"

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda/Miniconda first."
    exit 1
fi

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "WARNING: CUDA not found in PATH. nvdiffrast compilation may fail."
fi

CONDA_ENV_NAME="nvdiffrec"
PYTHON_VERSION="3.9"

echo ""
echo "[1/4] Creating conda environment: $CONDA_ENV_NAME"
conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c conda-forge -y

echo ""
echo "[2/4] Activating environment and upgrading pip"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME
python -m pip install --upgrade pip

echo ""
echo "[3/4] Installing dependencies"
pip install ninja imageio imageio-ffmpeg PyOpenGL glfw xatlas gdown
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
imageio_download_bin freeimage

echo ""
echo "[4/4] Verifying installation"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import nvdiffrast; print('nvdiffrast: OK')"
python -c "import xatlas; print('xatlas: OK')"

echo ""
echo "============================================"
echo "✓ Setup complete!"
echo ""
echo "To use the environment, run:"
echo "  conda activate $CONDA_ENV_NAME"
echo ""
echo "To train with intrinsic loss:"
echo "  python train.py --config configs/bob.json --use-intrinsic --intrinsic-lambda 0.1"
echo "============================================"
