#!/bin/bash
# Proper OpenFold installation script

set -e

echo "Installing OpenFold with proper dependencies..."

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc (NVIDIA CUDA Compiler) is not available"
    echo "Please load CUDA module first: module load CUDA/11.7.0"
    exit 1
fi

# Check Python version
# python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
# if ! python -c "import sys; sys.exit(0 if sys.version_info <= (3, 9) else 1)"; then
#     echo "Error: Python version $python_version is too high. OpenFold requires Python <= 3.9"
#     exit 1
# fi

# Check if PyTorch is installed
if ! python -c "import torch" 2>/dev/null; then
    echo "Error: PyTorch is not installed. Please install PyTorch first."
    exit 1
fi

echo "✓ nvcc available: $(nvcc --version | grep release)"
echo "✓ Python version: $python_version"
echo "✓ PyTorch available"

# Install ESMFold with OpenFold dependencies
echo "Installing fair-esm[esmfold]..."
pip install "fair-esm[esmfold]"

# Install DLLogger
echo "Installing dllogger..."
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'

# Install OpenFold from specific commit
echo "Installing OpenFold..."
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

echo "OpenFold installation complete!"
