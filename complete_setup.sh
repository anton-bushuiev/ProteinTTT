#!/bin/bash
# Complete setup script for ProteinTTT with proper OpenFold installation
set -e

chmod +x install_openfold.sh
chmod +x setup_permanent_config.sh

echo "Complete ProteinTTT Setup with OpenFold"
echo "======================================"

echo "Setting up permanent configuration..."
./setup_permanent_config.sh

echo "Initializing conda..."
source "${CONDA_DIR}/etc/profile.d/conda.sh"

echo "Creating conda environment..."
conda env create -f environment.yml

echo "Activating environment..."
conda activate proteinttt

echo " Installing OpenFold..."
./install_openfold.sh

echo "Installing ProteinTTT..."
pip install -e .

echo "Setup complete!"
