# ProteinTTT Setup Guide

ProteinTTT is a flexible protein structure prediction toolkit that supports multiple models. The configuration files provided are specifically designed for OpenFold integration, but ProteinTTT can work with various other models that have different requirements.

## Model Options and Requirements

### Option 1: OpenFold Model (Heavy Setup)
**When to use the provided configuration:**
- You want to use OpenFold for protein folding
- You need state-of-the-art folding accuracy
- You have access to NVIDIA GPUs
- You're working with complex protein structures

**Requirements:**
- **Use the provided configuration files**
- NVIDIA GPU (8GB+ VRAM)
- CUDA 11.7 with nvcc compiler
- Heavy computational resources

**Configuration files needed:**
- `environment.yml`
- `install_openfold.sh`
- `complete_setup.sh`
- All OpenFold-specific setup scripts

### Option 2: Alternative Models (Lightweight Setup)
**When you DON'T need the OpenFold configuration:**
- Using ESM models without folding
- Using lighter transformer models
- Running on CPU or non-NVIDIA hardware
- Focusing on sequence analysis rather than structure prediction

**Requirements:**
- **Skip the OpenFold configuration**
- Any GPU (NVIDIA/AMD) or CPU
- Standard Python environment (3.8-3.11)
- Minimal computational resources


## Configuration File Usage Guide

### OpenFold-Specific Files 
- `environment.yml` - **For OpenFold users only**
- `install_openfold.sh` - **For OpenFold installation**
- `complete_setup.sh` - **For complete OpenFold setup**

### Universal Files (Always Useful)
- `.bashrc_additions` - **For any user** (path and environment setup)

## Permanent Environment Setup

Change USERNAME, PROJECT_DIR, PROJECTID in .bashrc_additions.

To avoid retyping environment variables in each session:

```bash
#  permanent configuration to your bash profile
cat .bashrc_additions >> ~/.bashrc
source ~/.bashrc
```

This automatically configures:
- Project paths and variables
- CUDA module loading
- Conda environment activation

## Environment Variables

The setup configures these environment variables:
```bash
export USERNAME=""
export PROJECT_DIR ="/scratch/project/4"
export PROJECTID=""
export WORK="$PROJECT_DIR/$USERNAME"
export CONDA_DIR="${WORK}/miniconda3"
```
## Quick Start
### 1. Clone the Repository
```bash
git clone https://github.com/anton-bushuiev/ProteinTTT.git
cd ProteinTTT
```
#### Change USERNAME, PROJECT_DIR, PROJECTID in .bashrc_additions

### 3. One-Command Setup
```bash
chmod +x complete_setup.sh
./complete_setup.sh
```

## Manual Setup Instructions

### Step 1: Prepare CUDA Environment
```bash
# Load CUDA module (required for OpenFold)
module load CUDA/11.7.0

# Verify CUDA is available
nvcc --version
```

### Step 2: Create Conda Environment
```bash
# Create environment with Python 3.9
conda env create -f environment.yml
conda activate proteinttt
```

### Step 3: Install OpenFold Dependencies
```bash
# Install OpenFold with proper dependencies
chmod +x install_openfold.sh
./install_openfold.sh
```

### Step 4: Install ProteinTTT
```bash
# Install ProteinTTT in development mode
pip install -e .
```


## Next Steps

After successful setup:
0. Use demo.ipynb to verify installation
1. Test with sample protein sequences
2. Explore ProteinTTT functionality
3. Configure for your specific use case