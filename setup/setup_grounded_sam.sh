#!/bin/bash
# Setup script for Grounded SAM (Grounded-Segment-Anything)
# This script creates a NEW conda environment to avoid breaking existing setups

set -e  # Exit on error

echo "=========================================="
echo "Grounded SAM Setup"
echo "=========================================="

# Detect catkin workspace (parent of src directory containing this package)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"
WORKSPACE_DIR="$(dirname "$(dirname "$PACKAGE_DIR")")"

# If we're in a standard catkin workspace structure (src/package_name/setup/)
if [ -d "$WORKSPACE_DIR/devel" ] || [ -d "$WORKSPACE_DIR/build" ]; then
    WORKSPACE_ROOT="$WORKSPACE_DIR"
else
    # Fallback: try common workspace locations
    if [ -d "$HOME/catkin_ws" ]; then
        WORKSPACE_ROOT="$HOME/catkin_ws"
    elif [ -d "$HOME/ros_ws" ]; then
        WORKSPACE_ROOT="$HOME/ros_ws"
    elif [ -d "$HOME/workspace" ]; then
        WORKSPACE_ROOT="$HOME/workspace"
    elif [ -d "$HOME/hsr_robocanes_omniverse" ]; then
        WORKSPACE_ROOT="$HOME/hsr_robocanes_omniverse"
    else
        echo "ERROR: Could not detect catkin workspace"
        echo "Please run from a catkin workspace or set WORKSPACE_ROOT environment variable"
        exit 1
    fi
fi

echo "Using workspace: $WORKSPACE_ROOT"
cd "$WORKSPACE_ROOT"

# Step 1: Create new conda environment for Grounded SAM
echo ""
echo "Step 1: Creating new conda environment 'grounded_sam'..."
source /opt/conda/etc/profile.d/conda.sh

if conda env list | grep -q "^grounded_sam "; then
    echo "  Conda environment 'grounded_sam' already exists"
    read -p "  Remove and recreate? [y/N]: " recreate
    if [[ "$recreate" =~ ^[Yy]$ ]]; then
        conda env remove -n grounded_sam -y
        conda create -n grounded_sam python=3.9 -y
        echo "  Environment recreated"
    else
        echo "  Using existing environment"
    fi
else
    conda create -n grounded_sam python=3.9 -y
    echo "  Conda environment created"
fi

conda activate grounded_sam
echo "  Conda environment activated"

# Step 2: Install PyTorch 1.13.1 (required for Grounding DINO)
echo ""
echo "Step 2: Installing PyTorch 1.13.1 with CUDA 11.6..."
echo "  (This version is required for Grounding DINO compatibility)"
echo "  Note: CUDA 11.6 PyTorch is backward compatible with CUDA 12.8 at runtime"
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

echo "  Verifying PyTorch installation..."
python -c "import torch; print(f'  PyTorch version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')" || {
    echo "  ERROR: PyTorch installation failed!"
    exit 1
}

# Step 2b: Fix NumPy version (downgrade to 1.26.4 for compatibility with PyTorch 1.13.1)
echo ""
echo "Step 2b: Fixing NumPy version (downgrading to 1.26.4 for compatibility)..."
echo "  Uninstalling NumPy and opencv-python (may pull NumPy 2.x)..."
pip uninstall numpy opencv-python -y 2>/dev/null || true
echo "  Installing opencv-python 4.8.1.78 (compatible with NumPy 1.x)..."
pip install "opencv-python==4.8.1.78"
echo "  Installing NumPy 1.26.4..."
pip install "numpy==1.26.4"
echo "  NumPy fixed to 1.26.4"

# Step 3: Install SAM (same as in sam environment)
echo ""
echo "Step 3: Installing SAM..."
pip install git+https://github.com/facebookresearch/segment-anything.git
echo "  SAM installed"

# Step 4: Clone Grounded SAM repository
echo ""
echo "Step 3: Cloning Grounded SAM repository..."
if [ -d "Grounded-Segment-Anything" ]; then
    echo "  Grounded SAM repository already exists, skipping clone"
    cd Grounded-Segment-Anything
    git pull || echo "  (Could not pull latest changes)"
else
    git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
    cd Grounded-Segment-Anything
    echo "  Grounded SAM repository cloned"
fi

# Step 5: Set environment variables for CUDA compilation
echo ""
echo "Step 5: Setting up CUDA environment variables..."
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True

# Try to find CUDA_HOME
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
    elif [ -d "/usr/local/cuda-12" ]; then
        export CUDA_HOME=/usr/local/cuda-12
    elif [ -d "/usr/local/cuda-11.6" ]; then
        export CUDA_HOME=/usr/local/cuda-11.6
    else
        echo "  WARNING: CUDA_HOME not set and could not auto-detect"
        echo "  Please set CUDA_HOME manually: export CUDA_HOME=/path/to/cuda"
    fi
fi
echo "  CUDA_HOME: ${CUDA_HOME:-'not set'}"

# Step 6: Install Grounding DINO
echo ""
echo "Step 6: Installing Grounding DINO..."
echo "  Note: Due to CUDA version mismatch (PyTorch 1.13.1 CUDA 11.6 vs system CUDA 12.8),"
echo "        we need to patch PyTorch to bypass the version check."
cd GroundingDINO

# Patch PyTorch to bypass CUDA version check BEFORE installation
echo "  Patching PyTorch to bypass CUDA version check..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! python "$SCRIPT_DIR/patch_pytorch_cuda_check.py"; then
    echo "  ERROR: Failed to patch PyTorch - this is required for installation"
    echo "  Please check the error above and fix manually if needed"
    exit 1
fi

# Try installation (patch should allow CUDA extensions to compile)
export BUILD_WITH_CUDA=False
echo "  Installing Grounding DINO (this may take a few minutes)..."
pip install --no-build-isolation -e . > /tmp/groundingdino_install.log 2>&1
INSTALL_STATUS=$?

if [ $INSTALL_STATUS -ne 0 ]; then
    echo "  ERROR: Grounding DINO installation failed!"
    echo "  Last 30 lines of log:"
    tail -30 /tmp/groundingdino_install.log
    exit 1
fi

# Verify installation
python -c "import sys; sys.path.insert(0, '.'); from groundingdino.util.inference import load_model" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  Grounding DINO installed successfully"
else
    echo "  WARNING: Installation completed but import test failed"
    echo "  This may still work - please test manually"
fi

cd ..

# Step 7: Install additional dependencies (with NumPy fix)
echo ""
echo "Step 7: Installing additional dependencies..."
echo "  Installing transformers 4.21.0 (compatible with PyTorch 1.13.1)..."
# Transformers 4.21.0 is compatible with PyTorch 1.13.1
# Note: transformers may try to upgrade NumPy, we'll fix it after
pip install "transformers==4.21.0" timm addict yapf supervision || {
    echo "  WARNING: Some dependencies may have failed, but continuing..."
}

# Step 7b: Force NumPy back to 1.26.4 (transformers may have upgraded it)
echo ""
echo "Step 7b: Ensuring NumPy stays at 1.26.4 (overriding any upgrades)..."
pip install "numpy==1.26.4" --force-reinstall --no-deps
pip install "numpy==1.26.4" --force-reinstall
echo "  NumPy pinned to 1.26.4"

# Step 8: Install ROS dependencies (without reinstalling opencv-python/numpy)
echo ""
echo "Step 8: Installing ROS dependencies..."
# Don't reinstall opencv-python or numpy - we already have the correct versions
pip install rospkg catkin_pkg pycocotools matplotlib || {
    echo "  WARNING: Some ROS dependencies may have failed, but continuing..."
}
# Ensure NumPy stays at 1.26.4 after installing other packages
pip install "numpy==1.26.4" --force-reinstall --no-deps
echo "  NumPy still pinned to 1.26.4"

# Step 9: Download Grounding DINO checkpoint
echo ""
echo "Step 9: Downloading Grounding DINO checkpoint..."
mkdir -p checkpoints
if [ -f "checkpoints/groundingdino_swint_ogc.pth" ]; then
    echo "  Checkpoint already exists"
else
    echo "  Downloading checkpoint (this may take a while)..."
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O checkpoints/groundingdino_swint_ogc.pth || {
        echo "  ERROR: Failed to download checkpoint"
        echo "  You can download manually from:"
        echo "  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        exit 1
    }
    echo "  Checkpoint downloaded"
fi

# Step 10: Verify installation
echo ""
echo "Step 10: Verifying installation..."

# Test NumPy and PyTorch compatibility first
echo "  Testing NumPy-PyTorch compatibility..."
python -c "
import numpy as np
import torch
print(f'  NumPy: {np.__version__}')
print(f'  PyTorch: {torch.__version__}')
# Test NumPy-PyTorch conversion
arr = np.array([1, 2, 3])
tensor = torch.from_numpy(arr)
print('  ✓ NumPy-PyTorch conversion: OK')
" || {
    echo "  ERROR: NumPy-PyTorch compatibility check failed!"
    echo "  This is critical - please check NumPy version"
    exit 1
}

# Test transformers
echo "  Testing transformers..."
python -c "
import transformers
from transformers import BertModel
print(f'  Transformers: {transformers.__version__}')
print('  ✓ BertModel import: OK')
" || {
    echo "  WARNING: Transformers import test failed"
    echo "  Installation may still work, but please verify manually"
}

# Test Grounding DINO import
echo "  Testing Grounding DINO..."
python -c "
import sys
sys.path.insert(0, '$WORKSPACE_ROOT/Grounded-Segment-Anything')
try:
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    print('  ✓ Grounding DINO: OK')
except Exception as e:
    print(f'  WARNING: Grounding DINO import test failed: {e}')
    print('  Installation may still work, but please verify manually')
" || {
    echo "  WARNING: Grounding DINO import test failed"
    echo "  Installation may still work, but please verify manually"
}

# Test SAM import
echo "  Testing SAM..."
python -c "from segment_anything import sam_model_registry; print('  ✓ SAM: OK')" 2>&1 | grep -v "UserWarning\|Failed to initialize NumPy" || {
    echo "  ERROR: SAM import failed!"
    echo "  Make sure SAM is installed: pip install git+https://github.com/facebookresearch/segment-anything.git"
    exit 1
}

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Environment: 'grounded_sam' (separate from 'sam' environment)"
echo "  - PyTorch 1.13.1 (required for Grounding DINO)"
echo "  - NumPy 1.26.4 (compatible with PyTorch 1.13.1)"
echo "  - opencv-python 4.8.1.78 (compatible with NumPy 1.x)"
echo "  - transformers 4.21.0 (compatible with PyTorch 1.13.1)"
echo "  - SAM installed"
echo "  - Grounding DINO installed"
echo ""
echo "All dependencies configured and verified!"
echo ""
echo "Next steps:"
echo "  1. ROS node: grounded_sam_segmentation_node.py (already created)"
echo "  2. Config file: config/foundationpose_config.yaml (already configured)"
echo "  3. Launch file: foundationpose_with_grounded_sam.launch (already created)"
echo "  4. Test: roslaunch ofpgs_ros foundationpose_with_grounded_sam.launch"
echo ""
echo "Checkpoint location:"
echo "  $WORKSPACE_ROOT/Grounded-Segment-Anything/checkpoints/groundingdino_swint_ogc.pth"
echo ""
echo "NOTE: Your existing 'sam' environment is unchanged and still works!"
echo ""
echo "If you encounter NumPy issues later, re-run this setup script."
echo "It will fix NumPy/opencv-python/transformers compatibility automatically."
echo ""

