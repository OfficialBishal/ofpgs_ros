#!/bin/bash
# Setup script for SAM (Segment Anything Model)
# This script automates the installation process

set -e  # Exit on error

echo "=========================================="
echo "SAM (Segment Anything Model) Setup"
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

# Step 1: Clone SAM repository
echo ""
echo "Step 1: Cloning SAM repository..."
if [ -d "segment-anything" ]; then
    echo "  SAM repository already exists, skipping clone"
else
    git clone https://github.com/facebookresearch/segment-anything.git
    echo "  SAM repository cloned"
fi

cd segment-anything

# Step 2: Create conda environment
echo ""
echo "Step 2: Creating conda environment..."
if conda env list | grep -q "^sam "; then
    echo "  Conda environment 'sam' already exists"
    echo "  Activating existing environment..."
    source /opt/conda/etc/profile.d/conda.sh
    conda activate sam
else
    echo "  Creating new conda environment 'sam'..."
    conda create -n sam python=3.9 -y
    source /opt/conda/etc/profile.d/conda.sh
    conda activate sam
    echo "  Conda environment created"
fi

# Step 3: Install PyTorch with CUDA 12.1 (compatible with CUDA 12.8)
echo ""
echo "Step 3: Installing PyTorch with CUDA 12.1 support..."
echo "  (CUDA 12.8 is backward compatible with 12.1)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "  PyTorch installed"

# Step 4: Install SAM
echo ""
echo "Step 4: Installing SAM..."
pip install git+https://github.com/facebookresearch/segment-anything.git
echo "  SAM installed"

# Step 5: Install ROS and other dependencies
echo ""
echo "Step 5: Installing ROS and additional dependencies..."
pip install rospkg catkin_pkg opencv-python pycocotools matplotlib numpy
echo "  Dependencies installed"

# Step 5b: Install YOLO (for object detection strategy)
echo ""
echo "Step 5b: Installing YOLO (ultralytics) for object detection..."
read -p "Install YOLO for object detection strategy? [y/N] (default: y): " install_yolo
install_yolo=${install_yolo:-y}
if [[ "$install_yolo" =~ ^[Yy]$ ]]; then
    pip install ultralytics
    echo "  YOLO installed"
    echo "  Testing YOLO installation..."
    python -c "from ultralytics import YOLO; print('  YOLO import successful')" || {
        echo "  YOLO import test failed, but installation may still work"
    }
else
    echo "  Skipping YOLO installation"
    echo "  You can install it later with: pip install ultralytics"
fi

# Step 6: Download checkpoint
echo ""
echo "Step 6: Downloading SAM checkpoint..."
mkdir -p checkpoints

# Ask user which model to download
echo ""
echo "Which SAM model would you like to download?"
echo "  1) ViT-B (Fastest, ~375MB, ~4GB VRAM) - Recommended for testing"
echo "  2) ViT-L (Balanced, ~1.2GB, ~6GB VRAM)"
echo "  3) ViT-H (Best accuracy, ~2.4GB, ~8GB VRAM)"
echo "  4) Skip (download manually later)"
read -p "Enter choice [1-4] (default: 1): " model_choice
model_choice=${model_choice:-1}

case $model_choice in
    1)
        MODEL="vit_b"
        CHECKPOINT="sam_vit_b_01ec64.pth"
        URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        ;;
    2)
        MODEL="vit_l"
        CHECKPOINT="sam_vit_l_0b3195.pth"
        URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
        ;;
    3)
        MODEL="vit_h"
        CHECKPOINT="sam_vit_h_4b8939.pth"
        URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        ;;
    4)
        echo "  Skipping checkpoint download"
        echo "  You can download manually later using:"
        echo "    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O checkpoints/sam_vit_b.pth"
        CHECKPOINT=""
        ;;
    *)
        echo "  Invalid choice, skipping checkpoint download"
        CHECKPOINT=""
        ;;
esac

if [ -n "$CHECKPOINT" ]; then
    if [ -f "checkpoints/sam_${MODEL}.pth" ]; then
        echo "  Checkpoint already exists: checkpoints/sam_${MODEL}.pth"
    else
        echo "  Downloading ${MODEL} checkpoint..."
        wget "$URL" -O "checkpoints/sam_${MODEL}.pth"
        echo "  Checkpoint downloaded: checkpoints/sam_${MODEL}.pth"
    fi
fi

# Step 7: Verify installation
echo ""
echo "Step 7: Verifying installation..."
python -c "from segment_anything import sam_model_registry; print('  SAM imported successfully!')" || {
    echo "  SAM import failed!"
    exit 1
}

# Verify YOLO if installed
if python -c "from ultralytics import YOLO" 2>/dev/null; then
    echo "  YOLO imported successfully!"
    echo "  Note: YOLO models will be downloaded automatically on first use"
else
    echo "  YOLO not installed (optional for detection strategy)"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate sam"
echo "  2. Test SAM: python -c \"from segment_anything import sam_model_registry; print('OK')\""
echo "  3. Run ROS node: roslaunch ofpgs_ros foundationpose_with_sam.launch"
echo ""
echo "Checkpoint location:"
if [ -n "$CHECKPOINT" ]; then
    echo "  $WORKSPACE_ROOT/segment-anything/checkpoints/sam_${MODEL}.pth"
else
    echo "  (Download manually - see README_SAM_SETUP.md)"
fi
echo ""

