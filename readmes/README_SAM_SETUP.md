# SAM Setup

## Quick Setup

```bash
./setup/setup_sam.sh
```

## Manual Setup

```bash
# Clone SAM
cd ~/catkin_ws
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything

# Create environment
conda create -n sam python=3.9 -y
conda activate sam

# Install PyTorch (CUDA 12.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install rospkg catkin_pkg opencv-python pycocotools matplotlib numpy ultralytics

# Download checkpoint (choose one)
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O checkpoints/sam_vit_b.pth
# OR: sam_vit_l_0b3195.pth (1.2GB, 6GB VRAM)
# OR: sam_vit_h_4b8939.pth (2.4GB, 8GB VRAM)
```

## Verify

```bash
conda activate sam
python -c "from segment_anything import sam_model_registry; print('SAM OK')"
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

## Run

```bash
conda activate sam
source ~/catkin_ws/devel/setup.bash
rosrun ofpgs_ros sam_segmentation_node.py
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Use `vit_b` model |
| `No module 'segment_anything'` | Reinstall SAM |
| `No module 'rospy'` | `pip install rospkg catkin_pkg` |
