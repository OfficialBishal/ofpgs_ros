# Grounded SAM Setup

## Why Grounded SAM?

- Open-vocabulary detection (no class mapping needed)
- Better for custom objects (e.g., "cracker box")
- Text-based prompts

## Quick Setup

```bash
./setup/setup_grounded_sam.sh
```

## Manual Setup

```bash
# Create environment
conda create -n grounded_sam python=3.9 -y
conda activate grounded_sam

# Install PyTorch 1.13.1 (required)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install rospkg catkin_pkg opencv-python pycocotools matplotlib numpy transformers timm

# Clone Grounded SAM
cd ~/catkin_ws
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything

# Install Grounding DINO
cd GroundingDINO
pip install --no-build-isolation -e .
cd ..

# Download checkpoint
mkdir -p checkpoints
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O checkpoints/groundingdino_swint_ogc.pth
```

## Verify

```bash
conda activate grounded_sam
python -c "from groundingdino.util.inference import load_model; print('Grounding DINO OK')"
python -c "from segment_anything import sam_model_registry; print('SAM OK')"
```

## Run

```bash
source ~/catkin_ws/devel/setup.bash
roslaunch ofpgs_ros foundationpose_with_grounded_sam.launch
```

## Configuration

Edit `config/foundationpose_config.yaml`:

```yaml
grounded_sam:
  box_threshold: 0.80
  text_threshold: 0.80
```

Text prompt auto-generated from `object_name` (e.g., "cracker_box" → "cracker box").

## Troubleshooting

| Error | Fix |
|-------|-----|
| PyTorch version error | Must use PyTorch 1.13.1 |
| CUDA compilation error | Use `BUILD_WITH_CUDA=False` |
| Import error | Reinstall: `pip install --no-build-isolation -e GroundingDINO` |
