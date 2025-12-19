# FoundationPose + SAM Setup

## Quick Setup

```bash
# 1. Setup SAM
./setup/setup_sam.sh

# 2. Verify
conda activate sam
python -c "from segment_anything import sam_model_registry; print('SAM OK')"

conda activate foundationpose
python -c "from estimater import FoundationPose; print('FoundationPose OK')"
```

## Run

```bash
source ~/catkin_ws/devel/setup.bash
roslaunch ofpgs_ros foundationpose_with_sam.launch
```

All parameters in `config/foundationpose_config.yaml`.

## Architecture

```
RGB Image → SAM (sam env) → Mask → FoundationPose (foundationpose env) → 6D Pose
```

## Performance

| With SAM | Without SAM |
|----------|-------------|
| Position < 5cm | ~35cm |
| Height < 2cm | ~30cm |
| Orientation < 10° | ~26° |

## Segmentation Strategies

Edit `config/foundationpose_config.yaml`:

| Strategy | Use Case |
|----------|----------|
| `detection` | **Recommended** - YOLO auto-detects object |
| `center_point` | Object is centered in image |
| `point` | Known object location |
| `box` | Have bounding box |
| `automatic` | Generate all masks, filter by size |

## SAM Models

| Model | VRAM | Speed |
|-------|------|-------|
| `vit_b` | ~4GB | Fastest |
| `vit_l` | ~6GB | Balanced |
| `vit_h` | ~8GB | Most accurate |

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No module 'segment_anything'` | `conda activate sam && pip install git+https://github.com/facebookresearch/segment-anything.git` |
| `No module 'rospy'` | `pip install rospkg catkin_pkg` |
| `No module 'estimater'` | Check FoundationPose in PYTHONPATH |
| `CUDA out of memory` | Use `vit_b` model |
| Empty mask | Try `detection` strategy |

## File Locations

- **SAM Node**: `scripts/sam_segmentation_node.py`
- **FoundationPose Node**: `scripts/foundationpose_pose_estimation_node.py`
- **Config**: `config/foundationpose_config.yaml`
- **Launch**: `launch/foundationpose_with_sam.launch`
