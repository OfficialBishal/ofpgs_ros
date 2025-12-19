#!/bin/bash
# Wrapper script to run Grounded SAM Segmentation node
# Uses Docker's ROS1 setup + grounded_sam conda environment

# Activate grounded_sam conda environment FIRST (before ROS)
source /opt/conda/etc/profile.d/conda.sh
conda activate grounded_sam

# Source ROS setup (required for ROS Python packages and rospack)
source /opt/ros/noetic/setup.bash

# Source ROS workspace (required for rospack find)
if [ -f ~/hsr_robocanes_omniverse/devel/setup.bash ]; then
    source ~/hsr_robocanes_omniverse/devel/setup.bash
fi

# Get the package path using rospack
PACKAGE_PATH=$(rospack find ofpgs_ros 2>/dev/null)
if [ -z "$PACKAGE_PATH" ]; then
    echo "ERROR: Could not find ofpgs_ros package"
    exit 1
fi

# Set PYTHONPATH to include:
# 1. ROS Python packages (for rospy, etc.)
# 2. Grounded SAM modules
export PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH"

# Add Grounded SAM repository to PYTHONPATH
if [ -d "$HOME/hsr_robocanes_omniverse/Grounded-Segment-Anything" ]; then
    export PYTHONPATH="$HOME/hsr_robocanes_omniverse/Grounded-Segment-Anything:$PYTHONPATH"
fi

# Debug: Print environment variables to verify they're set
if [ "${DEBUG_GROUNDED_SAM:-0}" = "1" ] || [ -n "$ROS_MASTER_URI" ]; then
    echo "=== Grounded SAM Segmentation Environment Debug ==="
    echo "CONDA_PREFIX: $CONDA_PREFIX"
    echo "PYTHONPATH: $PYTHONPATH"
    echo "===================================================="
fi

# Run the node
exec python3 "$PACKAGE_PATH/scripts/grounded_sam_segmentation_node.py" "$@"

