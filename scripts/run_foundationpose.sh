#!/bin/bash
# Wrapper script to run FoundationPose node
# Uses Docker's ROS1 setup + foundationpose conda environment

# Activate foundationpose conda environment FIRST (before ROS)
source /opt/conda/etc/profile.d/conda.sh
conda activate foundationpose

# Set library paths to prioritize conda libraries (fixes libffi conflicts with cv_bridge)
# This MUST be set before sourcing ROS to ensure conda libraries are found first
if [ -d "$CONDA_PREFIX/lib" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    # Force conda's libffi.so.7 to be loaded first using LD_PRELOAD
    # libp11-kit is linked against libffi.so.7, so we need that version
    if [ -f "$CONDA_PREFIX/lib/libffi.so.7" ]; then
        export LD_PRELOAD="$CONDA_PREFIX/lib/libffi.so.7:$LD_PRELOAD"
    elif [ -f "$CONDA_PREFIX/lib/libffi.so.8" ]; then
        export LD_PRELOAD="$CONDA_PREFIX/lib/libffi.so.8:$LD_PRELOAD"
    fi
fi

# Source ROS setup (required for ROS Python packages and rospack)
source /opt/ros/noetic/setup.bash

# Source ROS workspace (required for rospack find)
# Try common workspace locations
if [ -f ~/catkin_ws/devel/setup.bash ]; then
    source ~/catkin_ws/devel/setup.bash
elif [ -f ~/ros_ws/devel/setup.bash ]; then
    source ~/ros_ws/devel/setup.bash
elif [ -f ~/workspace/devel/setup.bash ]; then
    source ~/workspace/devel/setup.bash
fi

# Re-apply LD_LIBRARY_PATH and LD_PRELOAD after ROS setup (in case ROS modified it)
# Also check if LD_PRELOAD was set by launch file (roslaunch env tag)
# Ensure conda lib is FIRST in LD_LIBRARY_PATH so libp11-kit finds conda's libffi
if [ -d "$CONDA_PREFIX/lib" ]; then
    # If LD_LIBRARY_PATH was set by launch file, it might only have conda lib
    # In that case, we need to preserve it and add other paths after
    if [ -n "$LD_LIBRARY_PATH" ] && echo "$LD_LIBRARY_PATH" | grep -q "^$CONDA_PREFIX/lib"; then
        # Launch file set it with conda lib first, just add other paths
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "^$CONDA_PREFIX/lib$" | tr '\n' ':' | sed 's/:$//')"
    else
        # Remove any existing conda lib from LD_LIBRARY_PATH to avoid duplicates
        export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "^$CONDA_PREFIX/lib$" | tr '\n' ':' | sed 's/:$//')
        # Add conda lib FIRST so it's found before system libraries
        export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    fi
    
    # Use LD_PRELOAD from environment if set (from launch file), otherwise set it
    # Prefer libffi.so.7 since libp11-kit is linked against that version
    if [ -z "$LD_PRELOAD" ]; then
        if [ -f "$CONDA_PREFIX/lib/libffi.so.7" ]; then
            export LD_PRELOAD="$CONDA_PREFIX/lib/libffi.so.7"
        elif [ -f "$CONDA_PREFIX/lib/libffi.so.8" ]; then
            export LD_PRELOAD="$CONDA_PREFIX/lib/libffi.so.8"
        fi
    else
        # LD_PRELOAD already set (from launch file), but ensure conda libffi.so.7 is first
        # Remove duplicates and trailing colons
        LD_PRELOAD_CLEAN=$(echo "$LD_PRELOAD" | tr ':' '\n' | grep -v "^$CONDA_PREFIX/lib/libffi.so" | grep -v '^$' | tr '\n' ':' | sed 's/:$//')
        if [ -f "$CONDA_PREFIX/lib/libffi.so.7" ]; then
            if [ -n "$LD_PRELOAD_CLEAN" ]; then
                export LD_PRELOAD="$CONDA_PREFIX/lib/libffi.so.7:$LD_PRELOAD_CLEAN"
            else
                export LD_PRELOAD="$CONDA_PREFIX/lib/libffi.so.7"
            fi
        elif [ -f "$CONDA_PREFIX/lib/libffi.so.8" ]; then
            if [ -n "$LD_PRELOAD_CLEAN" ]; then
                export LD_PRELOAD="$CONDA_PREFIX/lib/libffi.so.8:$LD_PRELOAD_CLEAN"
            else
                export LD_PRELOAD="$CONDA_PREFIX/lib/libffi.so.8"
            fi
        elif [ -n "$LD_PRELOAD_CLEAN" ]; then
            export LD_PRELOAD="$LD_PRELOAD_CLEAN"
        fi
    fi
fi

# Get the package path using rospack
PACKAGE_PATH=$(rospack find ofpgs_ros 2>/dev/null)
if [ -z "$PACKAGE_PATH" ]; then
    echo "ERROR: Could not find ofpgs_ros package"
    exit 1
fi

# Set PYTHONPATH to include:
# 1. ROS Python packages (for rospy, etc.)
# 2. FoundationPose directory (for estimater, etc.)
# Add FoundationPose to PYTHONPATH if it exists
# Try common locations
FOUNDATIONPOSE_PATH=""
if [ -d "$HOME/catkin_ws/src/FoundationPose" ]; then
    FOUNDATIONPOSE_PATH="$HOME/catkin_ws/src/FoundationPose"
elif [ -d "$HOME/ros_ws/src/FoundationPose" ]; then
    FOUNDATIONPOSE_PATH="$HOME/ros_ws/src/FoundationPose"
elif [ -d "$HOME/workspace/src/FoundationPose" ]; then
    FOUNDATIONPOSE_PATH="$HOME/workspace/src/FoundationPose"
elif [ -d "/opt/foundationpose" ]; then
    FOUNDATIONPOSE_PATH="/opt/foundationpose"
fi

if [ -n "$FOUNDATIONPOSE_PATH" ]; then
    export PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:$FOUNDATIONPOSE_PATH:$PYTHONPATH"
else
    export PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH"
    echo "WARNING: FoundationPose not found. Please install FoundationPose and update PYTHONPATH."
fi

# CUDA Memory Management
# Set PyTorch CUDA memory allocation config to reduce fragmentation
# max_split_size_mb: Maximum size of a memory chunk that can be split (in MB)
# Lower values reduce fragmentation but may slow down allocation
# Note: expandable_segments and roundup_power2_divisions are not available in older PyTorch versions
# Using only max_split_size_mb which is supported in PyTorch 1.x
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# Debug: Print environment variables to verify they're set
# Always print these when running under roslaunch to help debug
if [ "${DEBUG_FOUNDATIONPOSE:-0}" = "1" ] || [ -n "$ROS_MASTER_URI" ]; then
    echo "=== FoundationPose Environment Debug ==="
    echo "CONDA_PREFIX: $CONDA_PREFIX"
    echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    echo "LD_PRELOAD: $LD_PRELOAD"
    echo "PYTHONPATH: $PYTHONPATH"
    echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
    # Verify libffi resolution
    if [ -f "$CONDA_PREFIX/lib/libffi.so.7" ]; then
        echo "Testing libffi resolution..."
        LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" ldd /usr/lib/x86_64-linux-gnu/libp11-kit.so.0 2>/dev/null | grep ffi || echo "Could not test libffi resolution"
    fi
    echo "========================================"
fi

# Run the node
# LD_PRELOAD and LD_LIBRARY_PATH must be exported and set before exec
# The exec command will preserve these environment variables for Python
exec python3 "$PACKAGE_PATH/scripts/foundationpose_pose_estimation_node.py" "$@"
