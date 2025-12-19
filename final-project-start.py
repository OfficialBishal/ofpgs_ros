import os
import signal
import subprocess
import time
import json
import argparse
import shlex

from process_handler import process_handler

PACKAGE_NAME = "ofpgs_ros"
PACKAGE_PATH = subprocess.getoutput(f"rospack find {PACKAGE_NAME}")
print(f"Package Path: {PACKAGE_PATH}")

if 'error' in PACKAGE_PATH.lower():
    print(f'source and run program again!')
    exit(1)

LAUNCH_NAME = 'robocanes_hsr_correction_sim.launch'
# Use launch file from hsr_isaac_localization package (same as assignment6)
HSR_ISAAC_LOCALIZATION_PKG = "hsr_isaac_localization"
HSR_ISAAC_LOCALIZATION_PATH = subprocess.getoutput(f"rospack find {HSR_ISAAC_LOCALIZATION_PKG}")
if 'error' in HSR_ISAAC_LOCALIZATION_PATH.lower():
    print(f'Could not find {HSR_ISAAC_LOCALIZATION_PKG} package')
    exit(1)
LAUNCH_PATH = os.path.join(HSR_ISAAC_LOCALIZATION_PATH, 'launch', LAUNCH_NAME)
print(f'roslaunch path:', LAUNCH_PATH)

ISAAC_SIM_NAME = "isaac_sim"
ISAAC_SIM_PATH = subprocess.getoutput(f"rospack find {ISAAC_SIM_NAME}")
print(f"Isaac Sim path: {ISAAC_SIM_PATH}")

if 'error' in ISAAC_SIM_PATH.lower():
    print(f'Could not find {ISAAC_SIM_NAME} package')
    exit(1)

ISAAC_SIM_PYTHON_NAME = 'python.sh'
ISAAC_SIM_PYTHON_PATH = os.path.join(ISAAC_SIM_PATH, ISAAC_SIM_PYTHON_NAME)
print(f'Isaac Sim python path:', ISAAC_SIM_PYTHON_PATH)

# Use final-project-world.py (copy of assignment6 world for modifications)
# World file is now in the final-OfficialBishal package
ISAAC_WORLD_NAME = 'final-project-world.py'
ISAAC_WORLD_PATH = os.path.join(PACKAGE_PATH, ISAAC_WORLD_NAME)
print(f'Isaac Sim world path:', ISAAC_WORLD_PATH)

# Use RViz config from final-OfficialBishal package
RVIZ_NAME = 'hsr.rviz'
RVIZ_PATH = os.path.join(PACKAGE_PATH, 'rviz', RVIZ_NAME)
print(f'RVIZ path:', RVIZ_PATH)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Start Isaac Sim with final-project world")
parser.add_argument("--headless", action="store_true", 
                    help="Run Isaac Sim in headless mode (no GUI, saves GPU memory)")
parser.add_argument("--width", type=int, default=1280, help="Window width (ignored in headless)")
parser.add_argument("--height", type=int, default=720, help="Window height (ignored in headless)")
parser.add_argument("--renderer", type=str, default="RayTracedLighting",
                    choices=["RayTracedLighting", "PathTracing", "HydraStorm"],
                    help="Renderer for Isaac Sim")
args = parser.parse_args()

# Build sim config (headless mode reduces GPU memory usage significantly)
sim_config = {
    "width": args.width,
    "height": args.height,
    "sync_loads": True,
    "headless": args.headless,
    "renderer": args.renderer,
}
sim_config_str = shlex.quote(json.dumps(sim_config))

print(f"Isaac Sim config: {sim_config}")
if args.headless:
    print("=" * 60)
    print("RUNNING IN HEADLESS MODE - No GUI, reduced GPU memory usage")
    print("=" * 60)

# Commands to start processes
roslaunch_cmd = f"roslaunch {LAUNCH_PATH}"
isaac_sim_cmd = f"{ISAAC_SIM_PYTHON_PATH} {ISAAC_WORLD_PATH} --sim_app_config {sim_config_str}"
rviz_cmd = f"rviz -d {RVIZ_PATH}"

# Start processes
processes = {
    "roslaunch": process_handler.start_process(roslaunch_cmd, capture_output=False),
    "isaac_sim": process_handler.start_process(isaac_sim_cmd, capture_output=True),
    "rviz": process_handler.start_process(rviz_cmd, capture_output=False)
}

signal.signal(signal.SIGINT, lambda signum, frame: process_handler.cleanup_exit(processes))
signal.signal(signal.SIGTERM, lambda signum, frame: process_handler.cleanup_exit(processes))

# Wait indefinitely
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    process_handler.cleanup_exit(processes=processes)

