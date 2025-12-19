#!/usr/bin/env python3
"""
Headless start script for final project - runs Isaac Sim without GUI to save GPU memory.
This is useful when running FoundationPose which requires significant GPU memory.
"""

import os
import subprocess
import sys

# Import the main start script and run it with --headless flag
PACKAGE_PATH = os.path.dirname(os.path.abspath(__file__))
START_SCRIPT = os.path.join(PACKAGE_PATH, 'final-project-start.py')

# Run the main script with --headless flag
os.execv(sys.executable, [sys.executable, START_SCRIPT, '--headless'])

