#!/usr/bin/env python3
"""
Temporary patch to bypass PyTorch's CUDA version check during Grounding DINO installation.
This allows PyTorch 1.13.1 (CUDA 11.6) to compile CUDA extensions with CUDA 12.8.
"""

import sys
import os
import re

# Find PyTorch installation
try:
    import torch
    torch_path = os.path.dirname(torch.__file__)
    cpp_extension_path = os.path.join(torch_path, 'utils', 'cpp_extension.py')
    
    if not os.path.exists(cpp_extension_path):
        print(f"ERROR: Could not find {cpp_extension_path}")
        sys.exit(1)
    
    # Read the file
    with open(cpp_extension_path, 'r') as f:
        lines = f.readlines()
    
    # Check if already patched
    content_str = ''.join(lines)
    if '# PATCHED: CUDA version check bypassed' in content_str:
        print("PyTorch cpp_extension.py already patched")
        sys.exit(0)
    
    # Create backup
    backup_path = cpp_extension_path + '.backup'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            f.writelines(lines)
        print(f"Created backup: {backup_path}")
    
    # Find and patch the raise statement
    new_lines = []
    patched = False
    
    for i, line in enumerate(lines):
        # Look for the raise RuntimeError with CUDA_MISMATCH_MESSAGE
        if 'raise RuntimeError' in line and 'CUDA_MISMATCH_MESSAGE' in line:
            # Get indentation
            indent = len(line) - len(line.lstrip())
            # Replace with return statement
            new_lines.append(' ' * indent + '# PATCHED: CUDA version check bypassed for CUDA 12.8 compatibility\n')
            new_lines.append(' ' * indent + '# ' + line.lstrip())  # Comment out original
            new_lines.append(' ' * indent + 'return  # Always pass the check\n')
            patched = True
        else:
            new_lines.append(line)
    
    if not patched:
        print("WARNING: Could not find raise RuntimeError(CUDA_MISMATCH_MESSAGE) to patch")
        print("Trying alternative method...")
        
        # Alternative: patch the function directly by finding the function and modifying it
        new_lines = []
        in_function = False
        function_start = -1
        
        for i, line in enumerate(lines):
            if 'def _check_cuda_version' in line:
                in_function = True
                function_start = i
                new_lines.append(line)
            elif in_function:
                if 'raise RuntimeError' in line and 'CUDA_MISMATCH_MESSAGE' in line:
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(' ' * indent + '# PATCHED: CUDA version check bypassed\n')
                    new_lines.append(' ' * indent + '# ' + line.lstrip())
                    new_lines.append(' ' * indent + 'return  # Always pass\n')
                    patched = True
                elif line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""'):
                    # Check if we've left the function (new def/class at same or less indent)
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= 4:  # Top level
                        in_function = False
                        new_lines.append(line)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
    
    
    if patched:
        # Write patched file
        with open(cpp_extension_path, 'w') as f:
            f.writelines(new_lines)
        print(f"Patched {cpp_extension_path}")
        print("CUDA version check bypassed - installation should proceed")
    else:
        print("ERROR: Could not find raise RuntimeError(CUDA_MISMATCH_MESSAGE) to patch")
        print("The file structure may have changed.")
        print("You may need to manually edit the file or install without CUDA extensions.")
        sys.exit(1)
        
except Exception as e:
    print(f"ERROR: Failed to patch PyTorch: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

