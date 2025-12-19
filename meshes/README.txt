# Mesh Files Directory

This directory contains object mesh files organized by object name, following YCB dataset conventions.

## Folder Structure

Each object has its own folder containing:
- `textured.obj` - Main textured mesh file (YCB standard, preferred)
- `textured.mtl` - Material file (referenced by textured.obj)
- `texture_map.png` - Texture image (referenced by textured.mtl)
- `mesh.obj` - Alternative mesh file name (fallback)
- Other formats: `.ply`, `.stl`, `.dae` (not used by FoundationPose)

## Current Objects

- `mustard_bottle/` - Mustard bottle mesh
- `cracker_box/` - Cracker box mesh (YCB format with textures)

## Usage

The FoundationPose node will automatically find mesh files based on the `object_name` parameter.
It tries these names in order:
1. `meshes/{object_name}/textured.obj` (YCB standard)
2. `meshes/{object_name}/textured_simple.obj` (FoundationPose demo naming)
3. `meshes/{object_name}/mesh.obj` (generic name)
4. `meshes/{object_name}/{object_name}.obj` (object name as filename)

You can also specify the full path in the config file using the `mesh_file` parameter.

## Texture Files

If using `textured.obj`, ensure:
- `textured.mtl` is in the same folder
- Texture images (referenced in `.mtl`) are in the same folder
- All paths in `.mtl` are relative to the folder
