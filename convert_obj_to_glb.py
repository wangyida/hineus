"""
Simple OBJ to GLB converter using trimesh
"""

import trimesh
import os

# Load the OBJ file
obj_path = 'n2m_test/mesh_0.obj'
glb_path = 'n2m_test/mesh_0.glb'

print(f"Loading OBJ file: {obj_path}")
mesh = trimesh.load(obj_path)

print(f"Mesh info:")
print(f"  Vertices: {len(mesh.vertices)}")
print(f"  Faces: {len(mesh.faces)}")
print(f"  Has UV: {mesh.visual.uv is not None}")

# Export to GLB
print(f"\nExporting to GLB: {glb_path}")
mesh.export(glb_path, file_type='glb')

print(f"Successfully exported to {glb_path}")
print(f"GLB file size: {os.path.getsize(glb_path) / 1024 / 1024:.2f} MB")