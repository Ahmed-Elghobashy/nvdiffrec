import bpy
import os

# Optional: set your output folder
output_dir = "/home/elghobashy/relight/baseline/nvdiffrec/data/lion_head"
os.makedirs(output_dir, exist_ok=True)
obj_path = os.path.join(output_dir, "mesh.obj")

# Select all
bpy.ops.object.select_all(action='SELECT')

# Export as OBJ
bpy.ops.export_scene.obj(
    filepath=obj_path,
    use_selection=False,
    use_materials=True,
    use_mesh_modifiers=True,
    use_normals=True,
    use_uvs=True,
    path_mode='COPY',       
    axis_forward='-Z',
    axis_up='Y'
)

print(f"Exported to {obj_path}")
