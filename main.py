import open3d as o3d
import numpy as np

mesh = o3d.io.read_triangle_mesh("hailpad_normalized.stl")

print("Computing normal and rendering it.")
mesh.compute_vertex_normals()
print(np.asarray(mesh.triangle_normals))

mesh.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([mesh])