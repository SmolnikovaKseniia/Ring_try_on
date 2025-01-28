import open3d as o3d

# Load the 3D mesh file
mesh = o3d.io.read_triangle_mesh("data/models/ring/ring.obj")

# Visualize the mesh
o3d.visualization.draw_geometries([mesh])
