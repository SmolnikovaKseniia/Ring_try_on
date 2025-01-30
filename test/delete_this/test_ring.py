import trimesh
import pyrender
import matplotlib.pyplot as plt

# Load the 3D ring model
ring_mesh = trimesh.load("data/models/ring/ring.obj")

# Create a scene and add the ring mesh
scene = pyrender.Scene()
mesh = pyrender.Mesh.from_trimesh(ring_mesh)
scene.add(mesh)

# Render the ring
renderer = pyrender.OffscreenRenderer(400, 400)
color, _ = renderer.render(scene)

# Show the rendered 3D ring
plt.imshow(color)
plt.axis("off")
plt.show()
