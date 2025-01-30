import bpy
import math
from mathutils import Vector

# Clear the Blender scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Set up rendering
bpy.context.scene.render.engine = 'CYCLES'  # Use Cycles for realistic rendering
bpy.context.scene.cycles.device = 'CPU'
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.resolution_percentage = 100

# Load the hand image as the background
image_path = "data/images/original_0.png"  # Replace with your hand image path
bpy.ops.import_image.to_plane(files=[{"name": image_path}], directory="")
bg_image = bpy.context.object
bg_image.location = (0, 0, 0)
bg_image.scale = (8, 8, 8)  # Adjust scale to match the image proportions

# Load the ring model
ring_model_path = "data/models/ring/ring.obj"  # Replace with your ring model path
bpy.ops.import_scene.obj(filepath=ring_model_path)
ring = bpy.context.selected_objects[0]
ring.scale = (0.02, 0.02, 0.02)  # Adjust scale as needed

# Position the ring based on Mediapipe landmarks
# Example position (you need to replace with Mediapipe-calculated coordinates)
hand_landmark_x = 0.5  # Normalized x (from Mediapipe landmark)
hand_landmark_y = 0.3  # Normalized y (from Mediapipe landmark)

# Convert normalized coordinates to Blender's space
image_width, image_height = 1920, 1080  # Set the resolution of the background image
ring_x = (hand_landmark_x - 0.5) * image_width * 0.01  # Adjust scaling
ring_y = (hand_landmark_y - 0.5) * image_height * 0.01
ring.location = Vector((ring_x, ring_y, 0.2))  # Slight offset in Z to sit above the image

# Add lighting
light_data = bpy.data.lights.new(name="light", type="AREA")
light_data.energy = 500
light_object = bpy.data.objects.new(name="light", object_data=light_data)
bpy.context.collection.objects.link(light_object)
light_object.location = (0, 0, 5)  # Position above the scene

# Set up the camera
camera = bpy.data.objects["Camera"]
camera.location = (0, 0, 10)  # Position the camera
camera.rotation_euler = (math.radians(90), 0, math.radians(180))  # Face the plane

# Render the scene
output_path = "data/output/ring_on_hand.png"  # Replace with your desired output path
bpy.context.scene.render.filepath = output_path
bpy.ops.render.render(write_still=True)

print(f"Rendered image saved to {output_path}")
