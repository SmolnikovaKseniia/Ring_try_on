import bpy
from mathutils import Matrix
import math

# Clear the current scene (optional if needed)
bpy.ops.wm.read_factory_settings(use_empty=True)

# Load the preconfigured `.blend` file
blend_file_path = "test/blender_api/ring_light_camera1.blend"
bpy.ops.wm.open_mainfile(filepath=blend_file_path)

# OPTIONAL: Get the ring object if needed
ring = bpy.data.objects.get("Ring")  # Replace "Ring" with the actual name of your ring object
light = bpy.data.objects.get("Sun")
light1 = bpy.data.objects.get("Sun.001")
light2 = bpy.data.objects.get("Sun.002")
camera = bpy.data.objects.get("Camera")

angle = math.radians(180)
camera.rotation_euler = (angle, 0, 0)  # Replace with actual ring pose rotation
camera.location = (0, 0, 0)

# Available objects in the scene: ['Camera', 'Ring', 'Sun', 'Sun.001', 'Sun.002']
# light.rotation_euler = (1.5708, 0, 0)
# light.data.energy = 1000

light.rotation_euler = (1.5708, 0, 0)
light.data.energy = 1000
light.location = (0, -10, 0)

light1.rotation_euler = (5, -10, 0)
light1.data.energy = 1000
light1.location = (0, -10, 0)

light2.rotation_euler = (-5, -10, 0)
light2.data.energy = 1000
light2.location = (0, -10, 0)

constraint = light1.constraints.new(type='TRACK_TO')
constraint.target = ring  # Кільце як ціль
constraint.track_axis = 'TRACK_NEGATIVE_Z'  # Світло буде направлене вперед (-Z)
constraint.up_axis = 'UP_Y'  # Вісь вгору (щоб світло не переверталось)

# Робимо, щоб світло було спрямоване на кільце
constraint = light2.constraints.new(type='TRACK_TO')
constraint.target = ring  # Кільце як ціль
constraint.track_axis = 'TRACK_NEGATIVE_Z'  # Світло буде направлене вперед (-Z)
constraint.up_axis = 'UP_Y'  # Вісь вгору (щоб світло не переверталось)


if ring:
    ring.scale = (0.5, 0.5, 0.5)  # Adjust if necessary

# hand 0
xAxis = [-1.44126935e-01, -9.73194820e-01, -1.79218494e-01]
yAxis = [-9.88831371e-01,  1.34694243e-01,  6.37963973e-02]
zAxis = [-3.79466240e-02,  1.86411648e-01, -9.81738637e-01]

rotation_matrix = Matrix([
            [xAxis[0], yAxis[0], zAxis[0]],  # First row
            [xAxis[1], yAxis[1], zAxis[1]],  # Second row
            [xAxis[2], yAxis[2], zAxis[2]]  # Third row
])

euler_angles = rotation_matrix.to_euler()
ring.location = (6.80980694, 21.02022489, 330.20905856)
ring.rotation_euler = euler_angles

# hand 1

# Enable transparent background
bpy.context.scene.render.film_transparent = True

# Set render resolution
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1440

# Set render output file format to PNG with transparency
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.image_settings.color_mode = 'RGBA'  # Includes alpha
bpy.context.scene.render.image_settings.color_depth = '16'

# Set render output file
bpy.context.scene.render.filepath = "final_results/ring_renders/hand0render.png"

# Render the image
bpy.ops.render.render(write_still=True)

# print("Available objects in the scene:", [obj.name for obj in bpy.data.objects])
