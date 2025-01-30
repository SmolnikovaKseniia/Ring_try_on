import bpy
from mathutils import Matrix
import math

def load_mesh(filepath):
    # This assumes your add-on or Blender version has 'wm.obj_import' operator.
    bpy.ops.wm.obj_import(filepath=filepath)

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import the model (modify the path to your model)
model_path = "/Users/ivanna/Documents/MLWeek/Ring_try_on/data/models/ring/ring.obj"
load_mesh(model_path)

# Get the imported ring object 
ring = bpy.context.selected_objects[0]
ring.scale = (0.1, 0.1, 0.1)  # Adjust these values for desired size

# Create a new camera
bpy.ops.object.camera_add()
camera = bpy.context.object

xAxis = [-1.44126935e-01, -9.73194820e-01, -1.79218494e-01]
yAxis = [-9.88831371e-01,  1.34694243e-01,  6.37963973e-02]
zAxis = [-3.79466240e-02,  1.86411648e-01, -9.81738637e-01]

rotation_matrix = Matrix([
            [xAxis[0], yAxis[0], zAxis[0]],  # First row
            [xAxis[1], yAxis[1], zAxis[1]],  # Second row
            [xAxis[2], yAxis[2], zAxis[2]]  # Third row
])

euler_angles = rotation_matrix.to_euler()
bpy.data.objects["ring"].location = (6.80980694,  21.02022489, 330.20905856)
bpy.data.objects["ring"].rotation_euler = euler_angles

angle = math.radians(180)

# Set camera position (modify according to the pose)
camera.rotation_euler = (angle, 0, 0)  # Replace with actual ring pose rotation
camera.location = (0, 0, 0)

# Set as the active camera
bpy.context.scene.camera = camera

# Add a light source
bpy.ops.object.light_add(type='SUN', location=(0, -10, 0))
light = bpy.context.object
light.rotation_euler = (1.5708, 0, 0)
light.data.energy = 1000

# Add a light source
bpy.ops.object.light_add(type='SUN', location=(5, -10, 0))
light2 = bpy.context.object
light2.rotation_euler = (1.5708, 0, 0)
light2.data.energy = 1000

# Add a light source
bpy.ops.object.light_add(type='SUN', location=(-5, -10, 0))
light3 = bpy.context.object
light3.rotation_euler = (1.5708, 0, 0)
light3.data.energy = 1000

# Робимо, щоб світло було спрямоване на кільце
constraint = light2.constraints.new(type='TRACK_TO')
constraint.target = ring  # Кільце як ціль
constraint.track_axis = 'TRACK_NEGATIVE_Z'  # Світло буде направлене вперед (-Z)
constraint.up_axis = 'UP_Y'  # Вісь вгору (щоб світло не переверталось)

# Робимо, щоб світло було спрямоване на кільце
constraint = light3.constraints.new(type='TRACK_TO')
constraint.target = ring  # Кільце як ціль
constraint.track_axis = 'TRACK_NEGATIVE_Z'  # Світло буде направлене вперед (-Z)
constraint.up_axis = 'UP_Y'  # Вісь вгору (щоб світло не переверталось)


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
bpy.context.scene.render.filepath = "/Users/ivanna/Documents/MLWeek/Ring_try_on/results/ring_render6.png"

# Render the image
bpy.ops.render.render(write_still=True)