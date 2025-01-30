import bpy
import mathutils

# Your rotation matrix
rotation_matrix = mathutils.Matrix((
    ( 9.98803366e-01, -4.88940492e-02,  1.09935943e-03),
    ( 4.84338539e-02,  9.85785285e-01, -1.60877388e-01),
    ( 6.78221456e-03,  1.60738123e-01,  9.86973788e-01)
))

# Convert to Euler angles (XYZ order)
euler_angles = rotation_matrix.to_euler('XYZ')

print(euler_angles)
