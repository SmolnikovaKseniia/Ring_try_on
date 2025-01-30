import cv2
import numpy as np

def get_ring_position(landmarks_list, shift_factor=0.8):
    # Index finger MCP (5) and PIP (6) landmarks
    index_mcp = landmarks_list[5]  # Landmark 5: Index MCP
    index_pip = landmarks_list[6]  # Landmark 6: Index PIP

    # Calculate the position between MCP and PIP closer to PIP based on the shift_factor
    ring_coordinates = [
        (index_mcp.x * (1 - shift_factor) + index_pip.x * shift_factor),  # X coordinate
        (index_mcp.y * (1 - shift_factor) + index_pip.y * shift_factor)   # Y coordinate
    ]
    
    return ring_coordinates


# Function to draw ring position on the image
def draw_ring_position(image, ring_coordinates):
    height, width, _ = image.shape
    x = int(ring_coordinates[0] * width)  # Convert normalized x to pixel space
    y = int(ring_coordinates[1] * height)  # Convert normalized y to pixel space
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Draw the ring as a red circle

HAND_CANONICAL_MODEL_3D = np.array([
    [0.0, 0.0, 0.0],    # Landmark 0: Wrist
    [0.0, 0.05, 0.0],   # Landmark 5: Thumb CMC
    [0.03, 0.07, 0.05], # Landmark 9: Index MCP
    [0.05, 0.09, 0.05], # Landmark 13: Middle MCP
    [0.07, 0.11, 0.05], # Landmark 17: Ring MCP
])

# Corresponding 2D points in image space (obtained from hand landmarks)
def get_2d_points(landmarks_list):
    # Select landmarks for hand palm (0, 5, 9, 13, 17)
    points = []
    for idx in [0, 5, 9, 13, 17]:
        x = landmarks_list[idx].x
        y = landmarks_list[idx].y
        points.append([x, y])
    return np.array(points)

# Function to perform PnP Pose Estimation
def estimate_ring_pose(image, landmarks_list):
    # Get the 2D image coordinates of the hand palm landmarks
    image_points = get_2d_points(landmarks_list)
    
    # Define the corresponding 3D model points from the hand model (canonical model)
    object_points = HAND_CANONICAL_MODEL_3D

    # Assume the camera matrix (Intrinsic parameters)
    focal_length = 1  # Focal length (arbitrary units, should be in pixels)
    center = (image.shape[1] / 2, image.shape[0] / 2)  # Assume the image center is the principal point
    camera_matrix = np.array([[focal_length, 0, center[0]], 
                              [0, focal_length, center[1]], 
                              [0, 0, 1]], dtype=np.float32)

    # Distortion coefficients (assuming no lens distortion)
    dist_coeffs = np.zeros((4, 1))  # No distortion
    
    success, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

    if success:
        print("Pose estimation successful!")
    else:
        print("Pose estimation failed!")
    
    return rotation_vector, translation_vector

# Function to transform the ring position in the hand plane coordinates
def transform_ring_position(rotation_vector, translation_vector, ring_position_3d):
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    camera_position = np.dot(rvec_matrix, ring_position_3d.T) + translation_vector
    return camera_position