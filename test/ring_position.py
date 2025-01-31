import cv2
import numpy as np

def get_ring_position(landmarks_list, shift_factor=0.8):
    index_mcp = landmarks_list[5]  
    index_pip = landmarks_list[6] 

    ring_coordinates = [
        (index_mcp.x * (1 - shift_factor) + index_pip.x * shift_factor),  # X coordinate
        (index_mcp.y * (1 - shift_factor) + index_pip.y * shift_factor)   # Y coordinate
    ]
    
    return ring_coordinates


def draw_ring_position(image, ring_coordinates):
    height, width, _ = image.shape
    x = int(ring_coordinates[0] * width)  
    y = int(ring_coordinates[1] * height)  
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  

HAND_CANONICAL_MODEL_3D = np.array([
    [0.0, 0.0, 0.0],    
    [0.0, 0.05, 0.0],   
    [0.03, 0.07, 0.05], 
    [0.05, 0.09, 0.05], 
    [0.07, 0.11, 0.05], 
])


def get_2d_points(landmarks_list):
    points = []
    for idx in [0, 5, 9, 13, 17]:
        x = landmarks_list[idx].x
        y = landmarks_list[idx].y
        points.append([x, y])
    return np.array(points)

def estimate_ring_pose(image, landmarks_list):
    image_points = get_2d_points(landmarks_list)
    
    object_points = HAND_CANONICAL_MODEL_3D

    focal_length = 1  
    center = (image.shape[1] / 2, image.shape[0] / 2)  
    camera_matrix = np.array([[focal_length, 0, center[0]], 
                              [0, focal_length, center[1]], 
                              [0, 0, 1]], dtype=np.float32)

    
    dist_coeffs = np.zeros((4, 1))  # No distortion
    
    success, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

    if success:
        print("Pose estimation successful!")
    else:
        print("Pose estimation failed!")
    
    return rotation_vector, translation_vector

def transform_ring_position(rotation_vector, translation_vector, ring_position_3d):
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    camera_position = np.dot(rvec_matrix, ring_position_3d.T) + translation_vector
    return camera_position