import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def get_hand_landmarks(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks[0].landmark
    return None

def convert_landmarks_to_2d(landmarks, width, height):
    return np.array([
        [landmarks[i].x * width, landmarks[i].y * height] for i in [0, 1, 2, 4, 5, 6, 7, 8]
    ], dtype=np.float32)

def estimate_camera_pose(image, landmarks):
    height, width, _ = image.shape
    
    # 3D Model of the hand (Assumed values in mm)
    object_points = np.array([
        [0, 0, 0], [0, -5, -2], [3, -10, -3], [5, -14, -4],
        [2, -5, -1], [3, -10, -1], [3.5, -14, 0], [4, -16, 0]
    ], dtype=np.float32)
    
    # Convert normalized landmarks to 2D image coordinates
    image_points = convert_landmarks_to_2d(landmarks, width, height)
    
    # Camera intrinsic matrix (Assuming no distortion)
    focal_length = width
    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs
    )
    
    if success:
        return rotation_vector, translation_vector
    return None, None

def main():
    input_image_path = "C:\\Users\\Ksena\\Documents\\Ring_try_on\\test\\original_6.png"
    image = cv2.imread(input_image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    
    landmarks = get_hand_landmarks(image)
    if landmarks:
        rotation_vector, translation_vector = estimate_camera_pose(image, landmarks)
        if rotation_vector is not None:
            print("Rotation Vector:\n", rotation_vector)
            print("Translation Vector:\n", translation_vector)
    else:
        print("No hand detected.")

if __name__ == "__main__":
    main()
