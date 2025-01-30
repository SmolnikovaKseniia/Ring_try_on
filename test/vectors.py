# import cv2
# import mediapipe as mp
# import numpy as np

# # Initialize Mediapipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# def load_image(image_path):
#     """Load and convert the image to RGB."""
#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return image, image_rgb

# def detect_hands(image_rgb):
#     """Detect hands and landmarks in the image."""
#     with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
#         return hands.process(image_rgb)

# def extract_landmarks(results):
#     """Extract and return hand landmarks."""
#     hand_landmarks = []
#     if results.multi_hand_landmarks:
#         for hand in results.multi_hand_landmarks:
#             for idx, landmark in enumerate(hand.landmark):
#                 hand_landmarks.append([landmark.x, landmark.y, landmark.z])
#     return np.array(hand_landmarks)

# def load_depth_data(depth_file_path):
#     """Load and resize depth data."""
#     depth_data = np.loadtxt(depth_file_path, delimiter=",")
#     resized_depth_data = cv2.resize(depth_data, (1920, 1440), interpolation=cv2.INTER_CUBIC)
#     return resized_depth_data * 1000  # Convert from mm to meters

# def load_intrinsics(calibration_file_path):
#     """Load the intrinsic matrix from the calibration file."""
#     with open(calibration_file_path, "r") as f:
#         lines = f.readlines()[1:4]
#     return np.array([list(map(float, line.split(','))) for line in lines])

# def calculate_3d_landmarks(hand_landmarks, resized_depth_data, intrinsics):
#     """Calculate 3D landmarks using depth data and intrinsic parameters."""
#     fx, fy, cx, cy = intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]
#     h, w = resized_depth_data.shape
#     landmarks_3d = []
#     for landmark in hand_landmarks:
#         xL, yL = int(landmark[0] * w), int(landmark[1] * h)
#         depth_xy = resized_depth_data[yL, xL]
#         x3d = (xL - cx) * depth_xy / fx
#         y3d = (yL - cy) * depth_xy / fy
#         z3d = depth_xy
#         landmarks_3d.append([x3d, y3d, z3d])
#     return np.array(landmarks_3d)

# def calculate_ring_finger_position(landmarks_3d):
#     """Calculate the ring finger position based on 3D landmarks."""
#     landmark5, landmark6, landmark9 = landmarks_3d[5], landmarks_3d[6], landmarks_3d[9]
#     ring_position = (landmark5 + landmark6) / 2
#     return ring_position, landmark5, landmark6, landmark9

# def calculate_axes(landmark5, landmark6, landmark9):
#     """Calculate the direction of axes for coordinate system."""
#     x_axis_direction = (landmark6 - landmark5)
#     x_axis = x_axis_direction / np.linalg.norm(x_axis_direction)

#     y_axis_direction = (landmark9 - landmark5)
#     y_axis = y_axis_direction / np.linalg.norm(y_axis_direction)

#     z_axis_direction = np.cross(x_axis_direction, y_axis_direction)
#     z_axis = z_axis_direction / np.linalg.norm(z_axis_direction)

#     # Recompute yAxis to be perpendicular to xAxis
#     y_axis_new_direction = np.cross(z_axis, x_axis)
#     y_axis = y_axis_new_direction / np.linalg.norm(y_axis_new_direction)

#     return x_axis, y_axis, z_axis

# def calculate_transform_matrix(ring_position, x_axis, y_axis, z_axis):
#     """Calculate the transformation matrix based on ring position and axes."""
#     transform_matrix = np.eye(4)
#     transform_matrix[:3, 0] = y_axis
#     transform_matrix[:3, 1] = x_axis
#     transform_matrix[:3, 2] = -1 * z_axis
#     transform_matrix[:3, 3] = ring_position
#     return transform_matrix

# def main():
#     # File paths
#     image_path = "C:\\Users\\Ksena\\Documents\\Ring_try_on\\data\\ring_try_on_input_data\\ring_try_on_input_data\\images\\original_0.png"
#     depth_file_path = "C:\\Users\\Ksena\\Documents\\Ring_try_on\\data\\ring_try_on_input_data\\ring_try_on_input_data\\images\\depth_logs_0.txt"
#     calibration_file_path = "C:\\Users\\Ksena\\Documents\\Ring_try_on\\data\\ring_try_on_input_data\\ring_try_on_input_data\\images\\depth_calibration_logs_0.txt"

#     # Load image and process it
#     image, image_rgb = load_image(image_path)
#     results = detect_hands(image_rgb)

#     # Extract hand landmarks
#     hand_landmarks = extract_landmarks(results)

#     # Load and process depth data
#     resized_depth_data = load_depth_data(depth_file_path)

#     # Load intrinsic parameters
#     intrinsics = load_intrinsics(calibration_file_path)

#     # Calculate 3D landmarks
#     landmarks_3d = calculate_3d_landmarks(hand_landmarks, resized_depth_data, intrinsics)

#     # Calculate ring finger position and axes
#     ring_position, landmark5, landmark6, landmark9 = calculate_ring_finger_position(landmarks_3d)
#     x_axis, y_axis, z_axis = calculate_axes(landmark5, landmark6, landmark9)

#     # Calculate the transformation matrix and camera pose
#     transform_matrix = calculate_transform_matrix(ring_position, x_axis, y_axis, z_axis)
#     ring_position_homogeneous = np.append(ring_position, 1)
#     camera_pose = np.matmul(transform_matrix, ring_position_homogeneous)

#     translation_vector = transform_matrix[:3, 3]

#     rotation_matrix = transform_matrix[:3, :3]

# # Convert the rotation matrix to a rotation vector
#     rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
#     print("Вектор ротації",rotation_vector)
#     print("Вектор трансляції",translation_vector)

# if __name__ == "__main__":
#     main()
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def load_image(image_path):
    """Load and convert the image to RGB."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, image_rgb

def detect_hands(image_rgb):
    """Detect hands and landmarks in the image."""
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        return hands.process(image_rgb)

def extract_landmarks(results):
    """Extract and return hand landmarks."""
    hand_landmarks = []
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand.landmark):
                hand_landmarks.append([landmark.x, landmark.y, landmark.z])
    return np.array(hand_landmarks)

def load_depth_data(depth_file_path):
    """Load and resize depth data."""
    depth_data = np.loadtxt(depth_file_path, delimiter=",")
    resized_depth_data = cv2.resize(depth_data, (1920, 1440), interpolation=cv2.INTER_CUBIC)
    return resized_depth_data * 1000  # Convert from mm to meters

def load_intrinsics(calibration_file_path):
    """Load the intrinsic matrix from the calibration file."""
    with open(calibration_file_path, "r") as f:
        lines = f.readlines()[1:4]
    return np.array([list(map(float, line.split(','))) for line in lines])

def calculate_3d_landmarks(hand_landmarks, resized_depth_data, intrinsics):
    """Calculate 3D landmarks using depth data and intrinsic parameters."""
    fx, fy, cx, cy = intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]
    h, w = resized_depth_data.shape
    landmarks_3d = []
    for landmark in hand_landmarks:
        xL, yL = int(landmark[0] * w), int(landmark[1] * h)
        depth_xy = resized_depth_data[yL, xL]
        x3d = (xL - cx) * depth_xy / fx
        y3d = (yL - cy) * depth_xy / fy
        z3d = depth_xy
        landmarks_3d.append([x3d, y3d, z3d])
    return np.array(landmarks_3d)

def calculate_ring_finger_position(landmarks_3d):
    """Calculate the ring finger position based on 3D landmarks."""
    landmark5, landmark6, landmark9 = landmarks_3d[5], landmarks_3d[6], landmarks_3d[9]
    ring_position = (landmark5 + landmark6) / 2
    return ring_position, landmark5, landmark6, landmark9

def calculate_axes(landmark5, landmark6, landmark9):
    """Calculate the direction of axes for coordinate system."""
    x_axis_direction = (landmark6 - landmark5)
    x_axis = x_axis_direction / np.linalg.norm(x_axis_direction)

    y_axis_direction = (landmark9 - landmark5)
    y_axis = y_axis_direction / np.linalg.norm(y_axis_direction)

    z_axis_direction = np.cross(x_axis_direction, y_axis_direction)
    z_axis = z_axis_direction / np.linalg.norm(z_axis_direction)

    # Recompute yAxis to be perpendicular to xAxis
    y_axis_new_direction = np.cross(z_axis, x_axis)
    y_axis = y_axis_new_direction / np.linalg.norm(y_axis_new_direction)

    return x_axis, y_axis, z_axis

def calculate_transform_matrix(ring_position, x_axis, y_axis, z_axis):
    """Calculate the transformation matrix based on ring position and axes."""
    transform_matrix = np.eye(4)
    transform_matrix[:3, 0] = y_axis
    transform_matrix[:3, 1] = x_axis
    transform_matrix[:3, 2] = -1 * z_axis
    transform_matrix[:3, 3] = ring_position
    return transform_matrix

def extract_rotation_translation(transform_matrix):
    """Extract rotation vector, translation vector, and camera position from the transform matrix."""
    translation_vector = transform_matrix[:3, 3]
    rotation_matrix = transform_matrix[:3, :3]

    # Convert the rotation matrix to a rotation vector
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)

    # Camera position is simply the negative of the rotation matrix inverse multiplied by the translation
    camera_position = -np.linalg.inv(rotation_matrix).dot(translation_vector)

    return rotation_vector, translation_vector, camera_position

def main():
    # File paths
    image_path = "C:\\Users\\Ksena\\Documents\\Ring_try_on\\data\\ring_try_on_input_data\\ring_try_on_input_data\\images\\original_0.png"
    depth_file_path = "C:\\Users\\Ksena\\Documents\\Ring_try_on\\data\\ring_try_on_input_data\\ring_try_on_input_data\\images\\depth_logs_0.txt"
    calibration_file_path = "C:\\Users\\Ksena\\Documents\\Ring_try_on\\data\\ring_try_on_input_data\\ring_try_on_input_data\\images\\depth_calibration_logs_0.txt"

    # Load image and process it
    image, image_rgb = load_image(image_path)
    results = detect_hands(image_rgb)

    # Extract hand landmarks
    hand_landmarks = extract_landmarks(results)

    # Load and process depth data
    resized_depth_data = load_depth_data(depth_file_path)

    # Load intrinsic parameters
    intrinsics = load_intrinsics(calibration_file_path)

    # Calculate 3D landmarks
    landmarks_3d = calculate_3d_landmarks(hand_landmarks, resized_depth_data, intrinsics)

    # Calculate ring finger position and axes
    ring_position, landmark5, landmark6, landmark9 = calculate_ring_finger_position(landmarks_3d)
    x_axis, y_axis, z_axis = calculate_axes(landmark5, landmark6, landmark9)

    # Calculate the transformation matrix
    transform_matrix = calculate_transform_matrix(ring_position, x_axis, y_axis, z_axis)

    # Extract rotation and translation
    rotation_vector, translation_vector, camera_position = extract_rotation_translation(transform_matrix)

    print("Rotation vector:", rotation_vector)
    print("Translation vector:", translation_vector)
    print("Camera position:", camera_position)

if __name__ == "__main__":
    main()
