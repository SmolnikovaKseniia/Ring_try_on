import cv2
import mediapipe as mp
import numpy as np
import json

# Load Mediapipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Load the hand image
image_path = "data/images/original_0.png"  # Replace with your actual hand image path
hand_image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)

# Process the image to detect hand landmarks
results = hands.process(rgb_image)

if results.multi_hand_landmarks:
    hand_landmarks = results.multi_hand_landmarks[0]
    base = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]  # Base of ring finger
    tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]  # Tip of ring finger

    # Convert normalized coordinates to pixel coordinates
    base_coords = [int(base.x * hand_image.shape[1]), int(base.y * hand_image.shape[0])]
    tip_coords = [int(tip.x * hand_image.shape[1]), int(tip.y * hand_image.shape[0])]

    # Save the coordinates to a JSON file
    coordinates = {
        "base": base_coords,
        "tip": tip_coords,
        "image_dimensions": [hand_image.shape[1], hand_image.shape[0]]
    }
    with open("test/coordinates.json", "w") as f:
        json.dump(coordinates, f)
    print("Coordinates saved to coordinates.json")
else:
    print("No hand detected in the image.")
