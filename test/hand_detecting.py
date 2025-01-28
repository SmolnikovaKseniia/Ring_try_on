import cv2
import mediapipe as mp
import os
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def draw_points_on_image(image, landmarks):
    height, width, _ = image.shape
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

def draw_ring_position(image, ring_coordinates):
    height, width, _ = image.shape
    x = int(ring_coordinates[0] * width)
    y = int(ring_coordinates[1] * height)
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

def process_image(input_image_path):
    image = cv2.imread(input_image_path)

    if image is None:
        print("Error: Unable to load image.")
        return None

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            draw_points_on_image(image, hand_landmarks.landmark)
        
        # Process the landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        height, width, _ = image.shape
        landmarks_list = []

        for lm in hand_landmarks.landmark:
            x_px, y_px = int(lm.x * width), int(lm.y * height)
            z_value = lm.z  # Z-coordinate from Mediapipe
            landmarks_list.append([x_px, y_px, z_value])

        print(landmarks_list)

        # Ensure ring coordinates are calculated correctly
        # Use landmarks 6 and 5 for the thumb base and tip
        ring_coordinates = [
            (landmarks_list[5][0] + landmarks_list[6][0]) / 2,  # Midpoint between thumb base and tip
            (landmarks_list[5][1] + landmarks_list[6][1]) / 2,  # Midpoint Y
        ]

        print(f"Ring coordinates: {ring_coordinates}")
        
        # Draw ring position
        draw_ring_position(image, ring_coordinates)

    else:
        print("No hand detected")

    output_path = os.path.join(os.path.expanduser("~"), "Desktop", "image_with_points.png")
    cv2.imwrite(output_path, image)
    print(f"Image saved at: {output_path}")

    return image, results

def main():
    input_image_path = "C:\\Users\\Ksena\\Documents\\Ring_try_on\\test\\original_6.png" 
    image, results = process_image(input_image_path)

if __name__ == "__main__":
    main()
