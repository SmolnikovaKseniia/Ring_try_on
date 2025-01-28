import cv2
import mediapipe as mp
import os
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

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

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            draw_points_on_image(frame, hand_landmarks.landmark)

        # Process the landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        height, width, _ = frame.shape
        landmarks_list = []

        for lm in hand_landmarks.landmark:
            x_px, y_px = int(lm.x * width), int(lm.y * height)
            z_value = lm.z  # Z-coordinate from Mediapipe
            landmarks_list.append([x_px, y_px, z_value])

        # Calculate ring coordinates (using thumb base and tip as an example)
        ring_coordinates = [
            (landmarks_list[4][0] + landmarks_list[5][0]) / 2,  # Midpoint between thumb base and tip
            (landmarks_list[4][1] + landmarks_list[5][1]) / 2,  # Midpoint Y
        ]

        # Draw ring position
        draw_ring_position(frame, ring_coordinates)

    return frame

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 for the default webcam

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Process the frame to detect landmarks and draw the ring
        frame = process_frame(frame)

        # Display the processed frame
        cv2.imshow("Hand Tracking", frame)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
