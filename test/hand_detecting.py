import cv2
import mediapipe as mp
import os
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Function to draw points on the image
def draw_points_on_image(image, landmarks):
    height, width, _ = image.shape
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

# Function to draw ring position on the image
def draw_ring_position(image, ring_coordinates):
    height, width, _ = image.shape
    x = int(ring_coordinates[0] * width)  # Convert normalized x to pixel space
    y = int(ring_coordinates[1] * height)  # Convert normalized y to pixel space
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Draw the ring as a red circle

# Function to get the ring position between MCP and PIP of the index finger
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

# Function to process the image and detect landmarks
def process_image(input_image_path):
    image = cv2.imread(input_image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    # Convert image to RGB for MediaPipe processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        landmarks_list = []
        # Process each detected hand's landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            draw_points_on_image(image, hand_landmarks.landmark)
            # Convert hand landmarks to 2D coordinates
            height, width, _ = image.shape
            for lm in hand_landmarks.landmark:
                landmarks_list.append(lm)  # Append the landmark object directly

        # Get the ring position on the index finger, closer to PIP
        ring_position = get_ring_position(landmarks_list, shift_factor=0.7)
        print(f"Ring position: {ring_position}")

        # Draw the ring on the image
        draw_ring_position(image, ring_position)

    else:
        print("No hand detected")

    # Save the output image with the ring
    output_path = os.path.join(os.path.expanduser("~"), "Desktop", "image_with_ring.png")
    cv2.imwrite(output_path, image)
    print(f"Image saved at: {output_path}")

    return image

# Main function to call process_image with the input image
def main():
    input_image_path = "C:\\Users\\Ksena\\Documents\\Ring_try_on\\test\\original_6.png"  # Update with your image path
    image = process_image(input_image_path)

# Run the script
if __name__ == "__main__":
    main()
