import cv2
import mediapipe as mp
import numpy as np
import trimesh
import pyrender
from PIL import Image
import os

# Load Mediapipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Load the image
image_path = "data/images/original_0.png"  # Change this to your image file
hand_image = cv2.imread(image_path)
hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Load the 3D ring model
ring_mesh = trimesh.load("data/models/ring/ring.obj")
scene = pyrender.Scene()
mesh = pyrender.Mesh.from_trimesh(ring_mesh)
scene.add(mesh)

def put_ring_on(frame, x, y):
    """Render the 3D ring model at the detected hand position."""
    renderer = pyrender.OffscreenRenderer(400, 400)
    color, _ = renderer.render(scene)

    # Convert OpenGL render to OpenCV format
    ring_image = cv2.cvtColor(color, cv2.COLOR_RGBA2RGB)
    ring_resized = cv2.resize(ring_image, (100, 100))

    # Overlay 3D ring onto the image at (x, y)
    h, w, _ = frame.shape
    if 0 <= x - 50 < w and 0 <= y - 50 < h:
        frame[y-50:y+50, x-50:x+50] = cv2.addWeighted(frame[y-50:y+50, x-50:x+50], 0.5, ring_resized, 0.5, 0)

def save_image(frame):
    """Save the processed image."""
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    cv2.imwrite(os.path.join(desktop_path, "ring_image.png"), frame)

def main():
    """Main function to overlay the 3D ring automatically."""
    frame = hand_image.copy()  # Work on a copy to avoid modifying the original
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        base = hand_landmarks.landmark[6]  # Base of the ring finger
        pip = hand_landmarks.landmark[7]  # Tip of the ring finger

        x = int(base.x * frame.shape[1])
        y = int(base.y * frame.shape[0]) + 50  # Offset down

        # Automatically place the ring without needing keypress
        put_ring_on(frame, x, y)

    # Show the processed image
    cv2.imshow('Ring on Finger', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Save the image automatically
    save_image(frame)
    print("Image with ring saved on the desktop.")

    # Wait indefinitely until the user closes the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
