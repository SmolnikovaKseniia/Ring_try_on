import cv2
import mediapipe as mp
import numpy as np
import trimesh
import pyrender
from OpenGL.GL import *
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

ring_mesh = trimesh.load("data/models/ring/ring.obj")
scene = pyrender.Scene()
mesh = pyrender.Mesh.from_trimesh(ring_mesh)
scene.add(mesh)

def put_ring_on(frame, x, y):
    """Render the 3D ring model at the detected hand position."""
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Render the 3D model
    renderer = pyrender.OffscreenRenderer(400, 400)
    color, _ = renderer.render(scene)

    # Convert OpenGL render to OpenCV format
    ring_image = cv2.cvtColor(color, cv2.COLOR_RGBA2RGB)
    ring_resized = cv2.resize(ring_image, (100, 100))

    # Overlay 3D ring onto video feed at (x, y)
    h, w, _ = frame.shape
    if 0 <= x - 50 < w and 0 <= y - 50 < h:
        frame[y-50:y+50, x-50:x+50] = cv2.addWeighted(frame[y-50:y+50, x-50:x+50], 0.5, ring_resized, 0.5, 0)

def draw_circle(frame, center, radius):
    cv2.circle(frame, center, radius, (0, 0, 0), -1)

def save_image(frame):
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    cv2.imwrite(os.path.join(desktop_path, "ring_image.png"), frame)

def calculate_ring_radius(base, tip):
    distance = np.linalg.norm(np.array([base.x, base.y]) - np.array([tip.x, tip.y]))
    return max(5, int(distance * 100))  # Scale based on distance

def put_ring_on(frame, center, radius):
    return

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open the camera")
        return

    ring_on_finger = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            base = hand_landmarks.landmark[6]  # Base of the ring finger
            tip = hand_landmarks.landmark[7]  # Tip of the ring finger

            x = int(base.x * frame.shape[1])
            y = int(base.y * frame.shape[0]) + 50  # Offset down

            if ring_on_finger:
                put_ring_on(frame, x, y)
                # ring_radius = calculate_ring_radius(base, tip)  # Adjust radius to fit finger size
                # draw_circle(frame, (x, y), ring_radius)

        cv2.imshow('Ring on Finger', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            ring_on_finger = not ring_on_finger
        elif key == ord('s'):
            save_image(frame)
            print("Image saved on the desktop.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()