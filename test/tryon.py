import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=10, min_detection_confidence=0.5)

def draw_circle(frame, center, radius):
    cv2.circle(frame, center, radius, (0, 0, 255), -1)

def save_image(frame):
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    cv2.imwrite(os.path.join(desktop_path, "ring_image.png"), frame)

def calculate_ring_radius(base, tip):
    distance = np.linalg.norm(np.array([base.x, base.y]) - np.array([tip.x, tip.y]))
    return max(5, int(distance * 100))  # Scale radius based on the distance between base and tip

def calculate_ring_position(base, tip, frame_shape):
    # Середня точка між основою і кінчиком пальця
    avg_x = (base.x + tip.x) / 2
    avg_y = (base.y + tip.y) / 2

    # Конвертація нормалізованих координат у пікселі
    pixel_x = int(avg_x * frame_shape[1])
    pixel_y = int(avg_y * frame_shape[0])

    return pixel_x, pixel_y

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
            for hand_landmarks in results.multi_hand_landmarks:
                # Вибираємо ключові точки для безіменного пальця
                base = hand_landmarks.landmark[6]  # Основа безіменного пальця
                tip = hand_landmarks.landmark[5]  # Кінець безіменного пальця

                # Розраховуємо позицію кільця
                center = calculate_ring_position(base, tip, frame.shape)

                # Розраховуємо радіус кільця
                ring_radius = calculate_ring_radius(base, tip)

                if ring_on_finger:
                    draw_circle(frame, center, ring_radius)

        # Відображаємо кадр з кільцями
        cv2.imshow('Rings on Fingers', frame)

        # Обробка клавіш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            ring_on_finger = not ring_on_finger  # Вмикаємо/вимикаємо кільце
        elif key == ord('s'):
            save_image(frame)  # Зберігаємо зображення
            print("Image saved on the desktop.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
