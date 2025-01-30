import cv2
import numpy as np

# Load images
ring_path = "results/ring_render_new_1.png"  # Ring image
hand_path = "data/images/original_0.png"  # Hand image

# Read the images
hand_img = cv2.imread(hand_path)
ring_img = cv2.imread(ring_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

# Extract the alpha channel from the ring image
if ring_img.shape[2] == 4:  # Ensure image has alpha channel
    alpha_channel = ring_img[:, :, 3] / 255.0
    rgb_ring = ring_img[:, :, :3]
else:
    raise ValueError("Ring image must have an alpha channel.")

# Resize ring to fit on the hand (modify as needed)
scale_factor = 0.5  # Adjust to fit properly
ring_resized = cv2.resize(rgb_ring, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
alpha_resized = cv2.resize(alpha_channel, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

# Define position on the hand image
x_offset, y_offset = 200, 300  # Adjust these values as needed

# Get region of interest (ROI) on the hand image
h, w, _ = ring_resized.shape
hand_roi = hand_img[y_offset:y_offset+h, x_offset:x_offset+w]

# Blend images using alpha compositing
for c in range(3):  # Iterate over RGB channels
    hand_roi[:, :, c] = (1 - alpha_resized) * hand_roi[:, :, c] + alpha_resized * ring_resized[:, :, c]

# Replace the blended region in the hand image
hand_img[y_offset:y_offset+h, x_offset:x_offset+w] = hand_roi

# Save and display the result
cv2.imwrite("results/hand_with_ring2.png", hand_img)
cv2.imshow("Hand with Ring", hand_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
