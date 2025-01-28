import open3d as o3d
import numpy as np
import cv2

# Load the depth and RGB images
depth = cv2.imread("data/images/depth_0.png", cv2.IMREAD_UNCHANGED)
rgb = cv2.imread("data/images/original_0.png")

depth = depth.astype(np.float32)



