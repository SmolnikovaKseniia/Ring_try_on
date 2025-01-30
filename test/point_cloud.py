import numpy as np
import open3d as o3d
import cv2
from config import IMAGE_PATH, DEPTH_PATH, CALIBRATION_PATH

def load_calibration(file_path: str):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Parse intrinsic matrix
    intrinsic = np.array([list(map(float, line.split(','))) for line in lines[1:4]])

    # Parse extrinsic matrix (ensure it's a 4x4 matrix)
    extrinsic = np.array([list(map(float, line.split(','))) for line in lines[6:10]])

    return intrinsic, extrinsic
# Example usage
intrinsic, extrinsic = load_calibration(CALIBRATION_PATH)
print("Intrinsic Matrix:\n", intrinsic)
print("Extrinsic Matrix:\n", extrinsic)

def get_rgbd(image_path: str, depth_path: str, intrinsic: np.ndarray):
    """
    Creates an RGB-D frame from image and depth data using calibration parameters.
    """
    # Read RGB image and convert to RGB format
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load and resize depth image to match RGB image dimensions
    depth_data = np.loadtxt(depth_path, delimiter=',')  # Assuming CSV format for depth
    depth_resized = cv2.resize(depth_data, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert depth image to Open3D format
    depth_raw = o3d.geometry.Image(depth_resized.astype(np.float32))  # Depth in meters

    # Create Open3D RGB image
    color_raw = o3d.geometry.Image(image_rgb)

    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=1.0, depth_trunc=1000.0, convert_rgb_to_intensity=False)

    # Create PinholeCameraIntrinsic object and set intrinsic matrix
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.intrinsic_matrix = intrinsic

    return rgbd_image, camera_intrinsic


def get_point_cloud(image_path: str, depth_path: str, extrinsic: np.array, intrinsic: np.array) -> o3d.geometry.PointCloud:
    """
    Creates a point cloud from image and depth data using calibration parameters.
    Args:
        image_path (str): Path to the RGB image.
        depth_path (str): Path to the depth image.
        extrinsic (np.ndarray): Extrinsic matrix (4x4).
        intrinsic (np.ndarray): Intrinsic matrix (3x3).
    Returns:
        o3d.geometry.PointCloud: The point cloud generated from the RGB-D data.
    """
    # Load RGB and depth images
    rgbd_image, camera_intrinsic = get_rgbd(image_path, depth_path, intrinsic)

    # Create point cloud from the RGBD image using the camera intrinsic and extrinsic matrices
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic, extrinsic)

    return pcd


# Example usage
image_path = IMAGE_PATH
depth_path = DEPTH_PATH
# Assuming your depth data is stored as comma-separated values in a text file
depth_data = np.loadtxt(depth_path, delimiter=',')
depth_image = o3d.geometry.Image(depth_data.astype(np.float32))  # Convert to float32

pcd = get_point_cloud(image_path, depth_path, extrinsic, intrinsic)
o3d.visualization.draw_geometries([pcd])
