import numpy as np
from PIL import Image
import os
from scipy.spatial.transform import Rotation as R

## Chat Reference: 
# - https://chatgpt.com/c/6705c501-d5c0-8006-bd53-1c7f752864cc
# - https://chatgpt.com/g/g-YyyyMT9XH-chatgpt-classic/c/670b337e-c6ac-8006-9b9e-a7c6f5332e5a

# Step 1: Read depth image
def read_depth_image(image_path):
    with Image.open(image_path) as img:
        depth_image = np.array(img, dtype=np.uint16)
    
    # Create a mask of valid pixels
    valid_mask = depth_image > 0
    
    # Convert to depth in meters
    depth_in_meters = depth_image.astype(np.float32) / 256.0
    depth_in_meters[~valid_mask] = 0  # Optional: Set invalid depths to zero
    
    return depth_in_meters, valid_mask

# Step 2: Read calibration file
def read_calibration_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip()] = value.strip()
    
    P_rect_02 = np.array([float(x) for x in data['P_rect_02'].split()]).reshape(3, 4)
    fx = P_rect_02[0, 0]
    fy = P_rect_02[1, 1]
    cx = P_rect_02[0, 2]
    cy = P_rect_02[1, 2]
    
    return fx, fy, cx, cy

# Step 3: Backproject depth image to 3D points
def backproject_to_3d(depth_map, fx, fy, cx, cy):
    height, width = depth_map.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    
    x = (i - cx) / fx
    y = (j - cy) / fy
    y = -y

    X = x * depth_map
    Y = y * depth_map
    Z = depth_map
    
    points = np.stack((X, Y, Z), axis=-1)
    
    return points

# Step 4: Apply rotation to 3D points
def apply_rotation(point_cloud, roll=0, pitch=0, yaw=0):
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    
    r = R.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad], degrees=False)
    rotated_point_cloud = r.apply(point_cloud.reshape(-1, 3))
    
    return rotated_point_cloud.reshape(point_cloud.shape)

# Step 5: Project 3D points back to 2D depth map
def project_to_2d(points_3d, fx, fy, cx, cy, height, width):
    X = points_3d[:, :, 0]
    Y = points_3d[:, :, 1]
    Z = points_3d[:, :, 2]

    # Avoid division by zero
    Z[Z <= 0] = 1e-5
    
    i = (X * fx / Z) + cx
    j = (-Y * fy / Z) + cy
    
    i = np.clip(np.round(i), 0, width - 1).astype(np.int)
    j = np.clip(np.round(j), 0, height - 1).astype(np.int)

    depth_map = np.zeros((height, width), dtype=np.float32)
    depth_map[j, i] = Z
    
    return depth_map

# Step 6: Save depth image to PNG
def save_depth_image(depth_map, output_path):
    depth_map_uint16 = (depth_map * 256).astype(np.uint16)
    Image.fromarray(depth_map_uint16).save(output_path)

# Main function to perform augmentations and save rotated depth images
def augment_and_save_depth_images(image_path, calibration_file_path, output_dir):
    # Define rotation increments and limits
    rotation_range = np.arange(-1, 1.2, 0.2)  # From -1 to 1 degree in 0.2 degree increments
    
    # Read depth image
    depth_map, valid_mask = read_depth_image(image_path)
    
    # Read camera intrinsics
    fx, fy, cx, cy = read_calibration_file(calibration_file_path)
    
    # Back-project to 3D
    points_3d = backproject_to_3d(depth_map, fx, fy, cx, cy)
    
    # Prepare output directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get original filename
    original_filename = os.path.basename(image_path)
    
    # Iterate through roll, pitch, and yaw values
    for axis, axis_name in zip([0, 1, 2], ['roll', 'pitch', 'yaw']):
        for offset in rotation_range:
            # Create rotation vector (only rotate along the current axis)
            if axis_name == 'roll':
                rotated_points = apply_rotation(points_3d, roll=offset)
            elif axis_name == 'pitch':
                rotated_points = apply_rotation(points_3d, pitch=offset)
            elif axis_name == 'yaw':
                rotated_points = apply_rotation(points_3d, yaw=offset)
            
            # Project back to 2D
            height, width = depth_map.shape
            rotated_depth_map = project_to_2d(rotated_points, fx, fy, cx, cy, height, width)
            
            # Create output directory for this axis
            axis_output_dir = os.path.join(output_dir, axis_name)
            if not os.path.exists(axis_output_dir):
                os.makedirs(axis_output_dir)
            
            # Save the rotated depth image
            output_filename = f"{axis_name}_offset_{offset:+.1f}_{original_filename}"
            output_path = os.path.join(axis_output_dir, output_filename)
            save_depth_image(rotated_depth_map, output_path)
            print(f"Saved: {output_path}")

# Example usage
if __name__ == "__main__":
    # Define paths to your depth image and calibration file
    depth_image_path = 'path_to_your_depth_image.png'
    calibration_file_path = 'path_to_your_calibration_file.txt'
    
    # Define output directory
    output_dir = 'ROOT_OUTPUT_DIR'
    
    # Perform augmentations and save rotated depth images
    augment_and_save_depth_images(depth_image_path, calibration_file_path, output_dir)
