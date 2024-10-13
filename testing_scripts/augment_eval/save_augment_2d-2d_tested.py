import numpy as np
from PIL import Image
import os
import cv2  # OpenCV for remapping
import time  # To track runtime

# Step 1: Read depth image
# def read_depth_image(image_path):
#     with Image.open(image_path) as img:
#         depth_image = np.array(img, dtype=np.uint16)
    
#     # Create a mask of valid pixels
#     valid_mask = depth_image > 0
    
#     # Convert to depth in meters
#     depth_in_meters = depth_image.astype(np.float32) / 256.0
#     depth_in_meters[~valid_mask] = 0  # Set invalid depths to zero
    
#     return depth_in_meters, valid_mask

def read_depth_image(file_name):
    # loads depth map D from 16 bits png file as a numpy array,
    # refer to readme file in KITTI dataset
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    img_file = Image.open(file_name)
    image_depth = np.array(img_file, dtype=int)
    # img_file.close()

    # Consider empty depth
    assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
        "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    image_depth = image_depth.astype(np.float32) / 256.0
    depth = Image.fromarray(image_depth.astype('float32'), mode='F')
    return depth

# Step 2: Read calibration file
def read_calibration_file(filepath=None):
    if filepath is None:
        fx = 721.5377  # Focal length in x
        fy = 721.5377  # Focal length in y
        cx = 609.5593  # Principal point x
        cy = 172.8540  # Principal point y
        return fx, fy, cx, cy
    
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

# Functions to apply rotations directly to depth map
def apply_roll_to_depth_map(depth_map, phi_deg, fx, fy, cx, cy):
    phi = np.deg2rad(phi_deg)
    h, w = depth_map.shape
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    u_centered = u_coords - cx
    v_centered = v_coords - cy
    u_prime = u_centered * (1 - phi * v_centered / fy) + cx
    v_prime = v_coords - fy * phi
    map_x = u_prime.astype(np.float32)
    map_y = v_prime.astype(np.float32)
    # Use cv2.remap to apply the transformation
    transformed_depth_map = cv2.remap(depth_map, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return transformed_depth_map

def apply_pitch_to_depth_map(depth_map, theta_deg, fx, fy, cx, cy):
    theta = np.deg2rad(theta_deg)
    h, w = depth_map.shape
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    u_prime = u_coords - fx * theta  # Corrected sign
    v_prime = v_coords
    map_x = u_prime.astype(np.float32)
    map_y = v_prime.astype(np.float32)
    transformed_depth_map = cv2.remap(depth_map, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return transformed_depth_map


def apply_yaw_to_depth_map(depth_map, psi_deg, fx, fy, cx, cy):
    psi = np.deg2rad(psi_deg)
    h, w = depth_map.shape
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    u_centered = u_coords - cx
    v_centered = v_coords - cy
    u_prime = u_coords - psi * (fx / fy) * v_centered
    v_prime = v_coords + psi * (fy / fx) * u_centered
    map_x = u_prime.astype(np.float32)
    map_y = v_prime.astype(np.float32)
    transformed_depth_map = cv2.remap(depth_map, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return transformed_depth_map

# Step 6: Save depth image to PNG
def save_depth_image(depth_map, output_path):
    # Handle invalid depths (e.g., NaNs) if necessary
    depth_map = np.nan_to_num(depth_map, nan=0.0)
    depth_map_uint16 = (depth_map * 256).astype(np.uint16)
    Image.fromarray(depth_map_uint16).save(output_path)

# Main function to perform augmentations and save rotated depth images
def augment_and_save_depth_images(image_path, calibration_file_path, output_dir):
    # Define rotation increments and limits
    rotation_range = np.arange(-2, 2.2, 0.2)  # From -1 to 1 degree in 0.2 degree increments

    # Read depth image
    depth_map = np.array(read_depth_image(image_path))
    print("Nan count: ", np.count_nonzero(np.isnan(depth_map)))
    print("Inf count: ", np.count_nonzero(np.isinf(depth_map)))

    # Read camera intrinsics
    fx, fy, cx, cy = read_calibration_file(calibration_file_path)

    # Prepare output directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get original filename
    original_filename = os.path.basename(image_path)

    # Timing variables
    total_time = 0
    transform_count = 0

    # Iterate through roll, pitch, and yaw values
    for axis_name in ['roll', 'pitch', 'yaw']:
        for offset in rotation_range:
            # Start timing
            start_time = time.time()
            
            # Apply rotation directly to depth map
            if axis_name == 'roll':
                rotated_depth_map = apply_roll_to_depth_map(depth_map, phi_deg=offset, fx=fx, fy=fy, cx=cx, cy=cy)
            elif axis_name == 'pitch':
                rotated_depth_map = apply_pitch_to_depth_map(depth_map, theta_deg=offset, fx=fx, fy=fy, cx=cx, cy=cy)
            elif axis_name == 'yaw':
                rotated_depth_map = apply_yaw_to_depth_map(depth_map, psi_deg=offset, fx=fx, fy=fy, cx=cx, cy=cy)
            
            # Create output directory for this axis
            axis_output_dir = os.path.join(output_dir, axis_name)
            if not os.path.exists(axis_output_dir):
                os.makedirs(axis_output_dir)
            
            # Save the rotated depth image
            sign = "pos" if offset > 0 else "neg"
            output_filename = f"{axis_name}_offset_{sign}{abs(offset):.1f}_{original_filename}"
            output_path = os.path.join(axis_output_dir, output_filename)
            save_depth_image(rotated_depth_map, output_path)
            print(f"Saved: {output_path}")
            
            # Stop timing
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Update timing
            total_time += elapsed_time
            transform_count += 1

    # Calculate and print average runtime
    if transform_count > 0:
        average_time = total_time / transform_count
        print(f"Average runtime per transformation: {average_time:.4f} seconds")
    else:
        print("No transformations performed.")

# Example usage
if __name__ == "__main__":
    # Define paths to your depth image and calibration file
    depth_image_path = '/home/art-chris/testing/LRRU/data/kitti_depth/depth_selection/val_selection_cropped/velodyne_raw/2011_10_03_drive_0047_sync_velodyne_raw_0000000791_image_03.png'
    calibration_file_path = None  # Use default intrinsics if None

    # Define output directory
    output_dir = '/home/art-chris/testing/LRRU/testing_scripts/augment_eval/rotated_depth_images_2d-2d'

    os.makedirs(output_dir, exist_ok=True)

    # Perform augmentations and save rotated depth images
    augment_and_save_depth_images(depth_image_path, calibration_file_path, output_dir)
