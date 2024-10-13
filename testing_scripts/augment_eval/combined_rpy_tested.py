import numpy as np
from PIL import Image
import os
import cv2  # OpenCV for remapping
import time  # To track runtime

# Step 1: Read depth image

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
    return image_depth.astype('float32')

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

# Combined rotation function
def apply_combined_rotation_to_depth_map(depth_map, roll_deg, pitch_deg, yaw_deg, fx, fy, cx, cy):
    """
    Applies combined roll, pitch, and yaw rotations directly to the depth map.
    """
    # Convert degrees to radians
    phi = np.deg2rad(roll_deg)    # Roll
    theta = np.deg2rad(pitch_deg) # Pitch
    psi = np.deg2rad(yaw_deg)     # Yaw
    
    h, w = depth_map.shape
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    
    # Center coordinates
    u_centered = u_coords - cx
    v_centered = v_coords - cy
    
    # Compute changes in u and v
    # From roll
    delta_u_roll = - (u_centered) * phi * v_centered / fy
    delta_v_roll = - fy * phi
    
    # From pitch
    delta_u_pitch = - fx * theta
    delta_v_pitch = 0
    
    # From yaw
    delta_u_yaw = - psi * (fx / fy) * v_centered
    delta_v_yaw = psi * (fy / fx) * u_centered
    
    # Total changes
    delta_u = delta_u_roll + delta_u_pitch + delta_u_yaw
    delta_v = delta_v_roll + delta_v_pitch + delta_v_yaw
    
    # Compute new coordinates
    u_prime = u_coords + delta_u
    v_prime = v_coords + delta_v
    
    map_x = u_prime.astype(np.float32)
    map_y = v_prime.astype(np.float32)
    
    # Apply remapping
    transformed_depth_map = cv2.remap(depth_map, map_x, map_y, interpolation=cv2.INTER_NEAREST)
    
    return transformed_depth_map

# Step 6: Save depth image to PNG
def save_depth_image(depth_map, output_path):
    # Handle invalid depths (e.g., NaNs)
    depth_map = np.nan_to_num(depth_map, nan=0.0)
    depth_map_uint16 = (depth_map * 256).astype(np.uint16)
    Image.fromarray(depth_map_uint16).save(output_path)

# Main function to perform volumetric sweep across roll, pitch, and yaw
def augment_and_save_depth_images(image_path, calibration_file_path, output_dir):
    # Define rotation increments and limits for each axis
    rotation_range = np.arange(-5, 5, 2)  # From -1 to 1 degree in 0.2-degree increments

    # Read depth image
    depth_map = read_depth_image(image_path)

    # Read camera intrinsics
    fx, fy, cx, cy = read_calibration_file(calibration_file_path)

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get original filename
    original_filename = os.path.basename(image_path)
    base_filename, ext = os.path.splitext(original_filename)

    # Timing variables
    total_time = 0
    transform_count = 0

    # Iterate through all combinations of roll, pitch, and yaw
    for roll_deg in rotation_range:
        for pitch_deg in rotation_range:
            for yaw_deg in rotation_range:
                # Start timing
                start_time = time.time()
                
                # Apply combined rotation
                rotated_depth_map = apply_combined_rotation_to_depth_map(
                    depth_map, roll_deg, pitch_deg, yaw_deg, fx, fy, cx, cy
                )
                
                # Create output filename including roll, pitch, and yaw values
                output_filename = f"{base_filename}_roll_{roll_deg:+.1f}_pitch_{pitch_deg:+.1f}_yaw_{yaw_deg:+.1f}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save the rotated depth image
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
    output_dir = '/home/art-chris/testing/LRRU/testing_scripts/augment_eval/rotated_depth_images_rpy_combined'

    # Perform volumetric sweep and save rotated depth images
    augment_and_save_depth_images(depth_image_path, calibration_file_path, output_dir)
