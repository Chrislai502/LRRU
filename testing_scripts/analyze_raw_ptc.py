from PIL import Image
import numpy as np

# Load the image
image_path = '/home/art-chris/testing/LRRU/data/kitti_depth/depth_selection/val_selection_cropped/velodyne_raw/2011_10_03_drive_0047_sync_velodyne_raw_0000000791_image_03.png'
with Image.open(image_path) as img:
    # Ensure the image is in 'I;16' mode (16-bit unsigned integers)
    depth_image = np.array(img, dtype=np.uint16)

# Create a mask of valid pixels
valid_mask = depth_image > 0

# Compute depth in meters
depth_in_meters = depth_image.astype(np.float32) / 256.0

# Optionally, set invalid pixels to NaN
depth_in_meters[~valid_mask] = np.nan

