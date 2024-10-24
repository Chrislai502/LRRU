import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d

def read_depth_image(image_path):
    """
    Reads a KITTI depth image and converts it to a depth map in meters.
    """
    with Image.open(image_path) as img:
        depth_image = np.array(img, dtype=np.uint16)
    
    # Create a mask of valid pixels
    valid_mask = depth_image > 0
    
    # Convert to depth in meters
    depth_in_meters = depth_image.astype(np.float32) / 256.0
    depth_in_meters[~valid_mask] = 0  # Optional: Set invalid depths to zero
    
    return depth_in_meters, valid_mask

def read_calibration_file(filepath):
    """
    Reads the camera calibration file and extracts intrinsic parameters.
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip()] = value.strip()
    
    # Example for P_rect_02 (Left camera)
    P_rect_02 = np.array([float(x) for x in data['P_rect_02'].split()]).reshape(3, 4)
    fx = P_rect_02[0, 0]
    fy = P_rect_02[1, 1]
    cx = P_rect_02[0, 2]
    cy = P_rect_02[1, 2]
    
    return fx, fy, cx, cy

def backproject_to_3d(depth_map, fx, fy, cx, cy):
    """
    Back-projects the depth map into 3D space with corrected y-axis.
    """
    height, width = depth_map.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    
    # Compute normalized coordinates
    x = (i - cx) / fx
    y = (j - cy) / fy

    # Invert the y-axis to match camera coordinate system
    y = -y

    # Multiply by depth to get 3D coordinates
    X = x * depth_map
    Y = y * depth_map
    Z = depth_map
    
    # Stack into N x 3 points
    points = np.stack((X, Y, Z), axis=-1)
    
    return points

def visualize_two_point_clouds(points_list, valid_masks, colormaps, point_size=2.0):
    """
    Visualizes two 3D point clouds with different colormaps and increased point size.
    """
    import open3d as o3d
    import matplotlib.pyplot as plt
    import numpy as np

    geometries = []
    for idx, (points, valid_mask, cmap_name) in enumerate(zip(points_list, valid_masks, colormaps)):
        # Flatten the arrays
        points = points.reshape(-1, 3)
        valid_mask = valid_mask.flatten()
        
        # Keep only valid points
        points = points[valid_mask]
        
        # Normalize depth values for colormap
        depth = points[:, 2]
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        # Map depth to colors using specified colormap
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(depth_normalized)[:, :3]  # Exclude alpha channel
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        geometries.append(pcd)
    
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=3.0, origin=[0, 0, 0]
    )
    geometries.append(coordinate_frame)
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Two Point Clouds', width=800, height=600)
    
    # Add geometries
    for geometry in geometries:
        vis.add_geometry(geometry)
    
    # Customize render options
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0, 0, 0])  # Black background
    render_option.point_size = point_size  # Increase point size
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

# Example usage
if __name__ == "__main__":
    # Define paths to your depth images and calibration file
    depth_image_path1 = 'path_to_your_first_depth_image.png'
    depth_image_path2 = 'path_to_your_second_depth_image.png'
    depth_image_path1 = '/home/art-chris/testing/LRRU/data/kitti_depth/depth_selection/val_selection_cropped/velodyne_raw/2011_10_03_drive_0047_sync_velodyne_raw_0000000791_image_03.png'
    depth_image_path2 = '/home/art-chris/testing/LRRU/data/kitti_depth/depth_selection/val_selection_cropped/velodyne_raw/2011_10_03_drive_0047_sync_velodyne_raw_0000000761_image_02.png'
    depth_image_path1 = '/home/art-chris/testing/LRRU/data/kitti_depth/depth_selection/val_selection_cropped/velodyne_raw/2011_10_03_drive_0047_sync_velodyne_raw_0000000791_image_03.png'

    # depth_image_path1 = "/home/art-chris/testing/LRRU/testing_scripts/augment_eval/rotated_depth_images_rpy_combined/2011_10_03_drive_0047_sync_velodyne_raw_0000000791_image_03_roll_+1.0_pitch_+1.0_yaw_+1.0.png"
    depth_image_path2 = "/home/art-chris/testing/LRRU/testing_scripts/augment_eval/rotated_depth_images_rpy_combined/2011_10_03_drive_0047_sync_velodyne_raw_0000000791_image_03_roll_-5.0_pitch_-5.0_yaw_-5.0.png"


    # calibration_file_path = 'calib_cam_to_cam.txt'
    
    # Read depth images
    depth_map1, valid_mask1 = read_depth_image(depth_image_path1)
    depth_map2, valid_mask2 = read_depth_image(depth_image_path2)
    
    # Read camera intrinsics
    # fx, fy, cx, cy = read_calibration_file(calibration_file_path)
    fx = 721.5377  # Focal length in x
    fy = 721.5377  # Focal length in y
    cx = 609.5593  # Principal point x
    cy = 172.8540  # Principal point y
    
    # Back-project to 3D with corrected y-axis
    points_3d_1 = backproject_to_3d(depth_map1, fx, fy, cx, cy)
    points_3d_2 = backproject_to_3d(depth_map2, fx, fy, cx, cy)
    
    # Visualize the two 3D point clouds with different colormaps and increased point size
    points_list = [points_3d_1, points_3d_2]
    valid_masks = [valid_mask1, valid_mask2]
    colormaps = ['winter', 'autumn']  # You can choose different colormaps
    point_size = 2.0  # Increase point size
    
    visualize_two_point_clouds(points_list, valid_masks, colormaps, point_size)
