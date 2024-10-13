import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt


fx = 721.5377  # Focal length in x
fy = 721.5377  # Focal length in y
cx = 609.5593  # Principal point x
cy = 172.8540  # Principal point y


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

def backproject_to_3d(depth_map, fx, fy, cx, cy):
    """
    Back-projects the depth map into 3D space.
    """
    height, width = depth_map.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    
    # Compute normalized coordinates
    x = (i - cx) / fx
    y = (j - cy) / fy
    
    y = -y # Invert from image to camera frame
    x = -x
    
    # Multiply by depth to get 3D coordinates
    X = x * depth_map
    Y = y * depth_map
    Z = depth_map
    
    # Stack into N x 3 points
    points = np.stack((X, Y, Z), axis=-1)
    
    return points

def visualize_depth_map(depth_map, valid_mask):
    """
    Visualizes the 2D depth map with colormapping.
    """
    import matplotlib.pyplot as plt
    
    # Set invalid depths to NaN for better visualization
    depth_map_vis = np.copy(depth_map)
    depth_map_vis[~valid_mask] = 0
    
    plt.figure(figsize=(10, 5))
    plt.imshow(depth_map_vis, cmap='jet')
    plt.colorbar(label='Depth (m)')
    plt.title('Depth Map Visualization')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    # plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.show()

def visualize_point_cloud(points, valid_mask):
    """
    Visualizes the 3D point cloud with colormapping based on depth.
    """
    # Flatten the arrays
    points = points.reshape(-1, 3)
    valid_mask = valid_mask.flatten()
    
    # Keep only valid points
    points = points[valid_mask]
    
    # Normalize depth values for colormap
    depth = points[:, 2]
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    
    # Map depth to colors
    cmap = plt.get_cmap('jet')
    colors = cmap(depth_normalized)[:, :3]  # Exclude alpha channel
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=3.0, origin=[0, 0, 0]
    )
    
    # # Visualize
    # o3d.visualization.draw_geometries([pcd, coordinate_frame])
    
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point Cloud', width=800, height=600)
    
    # Add geometries
    vis.add_geometry(pcd)
    vis.add_geometry(coordinate_frame)
    
    # Customize render options
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0, 0, 0])  # Black background
    render_option.point_size = 1.0  # Adjust point size if necessary
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()


# Define the path to your depth image
depth_image_path = 'path_to_your_depth_image.png'
depth_image_path = '/home/art-chris/testing/LRRU/data/kitti_depth/depth_selection/val_selection_cropped/velodyne_raw/2011_10_03_drive_0047_sync_velodyne_raw_0000000791_image_03.png'

# Read the depth image
depth_map, valid_mask = read_depth_image(depth_image_path)

# # Visualize the 2D depth map
# visualize_depth_map(depth_map, valid_mask)

# Back-project to 3D
points_3d = backproject_to_3d(depth_map, fx, fy, cx, cy)

# Visualize the 3D point cloud
visualize_point_cloud(points_3d, valid_mask)
