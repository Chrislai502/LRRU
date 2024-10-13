import os
import numpy as np
import pandas as pd
import argparse
import time
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Custom modules (assuming they are available in your environment)
from dataloaders.kitti_loader import KittiDepth
from model import get as get_model
from summary import get as get_summary
from metric import get as get_metric
from utility import *

# MINIMIZE RANDOMNESS
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def rotate_point_cloud(pc, axis, angle_deg):
    """
    Rotates the point cloud around the specified axis by the given angle.
    Args:
        pc (numpy.ndarray): The point cloud, shape (N, 3).
        axis (str): Axis to rotate around ('roll', 'pitch', 'yaw').
        angle_deg (float): Rotation angle in degrees.
    Returns:
        numpy.ndarray: Rotated point cloud, shape (N, 3).
    """
    angle_rad = np.deg2rad(angle_deg)
    if axis == 'roll':
        R = np.array([[1, 0, 0],
                      [0, np.cos(angle_rad), -np.sin(angle_rad)],
                      [0, np.sin(angle_rad), np.cos(angle_rad)]])
    elif axis == 'pitch':
        R = np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                      [0, 1, 0],
                      [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
    elif axis == 'yaw':
        R = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                      [np.sin(angle_rad), np.cos(angle_rad), 0],
                      [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'roll', 'pitch', or 'yaw'")
    return (R @ pc.T).T

class KittiDepthAugmented(KittiDepth):
    def __init__(self, *args, rotation_axis=None, rotation_angle=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rotation_axis = rotation_axis
        self.rotation_angle = rotation_angle
        # Load camera intrinsic parameters
        self.K = self.get_camera_intrinsics()

    def get_camera_intrinsics(self):
        # Replace with your method to load camera intrinsics
        # For example, fx, fy, cx, cy
        # Return camera intrinsic matrix K
        fx = 721.5377
        fy = 721.5377
        cx = 609.5593
        cy = 172.8540
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
        return K

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # Load point cloud
        pc_path = self.get_point_cloud_path(idx)
        point_cloud = self.load_point_cloud(pc_path)
        # Apply rotation if specified
        if self.rotation_axis and self.rotation_angle is not None:
            point_cloud = rotate_point_cloud(point_cloud, self.rotation_axis, self.rotation_angle)
        # Project point cloud to image plane
        depth_map = self.project_point_cloud(point_cloud, self.K, sample['rgb'].size)
        # Update sample with augmented depth map
        sample['dep'] = torch.from_numpy(depth_map).float()
        return sample

    def get_point_cloud_path(self, idx):
        # Implement this method to get the path to the point cloud file
        # For example:
        # return self.pc_paths[idx]
        pass

    def load_point_cloud(self, pc_path):
        # Implement this method to load the point cloud from file
        # Return numpy array of shape (N, 3)
        # For example, using numpy.fromfile for KITTI binary point clouds
        # point_cloud = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        pass

    def project_point_cloud(self, point_cloud, K, image_size):
        """
        Projects the point cloud into the image plane to create a depth map.
        Args:
            point_cloud (numpy.ndarray): Point cloud (N, 3).
            K (numpy.ndarray): Camera intrinsic matrix (3, 3).
            image_size (tuple): (width, height)
        Returns:
            numpy.ndarray: Depth map (H, W)
        """
        width, height = image_size
        # Transform point cloud to camera coordinates
        # Apply extrinsic calibration if necessary
        # For simplicity, assume point_cloud is already in camera coordinates

        # Filter points in front of the camera
        valid = point_cloud[:, 2] > 0
        points_cam = point_cloud[valid].T  # Shape (3, N)

        # Project to image plane
        pixels = K @ points_cam  # Shape (3, N)
        pixels[:2] /= pixels[2]

        u = pixels[0]
        v = pixels[1]
        z = pixels[2]

        # Create depth map
        depth_map = np.zeros((height, width), dtype=np.float32)
        u = np.round(u).astype(np.int32)
        v = np.round(v).astype(np.int32)
        valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        depth_map[v[valid], u[valid]] = z[valid]
        return depth_map

def test(args):
    # Initialize lists to store evaluation metrics
    results_list = []
    
    # Define rotation parameters
    rotation_axes = ['roll', 'pitch', 'yaw']
    rotation_angles = np.arange(-1.0, 1.1, 0.2)  # From -1 to 1 degrees in 0.2 increments
    
    # DATASET
    print('Prepare data...')
    data_test = KittiDepth(args.test_option, args)
    loader_test = DataLoader(dataset=data_test, batch_size=1, shuffle=False, num_workers=1)
    print('Done!')
    
    # NETWORK
    print('Prepare model...')
    model = get_model(args)
    net = model(args)
    net.cuda()
    print('Done!')
    
    # METRIC
    print('Prepare metric...')
    metric = get_metric(args)
    metric = metric(args)
    print('Done!')
    
    # LOAD MODEL
    print('Load model...')
    if len(args.test_model) != 0:
        assert os.path.exists(args.test_model), "file not found: {}".format(args.test_model)
        checkpoint_ = torch.load(args.test_model, map_location='cpu')
        model = remove_moudle(checkpoint_)
        key_m, key_u = net.load_state_dict(model, strict=True)
        if key_u:
            print('Unexpected keys :', key_u)
        if key_m:
            print('Missing keys :', key_m)
    net = nn.DataParallel(net)
    net.eval()
    print('Done!')
    
    num_samples = len(loader_test)
    total_iterations = num_samples * len(rotation_axes) * len(rotation_angles)
    pbar_ = tqdm(total=total_iterations)
    
    with torch.no_grad():
        for batch_idx, sample_ in enumerate(loader_test):
            # Original sample without augmentation
            original_sample = sample_.copy()
            for axis in rotation_axes:
                for angle in rotation_angles:
                    # Apply augmentation
                    augmented_dataset = KittiDepthAugmented(args.test_option, args,
                                                            rotation_axis=axis,
                                                            rotation_angle=angle)
                    # Get the same sample with augmentation
                    augmented_sample = augmented_dataset[batch_idx]
                    samplep = {key: val.float().cuda() for key, val in augmented_sample.items()
                               if torch.is_tensor(val)}
                    samplep['d_path'] = augmented_sample['d_path']
                    # Run the model
                    output_ = net(samplep)
                    # Evaluate
                    if 'test' not in args.test_option:
                        metric_test = metric.evaluate(output_['results'][-1], samplep['gt'], 'test')
                    else:
                        metric_test = metric.evaluate(output_['results'][-1], samplep['dep'], 'test')
                    # Store results
                    results_list.append({
                        'sample_id': batch_idx,
                        'image_path': samplep['d_path'][0],
                        'rotation_axis': axis,
                        'rotation_angle': angle,
                        'RMSE': metric_test.data.cpu().numpy()[0, 0],
                        'MAE': metric_test.data.cpu().numpy()[0, 1],
                        # Add other metrics if needed
                    })
                    pbar_.update(1)
    pbar_.close()
    # Save results to CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('evaluation_results.csv', index=False)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Augmentation and Evaluation Script')
    parser.add_argument('--test_option', type=str, default='val_selection_cropped',
                        help='Dataset split to use for testing')
    parser.add_argument('--test_model', type=str, required=True,
                        help='Path to the pre-trained model to test')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0],
                        help='List of GPU ids to use')
    # Add other arguments as needed
    args = parser.parse_args()

    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpus))

    # Run the test function
    test(args)
