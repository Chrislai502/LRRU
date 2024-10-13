import argparse

arg = argparse.ArgumentParser(description='depth completion')
arg.add_argument('-p', '--project_name', type=str, default='inference')
arg.add_argument('-c', '--configuration', type=str, default='val_iccv.yml')
arg = arg.parse_args()
temp = arg.configuration
from configs import get as get_cfg
config = get_cfg(arg)

# ENVIRONMENT SETTINGS
import os
rootPath = os.path.abspath(os.path.dirname(__file__))
import functools
if len(config.gpus) == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpus[0])
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = functools.reduce(lambda x, y: str(x) + ',' + str(y), config.gpus)

import emoji
import os
import numpy as np
import argparse
import time
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import pandas as pd
import itertools

# Custom modules (assuming they are available in your environment)
from dataloaders.kitti_loader import KittiDepthValAugmented
from dataloaders.utils import outlier_removal as out_removal
from dataloaders.NNfill import fill_in_fast as filll
from model import get as get_model
from summary import get as get_summary
from metric import get as get_metric
from utility import *

# MINIMIZE RANDOMNESS
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def apply_combined_rotation_to_depth_map(depth_map, roll_deg, pitch_deg, yaw_deg, fx, fy, cx, cy):
    """
    Applies combined roll, pitch, and yaw rotations directly to the depth map.
    """
    # Convert degrees to radians
    phi = np.deg2rad(roll_deg)    # Roll
    theta = np.deg2rad(pitch_deg) # Pitch
    psi = np.deg2rad(yaw_deg)     # Yaw
    
    print(depth_map.shape)
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

def test(args):
    # Initialize lists to store evaluation metrics
    results_list = []
    print(args)
    
    # Define rotation parameters
    rotation_start = int(args.rotation_start)
    rotation_stop = int(args.rotation_stop)
    rotation_step = int(args.rotation_step)

    roll_values = list(range(rotation_start, rotation_stop, rotation_step))
    pitch_values = list(range(rotation_start, rotation_stop, rotation_step))
    yaw_values = list(range(rotation_start, rotation_stop, rotation_step))

    # All combinations
    rotation_combinations = list(itertools.product(roll_values, pitch_values, yaw_values))
    
    # DATASET
    print(emoji.emojize('Prepare data... :writing_hand:', variant="emoji_type"), end=' ')
    data_test = KittiDepthValAugmented(args.test_option, args)
    loader_test = DataLoader(dataset=data_test,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1)
    print('Done!')

    
     # NETWORK
    print(emoji.emojize('Prepare model... :writing_hand:', variant="emoji_type"), end=' ')
    model = get_model(args)
    net = model(args)
    net.cuda()
    print('Done!')
    total_params = count_parameters(net)
    
    # METRIC
    print('Prepare metric...')
    metric = get_metric(args)
    metric = metric(args)
    print('Done!')
    
    # LOAD MODEL
    print(emoji.emojize('Load model... :writing_hand:', variant="emoji_type"), end=' ')
    if len(args.test_model) != 0:
        assert os.path.exists(args.test_model), \
            "file not found: {}".format(args.test_model)

        checkpoint_ = torch.load(args.test_model, map_location='cpu')
        model = remove_moudle(checkpoint_)
        key_m, key_u = net.load_state_dict(model, strict=True)

        if key_u:
            print('Unexpected keys :')
            print(key_u)

        if key_m:
            print('Missing keys :')
            print(key_m)

    net = nn.DataParallel(net)
    net.eval()
    print('Done!')
    
    # # Read camera intrinsics
    # fx, fy, cx, cy = read_calibration_file(args.calibration_file)
    
    num_samples = len(loader_test)
    total_iterations = num_samples * len(rotation_combinations)
    pbar_ = tqdm(total=total_iterations)
    
    with torch.no_grad():
        for batch_idx, sample_ in enumerate(loader_test):
            # Get the depth map and RGB image
            depth_map = sample_['dep'][0].cpu().numpy()  # Shape H x W
            # rgb_image = sample_['rgb'][0].cpu().numpy()  # Shape C x H x W
            # rgb_image = np.transpose(rgb_image, (1, 2, 0))  # Convert to H x W x C
            # # Normalize RGB image if necessary
            # rgb_image = rgb_image * 255.0  # Assuming the model expects inputs in [0, 255]
            # rgb_image = rgb_image.astype(np.uint8)
            
            # # Get ground truth depth map
            # gt_depth_map = sample_['gt'][0].cpu().numpy()
            
            # Get camera intrinsics K (fx, fy, cx, cy)
            K = sample_['K'][0].cpu().numpy()[0]  # Assuming K is 1 x 4 tensor
            fx, fy, cx, cy = K[0], K[1], K[2], K[3]
            
            for roll_deg, pitch_deg, yaw_deg in rotation_combinations:
                # Apply rotations
                depth_map_rot = np.squeeze(np.array(depth_map.copy()))# Also converted to numpy
                depth_map_rot = apply_combined_rotation_to_depth_map(
                    depth_map_rot, roll_deg, pitch_deg, yaw_deg, fx, fy, cx, cy
                )
                
                # Cleaning the Maps
                dep_np = depth_map_rot
                dep_clear, _ = out_removal(dep_np)
                dep_clear = np.expand_dims(dep_clear, 0)
                dep_clear_torch = torch.from_numpy(dep_clear)
                
                ### 
                # ip_basic fill
                dep_np_ip = np.copy(depth_map_rot)
                dep_ip = filll(dep_np_ip, max_depth=100.0,
                                    extrapolate=True, blur_type='gaussian')
                dep_ip_torch = torch.from_numpy(dep_ip)
                
                # Convert depth map to tensor
                depth_map_rot = torch.from_numpy(depth_map_rot).unsqueeze(0)
                ###
                
                # Prepare sample for model
                # samplep = {key: val.float().cuda() for key, val in sample_.items()
                #     if torch.is_tensor(val) and key != 'K'}
                # samplep['ip'] = dep_ip_torch
                # samplep['dep_clear'] = dep_clear_torch
                
                torch.cuda.synchronize()
                t0 = time.time()
                
                samplep = {
                    'dep': depth_map_rot.float().cuda(),
                    'dep_clear' : dep_clear_torch.float().cuda(),
                    'gt' : sample_['gt'].squeeze(0).float().cuda(),
                    'rgb' : sample_['rgb'].squeeze(0).float().cuda(),
                    'ip' : dep_ip_torch.float().cuda(),
                    'd_path' : sample_['d_path']
                }
                
                # Check all the tensor shapes
                for key, val in samplep.items():
                    if torch.is_tensor(val):
                        print(key, val.shape)
                    else:
                        print(key, val)
                
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
                    'roll': roll,
                    'pitch': pitch,
                    'yaw': yaw,
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
    # # Parse command-line arguments
    # arg = argparse.ArgumentParser(description='depth completion')
    # arg.add_argument('-p', '--project_name', type=str, default='inference')
    # arg.add_argument('-c', '--configuration', type=str, default='val_iccv.yml')
    # args = arg.parse_args()
    
    # # # Set CUDA devices
    # # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpus))

    # # Run the test function
    # print(args)
    test(config)



'''
Depricated Zone:


def apply_roll_to_image(image, phi_deg, fx, fy, cx, cy):
    phi = np.deg2rad(phi_deg)
    h, w = image.shape[:2]
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    u_centered = u_coords - cx
    v_centered = v_coords - cy
    u_prime = u_centered * (1 - phi * v_centered / fy) + cx
    v_prime = v_coords - fy * phi
    map_x = u_prime.astype(np.float32)
    map_y = v_prime.astype(np.float32)
    transformed_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return transformed_image

def apply_pitch_to_image(image, theta_deg, fx, fy, cx, cy):
    theta = np.deg2rad(theta_deg)
    h, w = image.shape[:2]
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    u_prime = u_coords - fx * theta  # Corrected sign
    v_prime = v_coords
    map_x = u_prime.astype(np.float32)
    map_y = v_prime.astype(np.float32)
    transformed_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return transformed_image

def apply_yaw_to_image(image, psi_deg, fx, fy, cx, cy):
    psi = np.deg2rad(psi_deg)
    h, w = image.shape[:2]
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    u_centered = u_coords - cx
    v_centered = v_coords - cy
    u_prime = u_coords - psi * (fx / fy) * v_centered
    v_prime = v_coords + psi * (fy / fx) * u_centered
    map_x = u_prime.astype(np.float32)
    map_y = v_prime.astype(np.float32)
    transformed_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return transformed_image

'''