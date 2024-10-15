import argparse

arg = argparse.ArgumentParser(description='depth completion')
arg.add_argument('-p', '--project_name', type=str, default='inference')
arg.add_argument('-c', '--configuration', type=str, default='val_iccv.yml')
arg = arg.parse_args()
from configs import get as get_cfg
config = get_cfg(arg)
# Extract configuration filename
config_name = arg.configuration.split('.')[0] # name example: val_lrru_mini_kitti

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
import sqlite3
import time
from datetime import datetime

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

def get_dataset_model_info(config_name):
    """Extract dataset name and model size from configuration filename."""
    parts = config_name.split('_')
    dataset_name = parts[-1].capitalize()  # Example: "kitti" -> "Kitti"
    model_size = parts[-2]  # Example: "mini"
    return dataset_name, model_size

def check_existing_sample(conn, image_path, roll, pitch, yaw):
    """Check if a sample with the given roll, pitch, yaw, and image path exists."""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT COUNT(*) FROM evaluations WHERE image_path = ? AND roll = ? AND pitch = ? AND yaw = ?
    ''', (image_path, roll, pitch, yaw))
    return cursor.fetchone()[0] > 0  # Returns True if the sample exists

def create_or_connect_db(config):
    """Create or connect to the SQLite database based on the config."""
    # If a valid db_path is given, use it
    if config.db_path and os.path.exists(config.db_path):
        db_path = config.db_path
        print(f"Connecting to existing database at {db_path}...")
    else:
        # Else create a new database with a timestamped folder
        dataset_name, model_size = get_dataset_model_info(config_name)
        # timestamp = datetime.now().strftime('%Y%m%d')
        rpy_range = f"{config.rotation_start}_{config.rotation_stop}_{config.rotation_step}"
        # eval_folder = f"./eval_datasets/{timestamp}"
        eval_folder = f"./eval_datasets/"
        os.makedirs(eval_folder, exist_ok=True)
        db_path = os.path.join(eval_folder, f"{dataset_name}_eval_{model_size}_({rpy_range}).db")
        print(f"Creating new database at {db_path}...")

    # Create database connection
    conn = sqlite3.connect(db_path)
    return conn

def setup_database(conn):
    """Set up the necessary table if it doesn't exist."""
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                roll REAL NOT NULL,
                pitch REAL NOT NULL,
                yaw REAL NOT NULL,
                RMSE REAL,
                MAE REAL,
                iRMSE REAL,
                iMAE REAL,
                REL REAL,
                D1 REAL,
                D2 REAL,
                D3 REAL,
                UNIQUE(image_path, roll, pitch, yaw)
            )
        ''')

def insert_evaluation_results_batch(conn, results_list):
    """Insert accumulated results into the database in one transaction."""
    cursor = conn.cursor()
    cursor.executemany('''
        INSERT OR IGNORE INTO evaluations (
            image_path, roll, pitch, yaw, RMSE, MAE, iRMSE, iMAE, REL, D1, D2, D3
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', results_list)
    conn.commit()  # Commit after all the rows have been inserted in a batch

def test(args):

    # Initialize database connection
    conn = create_or_connect_db(config)
    setup_database(conn)
    
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
    
    num_samples = len(loader_test)
    total_iterations = num_samples * len(rotation_combinations)
    pbar_ = tqdm(total=total_iterations)

    # Found starting point flag, don't have to query database to check if it exists
    flag = True
    
    with torch.no_grad():
        for batch_idx, sample_ in enumerate(loader_test):

            # Path for the depth image
            depth_image_name_path = sample_['d_path'][0]

            # # Check if this image already exists in the database
            # roll_deg, pitch_deg, yaw_deg = rotation_combinations[0]
            # if check_existing_sample(conn, depth_image_name_path, roll_deg, pitch_deg, yaw_deg):
            #     print(f"Skipping already processed image-pointcloud pair: {depth_image_name_path}")
            #     continue

            torch.cuda.synchronize()
            t0 = time.time()

            # Row Entries List
            row_entries = []

            # Get the depth map and RGB image
            depth_map = sample_['dep'][0].cpu().numpy()  # Shape H x W
            
            # Get camera intrinsics K (fx, fy, cx, cy)
            K = sample_['K'][0].cpu().numpy()[0]  # Assuming K is 1 x 4 tensor
            fx, fy, cx, cy = K[0], K[1], K[2], K[3]

            for roll_deg, pitch_deg, yaw_deg in rotation_combinations:

                # Check if this specific sample already exists in the database
                if flag and check_existing_sample(conn, depth_image_name_path, roll_deg, pitch_deg, yaw_deg):
                    # print(f"Skipping already processed combination: {depth_image_name_path}, Roll: {roll_deg}, Pitch: {pitch_deg}, Yaw: {yaw_deg}")
                    pbar_.update(1)
                    continue
                else:
                    flag = False

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
                
                samplep = {
                    'dep': depth_map_rot.unsqueeze(0).float().cuda(),
                    'dep_clear' : dep_clear_torch.unsqueeze(0).float().cuda(),
                    'gt' : sample_['gt'].float().cuda(),
                    'rgb' : sample_['rgb'].float().cuda(),
                    'ip' : dep_ip_torch.float().cuda(),
                    'd_path' : sample_['d_path']
                }
                
                
                # Check all the tensor shapes
                # for key, val in samplep.items():
                #     if torch.is_tensor(val):
                #         print(key, val.shape)
                #     else:
                #         print(key, val)
                
                # Run the model
                output_ = net(samplep)
                
                torch.cuda.synchronize()
                t1 = time.time()
                
                # Evaluate
                if 'test' not in args.test_option:
                    metric_test = metric.evaluate(output_['results'][-1], samplep['gt'], 'test')
                else:
                    metric_test = metric.evaluate(output_['results'][-1], samplep['dep'], 'test')
                
                # if roll_deg == 0 and pitch_deg ==0 and yaw_deg ==0:
                #     print("Baseline Reached!, ", metric_test.data.cpu().numpy())
                # else:
                #     print(f"{roll_deg}, {pitch_deg}, {yaw_deg}", metric_test)

                # Add results for the current rotation to the batch list
                row_entries.append((
                    depth_image_name_path, roll_deg, pitch_deg, yaw_deg,
                    metric_test.data.cpu().numpy()[0, 0],  # RMSE
                    metric_test.data.cpu().numpy()[0, 1],  # MAE
                    metric_test.data.cpu().numpy()[0, 2],  # iRMSE
                    metric_test.data.cpu().numpy()[0, 3],  # iMAE
                    metric_test.data.cpu().numpy()[0, 4],  # REL
                    metric_test.data.cpu().numpy()[0, 5],  # D^1
                    metric_test.data.cpu().numpy()[0, 6],  # D^2
                    metric_test.data.cpu().numpy()[0, 7],  # D^3
                ))
                
                pbar_.update(1)
            
            # After processing all rotations for the current image, insert the results into the database in a batch
            insert_evaluation_results_batch(conn, row_entries)

        pbar_.close()
        
    # Close the database connection
    conn.close()

if __name__ == "__main__":
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