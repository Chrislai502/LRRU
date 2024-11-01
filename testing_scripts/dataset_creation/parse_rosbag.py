import numpy as np
import cv2
import yaml
import os
import re
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2 as pc2
from tqdm import tqdm

class ParseRosbag:
    def __init__(self, config):
        self.num_pairs_found = 0
        self.num_pairs_stored = 0
        self.last_stored_timestamp = None
        self.bridge = CvBridge()
        self.config = config

        self.K, self.D = None, None
        self.new_K, self.str_K, self.roi = None, None, None
        self.dataset_path = None

        self.backlog = []
        self.video_writer = None
        self.start_time = None

    def parse_rosbag(self):
        last_camera, last_lidar = None, None
        prev_t = 0

        # Create dataset directory architecture
        self.dataset_path = os.path.join(self.config['out_folder'], self.config['dataset'], 'val_selection_cropped')
        os.makedirs(self.dataset_path, exist_ok=True)
        for subdir in ('groundtruth_depth', 'image', 'intrinsics', 'velodyne_raw'):
            os.makedirs(os.path.join(self.dataset_path, subdir), exist_ok=True)

        # Open rosbag
        storage_options, converter_options = self.get_rosbag_options_read(self.config['rosbag'])
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

        metadata = reader.get_metadata()

        # Loop through messages
        with tqdm(total=metadata.message_count, postfix={ 'num_pairs': self.num_pairs_stored }) as bar:
            while reader.has_next() and (self.config['num_points'] is None or self.num_pairs_stored < self.config['num_points']):
                topic, data, t = reader.read_next()

                assert prev_t < t
                prev_t = t

                # Get camera info
                if self.K is None and topic == f'/{config["camera_frame"]}/camera_info':
                    msg_type = get_message(type_map[topic])
                    msg = deserialize_message(data, msg_type)
                    self.K, self.D = np.array(list(msg.k)).reshape((3, 3)), np.array(list(msg.d))

                    # Process backlog
                    for t, camera, lidar in self.backlog:
                        self.write_data(t, camera, lidar, bar)

                # Process camera or lidar data
                elif topic in (f'/{config["camera_frame"]}/image', f'/{config["lidar_frame"]}/points'):
                    msg_type = get_message(type_map[topic])
                    msg = deserialize_message(data, msg_type)

                    # Process camera data
                    if topic == f'/{config["camera_frame"]}/image':
                        camera = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

                        # Check if synced
                        if last_lidar is not None and abs(t - last_lidar[0]) <= config['time_tol'] * 10**9:
                            if self.num_pairs_found % config['sample_rate'] == 0:
                                self.write_data(last_lidar[0], camera, last_lidar[1], bar)
                            self.num_pairs_found += 1
                            last_lidar, last_camera = None, None
                        else:
                            last_camera = (t, camera)
                    
                    # Process lidar data
                    if topic == f'/{config["lidar_frame"]}/points':
                        pcd = pc2.read_points(msg, field_names=['x', 'y', 'z'])
                        lidar = np.array([pcd['x'], pcd['y'], pcd['z']]).T

                        # Check if synced
                        if last_camera is not None and abs(t - last_camera[0]) <= config['time_tol'] * 10**9:
                            if self.num_pairs_found % config['sample_rate'] == 0:
                                self.write_data(last_camera[0], last_camera[1], lidar, bar)
                            self.num_pairs_found += 1
                            last_lidar, last_camera = None, None
                        else:
                            last_lidar = (t, lidar)
                
                bar.update()
        
        if self.config['overlay']['create'] and self.video_writer is not None:
            self.video_writer.release()

        self.write_metadata(metadata)
    
    def write_metadata(self, metadata):
        with open(os.path.join(self.config['out_folder'], self.config['dataset'], 'metadata.yaml'), 'w') as file:
            basename = os.path.basename(config['rosbag'])
            match_split_mcap = re.match(r"^rosbag2_\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}_(\d+)\.mcap$", basename)

            if match_split_mcap:
                index = int(match_split_mcap.group(1))
            else:
                index = 'None'

            yaml.dump({
                'rosbag_name': basename,
                'starting_time': metadata.starting_time.nanoseconds,
                'index': index,
                'data_count': self.num_pairs_stored,
                'camera_frame': self.config['camera_frame'],
                'lidar_frame': self.config['lidar_frame'],
                'K': self.new_K.flatten().tolist()
            }, file, sort_keys=False)

    def get_rosbag_options_read(self, path, serialization_format='cdr'):
        storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='mcap')

        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format=serialization_format,
            output_serialization_format=serialization_format)

        return storage_options, converter_options

    def write_data(self, t, camera, lidar, bar):
        # If no K matrix yet, then save for later
        if self.K is None:
            self.backlog.append((t, camera, lidar))
            return

        # Get new K matrix with distortion
        if self.num_pairs_stored == 0:
            h, w = camera.shape[:2]
            self.new_K, self.roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h), 1, (w, h))
            self.str_K = " ".join([str(float(f"{val:.4f}\n")) for val in self.new_K.flatten()])

        # Undistort and crop image
        x, y, w, h = self.roi
        undistort_camera = cv2.undistort(camera, self.K, self.D, None, self.new_K)[y:y+h, x:x+w]

        # Create blank lidar image
        lidar_im = np.zeros(undistort_camera.shape[:2], dtype=np.uint16)

        # Project lidar points into image
        lidar_in_cam = (self.config['transform']['R'] @ lidar.T).T + self.config['transform']['T']
        lidar_in_image_3d = (self.new_K @ lidar_in_cam.T).T
        lidar_in_im = np.array([
            lidar_in_image_3d[:, 0] / lidar_in_image_3d[:, 2],
            lidar_in_image_3d[:, 1] / lidar_in_image_3d[:, 2]
        ]).T - np.array((x, y))
        within_im_mask = (lidar_in_im[:, 0] >= 0) & (lidar_in_im[:, 0] < lidar_im.shape[1]) & (lidar_in_im[:, 1] >= 0) & (lidar_in_im[:, 1] < lidar_im.shape[0])
        lidar_proj = lidar_in_im[within_im_mask].astype(int)

        # Populate lidar image
        lidar_within_im = lidar_in_cam[within_im_mask]
        depths = np.linalg.norm(lidar_within_im, axis=1)
        depths[depths >= 256] = 0
        lidar_im[lidar_proj[:, 1], lidar_proj[:, 0]] = depths * 256

        # Write data
        if self.num_pairs_stored >= 10**10:
            datapoint_name = f"{self.num_pairs_stored}"
        else:
            datapoint_name = f"{self.num_pairs_stored:0>10}"

        cv2.imwrite(os.path.join(self.dataset_path, 'groundtruth_depth', f'{datapoint_name}.png'), lidar_im)
        cv2.imwrite(os.path.join(self.dataset_path, 'image', f'{datapoint_name}.png'), undistort_camera)
        cv2.imwrite(os.path.join(self.dataset_path, 'velodyne_raw', f'{datapoint_name}.png'), lidar_im)

        with open(os.path.join(self.dataset_path, 'intrinsics', f'{datapoint_name}.txt'), "w") as file:
            file.write(self.str_K)
        
        # Create distance overlay
        if self.config['overlay']['create']:
            if self.video_writer is None:
                video_file = os.path.join(self.config['out_folder'], self.config['dataset'], 'overlay.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
                self.video_writer = cv2.VideoWriter(video_file, fourcc, self.config['overlay']['frame_rate'], (undistort_camera.shape[1], undistort_camera.shape[0]))
                self.start_time = t

            if self.last_stored_timestamp is not None and (config['overlay']['length'] is None or t - self.start_time < config['overlay']['length'] * 10**9):
                overlay_im = undistort_camera.copy()
                overlay = np.zeros_like(overlay_im)

                if np.any(within_im_mask):
                    depths_normalized = 255 * (depths - self.config['overlay']['depth_range'][0]) / (config['overlay']['depth_range'][1] - config['overlay']['depth_range'][0])
                    colors = cv2.applyColorMap(depths_normalized.astype(np.uint8)[:, np.newaxis], cv2.COLORMAP_HSV).squeeze(axis=1)
                    
                    for pt, color in zip(lidar_proj, colors):
                        overlay = cv2.circle(overlay, list(pt), self.config['overlay']['point_radius'], list(color.astype(float)), -1, 0)
                    
                    mask = overlay.astype(bool)
                    overlay_im[mask] = cv2.addWeighted(overlay_im, 1 - self.config['overlay']['opacity'], overlay, self.config['overlay']['opacity'], 0)[mask]
                
                num_frames = round((t - self.last_stored_timestamp) / 10**9 * self.config['overlay']['frame_rate'])

                for _ in range(num_frames):
                    self.video_writer.write(overlay_im)

            self.last_stored_timestamp = t

        # Update number of stored pairs
        self.num_pairs_stored += 1
        bar.set_postfix({ 'num_pairs': self.num_pairs_stored })

if __name__ == "__main__":
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Format config
    config['rosbag'] = os.path.normpath(config['rosbag'])
    if config['out_folder'] is None:
        config['out_folder'] = os.path.join('..', '..')
    config['transform']['R'] = np.array(config['transform']['R']).reshape((3, 3))
    config['transform']['T'] = np.array(config['transform']['T'])

    rosbag_parser = ParseRosbag(config)
    rosbag_parser.parse_rosbag()