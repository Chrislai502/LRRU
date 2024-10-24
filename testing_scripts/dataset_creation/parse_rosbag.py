import numpy as np
import cv2
import yaml
import os
import json
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2 as pc2

def get_rosbag_options_read(path, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='mcap')

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options

def main(config):
    last_camera, last_lidar = None, None
    camera_lidar_pairs = []
    K, D = None, None
    prev_t = 0
    bridge = CvBridge()

    # Create dataset directory architecture
    dataset_path = os.path.join('..', '..', 'data', config['dataset'], 'val_selection_cropped')
    os.makedirs(dataset_path, exist_ok=True)
    for subdir in ('groundtruth_depth', 'image', 'intrinsics', 'velodyne_raw'):
        os.makedirs(os.path.join(dataset_path, subdir), exist_ok=True)

    # Loop through rosbag
    storage_options, converter_options = get_rosbag_options_read(config['rosbag'])
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    while reader.has_next():
        topic, data, t = reader.read_next()
        msg_type = get_message(type_map[topic])

        assert prev_t < t
        prev_t = t

        if K is None and topic == f'{config["camera_frame"]}/camera_info':
            msg = deserialize_message(data, msg_type)
            K, D = np.array(list(msg.k)).reshape((3, 3)), np.array(list(msg.d))

        elif topic in (f'{config["camera_frame"]}/image', f'{config["lidar_frame"]}/points'):
            msg = deserialize_message(data, msg_type)

            if topic == f'{config["camera_frame"]}/image':
                camera = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

                if last_lidar is not None and abs(t - last_lidar[0]) <= config['time_tol']:
                    camera_lidar_pairs.append((last_lidar[0], camera, last_lidar))
                    last_lidar, last_camera = None, None
                else:
                    last_camera = (t, camera)
            
            if topic == f'{config["lidar_frame"]}/points':
                pcd = pc2.read_points(msg, field_names=['x', 'y', 'z'])
                lidar = np.array([pcd['x'], pcd['y'], pcd['z']]).T

                if last_camera is not None and abs(t - last_camera[0]) <= config['time_tol']:
                    camera_lidar_pairs.append((last_camera[0], last_camera, lidar))
                    last_lidar, last_camera = None, None
                else:
                    last_lidar = (t, lidar)
    
    # Loop through synced pairs
    for i, pair in enumerate(camera_lidar_pairs):
        t, camera, lidar = pair
        lidar_im = np.zeros(*camera.size[:2], dtype=np.uint16)

        lidar_in_cam = (config['transform']['R'] @ lidar.T).T + config['transform']['T']
        lidar_in_image_3d = (np.array(K).reshape(3, 3) @ lidar_in_cam.T).T
        lidar_in_im = np.array([
            lidar_in_image_3d[:, 0] / lidar_in_image_3d[:, 2],
            lidar_in_image_3d[:, 1] / lidar_in_image_3d[:, 2]
        ]).T
        lidar_within_im = np.all([
            lidar_in_im[:, 0] >= 0,
            lidar_in_im[:, 0] < lidar_im.shape[1],
            lidar_in_im[:, 1] >= 0,
            lidar_in_im[:, 1] < lidar_im.shape[0]
        ], axis=0)

        lidar_im[lidar_within_im] = np.linalg.norm(lidar_in_cam, axis=1) * 256

        cv2.imwrite(os.path.join(dataset_path, 'groundtruth_depth', f'{i}.png'), lidar_im)
        cv2.imwrite(os.path.join(dataset_path, 'image', f'{i}.png'), camera)
        np.savetxt(os.path.join(dataset_path, 'intrinsics', f'{i}.txt'), K.flatten())
        cv2.imwrite(os.path.join(dataset_path, 'velodyne_raw', f'{i}.png'), lidar_im)


if __name__ == "__main__":
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    config['transform']['R'] = np.array(config['transform']['R']).reshape((3, 3))
    config['transform']['T'] = np.array(config['transform']['T'])

    main(config)