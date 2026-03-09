import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import json
import os
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from rclpy.qos import QoSProfile
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R  # use scipy to handle quaternion
import message_filters

def camera_info_to_dict(camera_info):
    """Convert CameraInfo message to dictionary"""
    return {
        'height': camera_info.height,
        'width': camera_info.width,
        'distortion_model': camera_info.distortion_model,
        'd': list(camera_info.d),  # distortion coefficients
        'k': list(camera_info.k),  # intrinsic matrix
        'r': list(camera_info.r),  # rectification matrix
        'p': list(camera_info.p),  # projection matrix
        'binning_x': camera_info.binning_x,
        'binning_y': camera_info.binning_y,
        'roi': {
            'x_offset': camera_info.roi.x_offset,
            'y_offset': camera_info.roi.y_offset,
            'height': camera_info.roi.height,
            'width': camera_info.roi.width,
            'do_rectify': camera_info.roi.do_rectify
        }
    }

class CameraDataSaver(Node):
    def __init__(self):
        super().__init__('camera_data_saver')

        # set subscription topics
        self.bridge = CvBridge()
        self.image_count = 0
        self.max_images = 1000
        self.is_alive = True

        # create save folders
        self.root_dir = 'camera_data'
        os.makedirs(self.root_dir, exist_ok=True)
        self.save_dir_color = os.path.join(self.root_dir, 'color')
        self.save_dir_depth = os.path.join(self.root_dir, 'depth')
        self.save_dir_traj = os.path.join(self.root_dir, 'traj')
        os.makedirs(self.save_dir_color, exist_ok=True)
        os.makedirs(self.save_dir_depth, exist_ok=True)
        os.makedirs(self.save_dir_traj, exist_ok=True)

        # QoS policy
        qos_profile = QoSProfile(depth=10)

        # subscribe to camera info, save only once
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, qos_profile)
        self.camera_info_saved = False

        # for temporarily storing synchronized messages
        self.latest_rgb_msg = None
        self.latest_depth_msg = None

        # use message_filters to synchronize RGB and depth images
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.sync_callback)

        # create Transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # timer, call save function every 0.1 seconds
        self.timer = self.create_timer(0.1, self.timer_save_callback)

    def camera_info_callback(self, msg: CameraInfo):
        """receive camera info and save once"""
        if not self.camera_info_saved:
            camera_info_dict = camera_info_to_dict(msg)
            camera_info_filename = os.path.join(self.root_dir, 'camera_info.json')
            with open(camera_info_filename, 'w') as f:
                json.dump(camera_info_dict, f, indent=4)
            self.camera_info_saved = True
            self.get_logger().info("Camera info saved.")
            # destroy subscription, as it is only needed once
            self.destroy_subscription(self.camera_info_sub)

    def sync_callback(self, rgb_msg: Image, depth_msg: Image):
        """synchronization callback, temporarily store aligned messages"""
        self.latest_rgb_msg = rgb_msg
        self.latest_depth_msg = depth_msg

    def timer_save_callback(self):
        """timer callback, for saving data"""
        if self.image_count >= self.max_images:
            if self.is_alive:  # avoid repeated printing when node is shutting down
                self.get_logger().info("Reached max image count. Shutting down.")
                self.is_alive = False
                self.destroy_node()
            return

        if not self.camera_info_saved:
            self.get_logger().warn("Waiting for camera info...")
            return

        # check if there are new synchronized messages
        if self.latest_rgb_msg is None or self.latest_depth_msg is None:
            self.get_logger().warn("No sync data updated...")
            return

        # read and clear cached messages to avoid duplicate saving
        rgb_msg = self.latest_rgb_msg
        depth_msg = self.latest_depth_msg
        self.latest_rgb_msg = None
        self.latest_depth_msg = None

        try:
            # get pose aligned with image timestamp
            transform = self.tf_buffer.lookup_transform('map', 'camera_link', rclpy.time.Time())
            pose = transform.transform.translation
            rotation = transform.transform.rotation

            # get 4x4 matrix pose
            pose_matrix = self.get_pose_matrix(pose, rotation)

            # convert images
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
            depth_image = depth_image / 1000.0  # convert to meters

            # save RGB image
            rgb_filename = os.path.join(self.save_dir_color, f'{self.image_count}.png')
            cv2.imwrite(rgb_filename, rgb_image)

            # save depth image
            depth_filename = os.path.join(self.save_dir_depth, f'{self.image_count}.npy')
            np.save(depth_filename, depth_image)

            # save camera pose
            pose_filename = os.path.join(self.save_dir_traj, f'{self.image_count}.npy')
            np.save(pose_filename, pose_matrix)

            self.get_logger().info(f"Saved aligned data frame {self.image_count}")

            # update counter
            self.image_count += 1

        except Exception as e:
            self.get_logger().error(f"Error saving data: {e}")

    def get_pose_matrix(self, translation, rotation):
        """convert pose to 4x4 matrix"""
        pose_matrix = np.eye(4)
        pose_matrix[0, 3] = translation.x
        pose_matrix[1, 3] = translation.y
        pose_matrix[2, 3] = translation.z
        pose_matrix[:3, :3] = self.quaternion_to_rotation_matrix(rotation)
        return pose_matrix

    def quaternion_to_rotation_matrix(self, q):
        """convert quaternion to rotation matrix"""
        r = R.from_quat([q.x, q.y, q.z, q.w])  # use scipy to handle quaternion
        return r.as_matrix()  # return 3x3 rotation matrix

def main(args=None):
    rclpy.init(args=args)
    node = CameraDataSaver()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
