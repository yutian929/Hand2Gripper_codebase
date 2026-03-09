#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import json
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from rclpy.time import Time

class RGBDPlayer(Node):
    def __init__(self):
        super().__init__('rgbd_player')
        
        # declare parameters: data path
        self.declare_parameter('data_dir', '')
        self.declare_parameter('frequency', 30.0)
        self.declare_parameter('loop', False) # whether to loop playback

        data_dir = self.get_parameter('data_dir').get_parameter_value().string_value
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.loop = self.get_parameter('loop').get_parameter_value().bool_value

        if not data_dir:
            self.get_logger().error("Please specify 'data_dir' parameter!")
            return

        # path concatenation
        self.rgb_path = os.path.join(data_dir, "video_L.mp4")
        self.depth_path = os.path.join(data_dir, "depth.npy")
        self.intrinsics_path = os.path.join(data_dir, "camera_intrinsics.json")

        # 1. load intrinsics
        self.load_intrinsics()

        # 2. load depth data (N, H, W) float32
        self.get_logger().info(f"Loading depth from {self.depth_path}...")
        self.depth_data = np.load(self.depth_path)
        self.num_frames = self.depth_data.shape[0]
        self.get_logger().info(f"Loaded {self.num_frames} depth frames.")

        # 3. Open video
        self.cap = cv2.VideoCapture(self.rgb_path)
        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open video: {self.rgb_path}")
            return

        # 4. create publishers (Publisher)
        # use generic names, remap later in Launch to match your rosgraph.pdf
        self.pub_color = self.create_publisher(Image, 'camera/color/image_raw', 10)
        self.pub_depth = self.create_publisher(Image, 'camera/aligned_depth_to_color/image_raw', 10)
        self.pub_info = self.create_publisher(CameraInfo, 'camera/color/camera_info', 10)

        self.bridge = CvBridge()
        self.frame_idx = 0
        
        # timer
        self.timer = self.create_timer(1.0 / self.frequency, self.timer_callback)
        self.get_logger().info("RGB-D Playback Started.")

    def load_intrinsics(self):
        """parse JSON and build CameraInfo message"""
        with open(self.intrinsics_path, 'r') as f:
            data = json.load(f)
            # because alignment was done during collection, we only need Color camera intrinsics
            intr = data['color'] 

        self.camera_info = CameraInfo()
        self.camera_info.header.frame_id = "camera_color_optical_frame" # corresponding to frames.pdf
        self.camera_info.width = intr['width']
        self.camera_info.height = intr['height']
        self.camera_info.distortion_model = "plumb_bob"
        
        # D: distortion coefficients
        self.camera_info.d = intr['disto']

        # K: intrinsic matrix 3x3
        self.camera_info.k = [
            intr['fx'], 0.0, intr['cx'],
            0.0, intr['fy'], intr['cy'],
            0.0, 0.0, 1.0
        ]

        # P: projection matrix 3x4 (for undistorted images, usually same as K but pad 0 in 4th column)
        self.camera_info.p = [
            intr['fx'], 0.0, intr['cx'], 0.0,
            0.0, intr['fy'], intr['cy'], 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        
        # R: rotation matrix (identity)
        self.camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    def timer_callback(self):
        if self.frame_idx >= self.num_frames:
            if self.loop:
                self.frame_idx = 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.get_logger().info("Looping video...")
            else:
                self.get_logger().info("Playback finished.")
                self.timer.cancel()
                return

        # 1. read RGB frame
        ret, frame_bgr = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to read video frame (End of stream?)")
            return

        # 2. read Depth frame
        # your npy stores (N, H, W) in meters, float32
        frame_depth_m = self.depth_data[self.frame_idx]

        # 3. construct ROS messages
        now = self.get_clock().now().to_msg()
        
        # RGB Message
        msg_color = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
        msg_color.header.stamp = now
        msg_color.header.frame_id = "camera_color_optical_frame"

        # Depth Message
        # RTAB-MAP accepts 32FC1 (meters) or 16UC1 (millimeters). Your data is in meters, 32FC1 is most convenient and highest precision.
        msg_depth = self.bridge.cv2_to_imgmsg(frame_depth_m, encoding="32FC1")
        msg_depth.header.stamp = now
        msg_depth.header.frame_id = "camera_color_optical_frame"

        # Camera Info Message
        self.camera_info.header.stamp = now

        # 4. publish
        self.pub_color.publish(msg_color)
        self.pub_depth.publish(msg_depth)
        self.pub_info.publish(self.camera_info)

        self.frame_idx += 1

def main(args=None):
    rclpy.init(args=args)
    node = RGBDPlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()