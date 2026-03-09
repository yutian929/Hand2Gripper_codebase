#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import json
import os
import sys
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException

class RGBDPlayerAndMapperANDMasker(Node):
    def __init__(self):
        super().__init__('rgbd_player_mapper_masker')
        
        # --- Parameter Declaration ---
        self.declare_parameter('data_dir', '')
        self.declare_parameter('frequency', 30.0)
        self.declare_parameter('output_json', 'camera_link_traj.json')

        # --- Get Parameters ---
        data_dir = self.get_parameter('data_dir').get_parameter_value().string_value
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        output_filename = self.get_parameter('output_json').get_parameter_value().string_value
        
        if not data_dir:
            self.get_logger().error("Must specify 'data_dir' parameter!")
            sys.exit(1)

        # path setup
        self.rgb_path = os.path.join(data_dir, "video_L.mp4")
        self.depth_path = os.path.join(data_dir, "depth.npy")
        self.intrinsics_path = os.path.join(data_dir, "camera_intrinsics.json")
        self.mask_path = os.path.join(data_dir, "segmentation_processor", "masks_arm.npy")
        self.output_path = os.path.join(data_dir, output_filename)

        # --- Initialize TF Listener ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Data Loading ---
        self.load_intrinsics()
        
        self.get_logger().info(f"Loading depth from {self.depth_path}...")
        self.depth_data = np.load(self.depth_path)
        self.num_frames = self.depth_data.shape[0]
        
        self.get_logger().info(f"Loading masks from {self.mask_path}...")
        self.masks_data = np.load(self.mask_path)

        self.cap = cv2.VideoCapture(self.rgb_path)
        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open video: {self.rgb_path}")
            sys.exit(1)

        # --- Publisher ---
        self.pub_color = self.create_publisher(Image, 'camera/color/image_raw', 10)
        self.pub_depth = self.create_publisher(Image, 'camera/aligned_depth_to_color/image_raw', 10)
        self.pub_info = self.create_publisher(CameraInfo, 'camera/color/camera_info', 10)

        self.bridge = CvBridge()
        self.frame_idx = 0
        
        # --- [Modified] Store trajectory data ---
        # changed to dictionary structure, Key is frame_idx
        self.trajectory_record = {}

        # --- Timer ---
        self.timer = self.create_timer(1.0 / self.frequency, self.timer_callback)
        self.get_logger().info(f"Start playback and recording! Total frames: {self.num_frames}")

    def load_intrinsics(self):
        with open(self.intrinsics_path, 'r') as f:
            data = json.load(f)
            intr = data['color'] 

        self.camera_info = CameraInfo()
        self.camera_info.header.frame_id = "camera_color_optical_frame"
        self.camera_info.width = intr['width']
        self.camera_info.height = intr['height']
        self.camera_info.distortion_model = "plumb_bob"
        self.camera_info.d = intr['disto']
        self.camera_info.k = [intr['fx'], 0.0, intr['cx'], 0.0, intr['fy'], intr['cy'], 0.0, 0.0, 1.0]
        self.camera_info.p = [intr['fx'], 0.0, intr['cx'], 0.0, 0.0, intr['fy'], intr['cy'], 0.0, 0.0, 0.0, 1.0, 0.0]
        self.camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    def timer_callback(self):
        # 1. Check if finished
        if self.frame_idx >= self.num_frames:
            self.get_logger().info("Playback finished, saving data...")
            self.save_trajectory()
            self.timer.cancel()
            raise SystemExit 
            return

        # 2. Read images
        ret, frame_bgr = self.cap.read()
        if not ret:
            self.get_logger().warn("Video stream read failed")
            return
        
        # get depth map (use copy to avoid modifying original data)
        frame_depth_m = self.depth_data[self.frame_idx].copy()

        # --- [New] Mask processing logic ---
        if self.frame_idx < self.masks_data.shape[0]:
            # get current frame mask (bool type)
            mask = self.masks_data[self.frame_idx]
            
            # convert mask to uint8 (0 or 255) for morphological operations
            mask_uint8 = mask.astype(np.uint8) * 255
            
            # 1. Dilate Mask (expand outward by 10 pixels)
            kernel = np.ones((10, 10), np.uint8) 
            dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)

            # 2. Create Boolean Indexing
            invalid_region = dilated_mask > 0

            # 3. Process depth map (set occluded area depth to 0)
            frame_depth_m[invalid_region] = 0.0

            # 4. Process RGB image (black out occluded areas)
            frame_bgr[invalid_region] = 0
        # ---------------------------

        # 3. construct messages
        now = self.get_clock().now()
        now_msg = now.to_msg() 

        msg_color = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
        msg_color.header.stamp = now_msg
        msg_color.header.frame_id = "camera_color_optical_frame"

        msg_depth = self.bridge.cv2_to_imgmsg(frame_depth_m, encoding="32FC1")
        msg_depth.header.stamp = now_msg
        msg_depth.header.frame_id = "camera_color_optical_frame"

        self.camera_info.header.stamp = now_msg

        # 4. publish
        self.pub_color.publish(msg_color)
        self.pub_depth.publish(msg_depth)
        self.pub_info.publish(self.camera_info)

        # ==========================================
        # 5. [Modified] Listen to TF and set default values
        # ==========================================
        
        # Default values: Identity Pose
        tx, ty, tz = 0.0, 0.0, 0.0
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        tf_ts = None # None means no valid TF obtained
        got_tf = False

        try:
            # query map -> camera_link
            if self.tf_buffer.can_transform('map', 'camera_link', Time()):
                t = self.tf_buffer.lookup_transform(
                    'map', 
                    'camera_link', 
                    Time()
                )
                
                # update to real values
                tx = t.transform.translation.x
                ty = t.transform.translation.y
                tz = t.transform.translation.z
                qx = t.transform.rotation.x
                qy = t.transform.rotation.y
                qz = t.transform.rotation.z
                qw = t.transform.rotation.w
                
                tf_ts = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
                got_tf = True
        except TransformException:
            # keep default values
            pass

        # 6. [Modified] Record data (large dictionary structure)
        self.trajectory_record[self.frame_idx] = {
            "img_timestamp": now.nanoseconds * 1e-9,
            "pose": {
                "tx": tx, "ty": ty, "tz": tz,
                "qx": qx, "qy": qy, "qz": qz, "qw": qw
            },
            "tf_timestamp": tf_ts
        }

        # progress print
        if self.frame_idx % 30 == 0:
            status = "Tracking" if got_tf else "Waiting/Lost (Using Default 0)"
            self.get_logger().info(f"Frame {self.frame_idx}/{self.num_frames} - {status}")

        self.frame_idx += 1

    def save_trajectory(self):
        """save record to JSON"""
        try:
            with open(self.output_path, 'w') as f:
                # use indent=4 for formatted output
                json.dump(self.trajectory_record, f, indent=4)
            self.get_logger().info(f"Trajectory successfully saved to: {self.output_path}")
            self.get_logger().info(f"Total recorded {len(self.trajectory_record)} frame data")
            self.get_logger().info(f">>> VSLAM PLAYBACK COMPLETE <<<")
        except Exception as e:
            self.get_logger().error(f"Failed to save JSON: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RGBDPlayerAndMapperANDMasker()
    
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        node.get_logger().info("Received interrupt signal, saving existing data...")
        node.save_trajectory()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()