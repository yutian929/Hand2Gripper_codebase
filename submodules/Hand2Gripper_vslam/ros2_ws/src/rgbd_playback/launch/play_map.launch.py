import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    # 1. Declare data path parameter
    # Default path set to the directory you mentioned, can be overridden at runtime with data_dir:=...
    data_dir_arg = DeclareLaunchArgument(
        'data_dir',
        default_value='./output', 
        description='Path to directory containing video_L.mp4, depth.npy and camera_intrinsics.json'
    )
    
    data_dir = LaunchConfiguration('data_dir')

    # 2. RTAB-MAP General Parameter Configuration
    # frame_id: robot base coordinate system (usually camera_link)
    # subscribe_depth: use RGB-D mode
    # approx_sync: False (we use Python script to achieve precise timestamp synchronization)
    rtabmap_parameters = [{
        'frame_id': 'camera_link',
        'subscribe_depth': True,
        'subscribe_odom_info': True,
        'approx_sync': True,
        'wait_imu_to_init': False,
        'queue_size': 30,
        'publish_tf': True  # let rtabmap publish map -> odom TF
    }]

    # 3. Topic Remapping Configuration
    # Left side is RTAB-MAP node standard input name, right side is actual Topic name sent by our player node
    rtabmap_remappings = [
        ('rgb/image',       '/camera/camera/color/image_raw'),
        ('rgb/camera_info', '/camera/camera/color/camera_info'),
        ('depth/image',     '/camera/camera/aligned_depth_to_color/image_raw')
    ]

    return LaunchDescription([
        data_dir_arg,

        # ---------------------------------------------------------
        # Node A: Static TF Publishing (Critical: Coordinate System Transformation)
        # ---------------------------------------------------------
        # Convert camera_link (X forward, Y left, Z up) to camera_color_optical_frame (Z forward, X right, Y down)
        # Quaternion [-0.5, 0.5, -0.5, 0.5] is the standard ROS to Optical rotation transformation
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_link_to_optical',
            arguments=['0', '0', '0', '-0.5', '0.5', '-0.5', '0.5', 'camera_link', 'camera_color_optical_frame']
        ),

        # ---------------------------------------------------------
        # Node B: Custom Data Player + Recorder (replaces the original player)
        # ---------------------------------------------------------
        Node(
            package='rgbd_playback',
            # [Modified] Executable changed to new name registered in setup.py
            executable='player_mapper', 
            name='rgbd_player_mapper_node',
            output='screen',
            parameters=[{
                'data_dir': data_dir,
                'frequency': 30.0,
                # [Note] Removed loop parameter here, because the goal is offline evaluation, save and exit after one run
                'output_json': 'camera_link_traj.json' # results will be saved in data_dir
            }],
            # Remappings remain unchanged
            remappings=[
                ('camera/color/image_raw', '/camera/camera/color/image_raw'),
                ('camera/aligned_depth_to_color/image_raw', '/camera/camera/aligned_depth_to_color/image_raw'),
                ('camera/color/camera_info', '/camera/camera/color/camera_info')
            ]
        ),

        # ---------------------------------------------------------
        # Node C: RGBD Odometry (Frontend: Visual Odometry)
        # ---------------------------------------------------------
        # Input: RGB image + depth image
        # Output: /odom (TF: odom -> camera_link)
        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            output='screen',
            parameters=rtabmap_parameters,
            remappings=rtabmap_remappings
        ),

        # ---------------------------------------------------------
        # Node D: RTAB-MAP SLAM (Backend: Mapping and Loop Closure)
        # ---------------------------------------------------------
        # Input: RGB image + depth image + odometry
        # Output: /mapData, /grid_map (TF: map -> odom)
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            output='screen',
            parameters=rtabmap_parameters,
            remappings=rtabmap_remappings,
            arguments=['-d'] # -d parameter means clear database at startup, start new mapping
        ),

        # ---------------------------------------------------------
        # Node E: Visualization (Visualization Interface)
        # ---------------------------------------------------------
        Node(
            package='rtabmap_viz',
            executable='rtabmap_viz',
            output='screen',
            parameters=rtabmap_parameters,
            remappings=rtabmap_remappings
        ),
    ])