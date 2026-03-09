import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    data_dir_arg = DeclareLaunchArgument(
        'data_dir',
        default_value='./data/raw/epic/0', 
        description='Path to the directory containing video_L.mp4, depth.npy and json'
    )

    # 1. RGBD Player Node
    player_node = Node(
        package='rgbd_playback',
        executable='player',
        name='rgbd_player_node',
        output='screen',
        parameters=[{
            'data_dir': LaunchConfiguration('data_dir'),
            'frequency': 30.0,
            'loop': True
        }],
        remappings=[
            ('camera/color/image_raw', '/camera/camera/color/image_raw'),
            ('camera/aligned_depth_to_color/image_raw', '/camera/camera/aligned_depth_to_color/image_raw'),
            ('camera/color/camera_info', '/camera/camera/color/camera_info')
        ]
    )

    # 2. Static TF Publishing (Key Modification)
    # Goal: Convert camera_link (X forward, Y left, Z up) to camera_color_optical_frame (Z forward, X right, Y down)
    # Parameter order here: x y z qx qy qz qw
    # Quaternion [-0.5, 0.5, -0.5, 0.5] corresponds to: Roll=-90, Pitch=0, Yaw=-90 rotation
    # Transformation result: Link X-axis becomes Optical Z-axis
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_link_broadcaster',
        arguments = ['0', '0', '0', '-0.5', '0.5', '-0.5', '0.5', 'camera_link', 'camera_color_optical_frame']
    )

    return LaunchDescription([
        data_dir_arg,
        player_node,
        static_tf
    ])