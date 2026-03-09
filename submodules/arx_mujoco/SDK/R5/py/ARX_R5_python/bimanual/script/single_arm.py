from typing import List, Tuple, Union, Optional, Dict, Any
import numpy as np
import os
import sys
import arx_r5_python as arx


def quaternion_to_euler(quat: np.ndarray) -> Tuple[float, float, float]:
    """
    convert quaternion to Euler angles (roll, pitch, yaw)
    Parameters:
        quat: np.ndarray, array of length 4 [w, x, y, z]
    Returns:
        roll, pitch, yaw: Euler angles in radians
    """
    w, x, y, z = quat

    # calculate roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # calculate pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.pi / 2 * np.sign(sinp)  # use 90 degree limit
    else:
        pitch = np.arcsin(sinp)

    # calculate yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion.

    Parameters:
        roll: rotation angle around x-axis (radians)
        pitch: rotation angle around y-axis (radians)
        yaw: rotation angle around z-axis (radians)

    Returns:
        np.ndarray: quaternion array of length 4 [w, x, y, z]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])

class SingleArm:
    """
    Base class for a single robot arm.

    Args:
        config (Dict[str, sAny]): Configuration dictionary for the robot arm

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for the robot arm
        num_joints (int): Number of joints in the arm
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_joints = config.get(
            "num_joints", 7
        )  # Default to 7 joints if not specified
        self.dt = config.get(
            "dt", 0.05
        )  # Default to 0.05s time step if not specified. set up the control frequency

        current_dir = os.path.dirname(os.path.abspath(__file__))
        type = config.get("type",0)
        if(type == 0):
            urdf_path = os.path.join(current_dir,"X5liteaa0.urdf")
        else:
            urdf_path = os.path.join(current_dir,"R5_master.urdf")
        self.arm = arx.InterfacesPy(urdf_path,config.get("can_port", "can0"),type)
        
        # use speed parameters from config, default slower speed
        max_vel = config.get("max_velocity", 200)      # default 200, originally 500
        max_acc = config.get("max_acceleration", 500)  # default 500, originally 2000
        step = config.get("step", 10)
        self.arm.arx_x(max_vel, max_acc, step)

    def set_speed(self, max_velocity: int = 200, max_acceleration: int = 500, step: int = 10):
        """
        set robotic arm movement speed and acceleration
        
        Args:
            max_velocity: maximum speed (recommended 50-500, smaller is slower)
            max_acceleration: maximum acceleration (recommended 100-2000, smaller is smoother)
            step: step parameter
        """
        self.arm.arx_x(max_velocity, max_acceleration, step)
        print(f"[SingleArm] Speed set: vel={max_velocity}, acc={max_acceleration}, step={step}")

    def get_joint_names(self) -> List[str]:
        """
        Get the names of all joints in the arm.

        Returns:
            List[str]: List of joint names. Shape: (num_joints,)
        """
        return NotImplementedError

    def go_home(self) -> bool:
        """
        Move the robot arm to a pre-defined home pose.

        Returns:
            bool: True if the action was successful, False otherwise
        """
        self.arm.set_arm_status(1)
        return True

    def gravity_compensation(self) -> bool:
        self.arm.set_arm_status(3)
        return True

    def protect_mode(self) -> bool:
        self.arm.set_arm_status(2)
        return True

    def set_joint_positions(
        self,
        positions: Union[float, List[float], np.ndarray],  # Shape: (num_joints,)
        **kwargs
    ) -> bool:
        """
        Move the arm to the given joint position(s).

        Args:
            positions: Desired joint position(s). Shape: (6)
            **kwargs: Additional arguments

        """
        self.arm.set_joint_positions(positions)
        self.arm.set_arm_status(5)

    def set_ee_pose(
        self,
        pos: Optional[Union[List[float], np.ndarray]] = None,  # Shape: (3,)
        quat: Optional[Union[List[float], np.ndarray]] = None,  # Shape: (4,)
        **kwargs
    ) -> bool:
        """
        Move the end effector to the given pose.

        Args:
            pos: Desired position [x, y, z]. Shape: (3,)
            ori: Desired orientation (quaternion).
                 Shape: (4,) (w, x, y, z)
            **kwargs: Additional arguments

        """

        self.arm.set_ee_pose([pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]])
        self.arm.set_arm_status(4)

    def set_ee_pose_xyzrpy(
        self,
        xyzrpy: Optional[Union[List[float], np.ndarray]] = None,  # Shape: (6,)
        **kwargs
    ) -> bool:
        """
        Move the end effector to the given pose.

        Args:
            xyzrpy: Desired position [x, y, z, rol, pitch, yaw]. Shape: (6,)
            **kwargs: Additional arguments

        """
        quat = euler_to_quaternion(xyzrpy[3], xyzrpy[4], xyzrpy[5])

        self.arm.set_ee_pose(
            [xyzrpy[0], xyzrpy[1], xyzrpy[2], quat[0], quat[1], quat[2], quat[3]]
        )
        self.arm.set_arm_status(4)

    def set_catch_pos(self, pos: float):
        self.arm.set_catch(pos)

    def get_joint_positions(
        self, joint_names: Optional[Union[str, List[str]]] = None
    ) -> Union[float, List[float]]:
        """
        Get the current joint position(s) of the arm.

        Args:
            joint_names: Name(s) of the joint(s) to get positions for. Shape: (num_joints,) or single string. If None,
                            return positions for all joints.

        """
        return self.arm.get_joint_positions()

    def get_joint_velocities(
        self, joint_names: Optional[Union[str, List[str]]] = None
    ) -> Union[float, List[float]]:
        """
        Get the current joint velocity(ies) of the arm.

        Args:
            joint_names: Name(s) of the joint(s) to get velocities for. Shape: (num_joints,) or single string. If None,
                            return velocities for all joints.

        """
        return self.arm.get_joint_velocities()

    def get_joint_currents(
        self, joint_names: Optional[Union[str, List[str]]] = None
    ) -> Union[float, List[float]]:
        return self.arm.get_joint_currents()

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the current end effector pose of the arm.

        Returns:
            End effector pose as (position, quaternion)
            Shapes: position (3,), quaternion (4,) [w, x, y, z]
        """
        xyzwxyz = self.arm.get_ee_pose()

        return xyzwxyz

    def get_ee_pose_xyzrpy(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        xyzwxyz = self.arm.get_ee_pose()

        array = np.array([xyzwxyz[3], xyzwxyz[4], xyzwxyz[5], xyzwxyz[6]])

        roll, pitch, yaw = quaternion_to_euler(array)

        xyzrpy = np.array([xyzwxyz[0], xyzwxyz[1], xyzwxyz[2], roll, pitch, yaw])

        return xyzrpy

    def __del__(self):
        # or can directly release resources in destructor
        print("destroy SingleArm object")
        #self.cleanup()
