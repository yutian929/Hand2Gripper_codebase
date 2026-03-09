#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
real single arm controller
Coordinate system explanation:
- flange_init: flange coordinate system when arm is at zero position (consistent with bimanual FK/IK)
- flange: current flange coordinate system
- gripper: gripper coordinate system (offset in flange X-axis direction by GRIPPER_OFFSET)
"""

import time
import json
import os
import numpy as np
from typing import Optional, List
from scipy.spatial.transform import Rotation as R
from bimanual import SingleArm


# default gripper calibration data
DEFAULT_GRIPPER_CALIBRATION = {
    -1.0: {"mean": 0.0}, -0.5: {"mean": 0.0}, 0.0: {"mean": 0.0},
    0.5: {"mean": 4.0}, 1.0: {"mean": 12.5}, 1.5: {"mean": 20.5},
    2.0: {"mean": 30.0}, 2.5: {"mean": 38.5}, 3.0: {"mean": 47.5},
    3.5: {"mean": 56.5}, 4.0: {"mean": 65.0}, 4.5: {"mean": 73.5},
    5.0: {"mean": 82.0}
}

# gripper offset (end-effector relative to flange, in flange X-axis direction)
GRIPPER_OFFSET = np.array([0.16, 0.0, 0.0])


class RealSingleArm:
    """
    real single arm controller
    
    all poses are represented in flange_init coordinate system (consistent with bimanual FK/IK)
    """
    
    def __init__(self, can_port: str = 'can0', arm_type: int = 0, 
                 calib_path: str = "gripper_calibration.json",
                 max_velocity: int = 200, max_acceleration: int = 500):
        """
        initialize single arm controller
        
        Args:
            can_port: CAN port name
            arm_type: arm type ID
            calib_path: gripper calibration file path
            max_velocity: maximum speed (50-500, smaller is slower)
            max_acceleration: maximum acceleration (100-2000, smaller is smoother)
        """
        self.config = {
            "can_port": can_port, 
            "type": arm_type,
            "max_velocity": max_velocity,
            "max_acceleration": max_acceleration
        }
        print(f"[RealSingleArm] Connecting on {can_port}, vel={max_velocity}, acc={max_acceleration}")
        self.arm = SingleArm(self.config)
        self.calib_points = self._load_calibration(calib_path)
    
    def _load_calibration(self, calib_path: str) -> List:
        """load gripper calibration data"""
        calib_data = DEFAULT_GRIPPER_CALIBRATION
        if os.path.exists(calib_path):
            try:
                with open(calib_path, 'r') as f:
                    calib_data = json.load(f)
            except Exception:
                pass
        points = [(v["mean"], float(k)) for k, v in calib_data.items()]
        points.sort(key=lambda x: x[0])
        return points
    
    def _real_to_set_width(self, real_mm: float) -> float:
        """real width (mm) -> set value"""
        if not self.calib_points:
            return real_mm
        pts = self.calib_points
        if real_mm <= pts[0][0]:
            return pts[0][1]
        if real_mm >= pts[-1][0]:
            return pts[-1][1]
        for i in range(len(pts) - 1):
            r1, s1 = pts[i]
            r2, s2 = pts[i + 1]
            if r1 <= real_mm <= r2 and abs(r2 - r1) > 1e-6:
                return s1 + (real_mm - r1) / (r2 - r1) * (s2 - s1)
        return pts[-1][1]
    
    def set_speed(self, max_velocity: int = 200, max_acceleration: int = 500):
        """
        set movement speed and acceleration
        
        Args:
            max_velocity: maximum speed (recommended 50-500)
            max_acceleration: maximum acceleration (recommended 100-2000)
        """
        self.arm.set_speed(max_velocity, max_acceleration)
    
    # ==================== coordinate transformation tools ====================
    
    @staticmethod
    def T_to_xyzrpy(T: np.ndarray) -> np.ndarray:
        """4x4 matrix -> [x, y, z, roll, pitch, yaw]"""
        return np.concatenate([T[:3, 3], R.from_matrix(T[:3, :3]).as_euler('xyz')])
    
    @staticmethod
    def xyzrpy_to_T(xyzrpy: np.ndarray) -> np.ndarray:
        """[x, y, z, roll, pitch, yaw] -> 4x4 matrix"""
        T = np.eye(4)
        T[:3, 3] = xyzrpy[:3]
        T[:3, :3] = R.from_euler('xyz', xyzrpy[3:]).as_matrix()
        return T
    
    @staticmethod
    def gripper_to_flange(T_ref_gripper: np.ndarray) -> np.ndarray:
        """
        gripper pose -> flange pose
        T_ref_flange = T_ref_gripper @ inv(T_flange_gripper)
        """
        T_flange_gripper = np.eye(4)
        T_flange_gripper[:3, 3] = GRIPPER_OFFSET
        return T_ref_gripper @ np.linalg.inv(T_flange_gripper)
    
    @staticmethod
    def flange_to_gripper(T_ref_flange: np.ndarray) -> np.ndarray:
        """
        flange pose -> gripper pose
        T_ref_gripper = T_ref_flange @ T_flange_gripper
        """
        T_flange_gripper = np.eye(4)
        T_flange_gripper[:3, 3] = GRIPPER_OFFSET
        return T_ref_flange @ T_flange_gripper
    
    # ==================== pose reading ====================
    
    def go_home(self):
        """go to zero position"""
        self.arm.go_home()
    
    def get_joint_positions(self) -> np.ndarray:
        """get current joint positions (radians), Shape: (7,)->(6,)"""
        return self.arm.get_joint_positions()[:6]
    
    def get_gripper_width(self, teacher=False) -> float:
        """
        get current gripper width (unit: meters)
        gripper_joint is motor position, need linear interpolation through calib_points to get actual width
        """
        gripper_joint = self.arm.get_joint_positions()[6]
        pts = self.calib_points
        # pts: List of (real_mm, set_val)
        # reverse interpolation: known set_val, find real_mm
        if not pts:
            return gripper_joint / 1000.0  # fallback
        pts = sorted(pts, key=lambda x: x[1])  # sort by set_val
        if gripper_joint <= pts[0][1]:
            width_mm = pts[0][0]
        elif gripper_joint >= pts[-1][1]:
            width_mm = pts[-1][0]
        else:
            for i in range(len(pts) - 1):
                r1, s1 = pts[i]
                r2, s2 = pts[i + 1]
                if s1 <= gripper_joint <= s2 and abs(s2 - s1) > 1e-6:
                    width_mm = r1 + (gripper_joint - s1) / (s2 - s1) * (r2 - r1)
                    break
            else:
                width_mm = pts[-1][0]
        if teacher:
            width_mm = width_mm*0.082/0.016  # special handling for teacher arm
        return width_mm / 1000.0  # return meters
    
    def get_flange_pose(self) -> np.ndarray:
        """
        get current flange pose (4x4)
        return T_flange_init_flange (current flange pose in flange_init coordinate system)
        """
        xyzrpy = self.arm.get_ee_pose_xyzrpy()
        return self.xyzrpy_to_T(xyzrpy)
    
    def get_gripper_pose(self) -> np.ndarray:
        """
        get current gripper pose (4x4)
        return T_flange_init_gripper (current gripper pose in flange_init coordinate system)
        """
        return self.flange_to_gripper(self.get_flange_pose())
    
    # ==================== pose control ====================

    def set_joint_positions(self, positions: np.ndarray):
        """
        set joint positions
        
        Args:
            positions: target joint positions (radians), Shape: (6,)
        """
        self.arm.set_joint_positions(positions)
    
    def set_flange_pose(self, T_flange_init_flange: np.ndarray):
        """
        set flange target pose
        
        Args:
            T_flange_init_flange: target pose of flange in flange_init coordinate system (4x4)
        """
        xyzrpy = self.T_to_xyzrpy(T_flange_init_flange)
        self.arm.set_ee_pose_xyzrpy(xyzrpy)
    
    def set_gripper_pose(self, T_flange_init_gripper: np.ndarray):
        """
        set gripper target pose (automatically convert to flange pose)
        
        Args:
            T_flange_init_gripper: target pose of gripper in flange_init coordinate system (4x4)
        """
        T_flange_init_flange = self.gripper_to_flange(T_flange_init_gripper)
        self.set_flange_pose(T_flange_init_flange)
    
    def set_gripper_width(self, width_m: float):
        """set gripper width (real width m)"""
        set_val = self._real_to_set_width(width_m * 1000)  # convert to mm
        self.arm.set_catch_pos(set_val)
    
    def move_to(self, T_flange_init_ee: np.ndarray, gripper_width_m: float = 0.030, 
                is_gripper_pose: bool = True):
        """
        move to target pose
        
        Args:
            T_flange_init_ee: target pose (4x4), in flange_init coordinate system
            gripper_width_mm: gripper width (mm)
            is_gripper_pose: True=input is gripper pose, False=input is flange pose
        """
        if is_gripper_pose:
            self.set_gripper_pose(T_flange_init_ee)
        else:
            self.set_flange_pose(T_flange_init_ee)
        self.set_gripper_width(gripper_width_m)
    
    def execute_trajectory(self, poses: List[np.ndarray], 
                           gripper_widths: Optional[List[float]] = None,
                           is_gripper_pose: bool = True, dt: float = 0.1):
        """
        execute trajectory
        
        Args:
            poses: pose list (4x4 matrices), in flange_init coordinate system
            gripper_widths: gripper width list (mm), None keeps 30mm
            is_gripper_pose: whether it is gripper pose
            dt: control period (seconds)
        """
        n = len(poses)
        if gripper_widths is None:
            gripper_widths = [30.0] * n
        
        print(f"[RealSingleArm] Executing {n} steps...")
        
        for i, (pose, width) in enumerate(zip(poses, gripper_widths)):
            t0 = time.time()
            if not np.any(np.isnan(pose)):
                try:
                    self.move_to(pose, width, is_gripper_pose)
                except Exception as e:
                    print(f"[RealSingleArm] Step {i} error: {e}")
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
        
        print("[RealSingleArm] Trajectory finished.")


if __name__ == "__main__":
    # initialize with slower speed
    arm = RealSingleArm(can_port='can3', max_velocity=100, max_acceleration=300)
    print("entering gravity compensation mode...")
    arm.arm.gravity_compensation()
    print("start reading gripper width (mm), press Ctrl+C to exit")
    try:
        while True:
            width = arm.get_gripper_width()
            print(f"Gripper width: {width:.4f} m", end='\r')
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nExit.")
    
    # follower arms are all 0.082, but master arm can0 is 0.016