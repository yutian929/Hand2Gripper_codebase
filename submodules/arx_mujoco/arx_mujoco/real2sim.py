#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real2Sim: Generate simulated RGB and Mask images from end-effector pose in real camera coordinate system
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import bimanual
from .sim.mujoco_single_arm import MujocoSingleArm
from .real.camera.camera_utils import load_eye_to_hand_matrix, load_camera_intrinsics

@dataclass
class Real2SimResult:
    """single frame rendering result"""
    rgb: np.ndarray              # (H, W, 3) uint8
    mask: np.ndarray             # (H, W) uint8, robotic arm area is 255
    joint_angles: np.ndarray     # (6,) joint angles
    T_world_flange: np.ndarray   # (4,4) flange world pose
    T_world_gripper: np.ndarray  # (4,4) gripper world pose
    ik_success: bool             # whether IK succeeded


class Real2Sim:
    """
    Real2Sim renderer
    
    Used to generate simulated RGB and Mask images from end-effector pose in camera link coordinate system.
    Supports:
    - Set camera pose (relative to initial flange)
    - Input end-effector pose (camera link coordinate system)
    - Automatic IK solving
    - Render RGB+Mask
    """
    
    # gripper offset relative to flange (in flange coordinate system, X-axis direction)
    GRIPPER_OFFSET = np.array([0.16, 0.0, 0.0])
    
    def __init__(
        self,
        xml_path: str,
        T_flange_init_camlink: np.ndarray,
        width: int = 640,
        height: int = 480,
        fov: float = 43.27,
        gripper_offset: Optional[np.ndarray] = None,
        verbose: bool = False
    ):
        """
        initialize Real2Sim renderer
        
        Args:
            xml_path: MuJoCo XML model path
            T_flange_init_camlink: 4x4 transformation matrix of camera link in initial flange coordinate system
                                   (from hand-eye calibration results)
            width: rendering image width
            height: rendering image height
            fov: camera vertical FOV (degrees)
            gripper_offset: gripper offset relative to flange, default [0.15, 0, 0]
            verbose: whether to print debug information
        """
        self.xml_path = xml_path
        self.T_flange_init_camlink = np.array(T_flange_init_camlink, dtype=np.float64)
        self.width = width
        self.height = height
        self.fov = fov
        self.verbose = verbose
        
        if gripper_offset is not None:
            self.GRIPPER_OFFSET = np.array(gripper_offset, dtype=np.float64)
        
        # initialize MuJoCo
        self.arm = MujocoSingleArm(xml_path, verbose=verbose)
        self.arm.init_renderer(width, height)
        
        # calculate camera pose in world coordinate system
        self.T_world_flange_init = self.arm.get_body_pose("link6")
        self.T_world_camlink = self.T_world_flange_init @ self.T_flange_init_camlink
        
        # set camera
        self.arm.set_camera_pose("render_camera", self.T_world_camlink)
        self.arm.set_camera_fov("render_camera", self.fov)
        
        # cache last successful joint angles (for fallback when IK fails)
        self._last_valid_joints = np.zeros(6)
        
        if self.verbose:
            print(f"[Real2Sim] Initialized")
            print(f"  T_world_camlink:\n{self.T_world_camlink}")
    
    # ==================== coordinate transformation tools ====================
    
    @staticmethod
    def T_to_xyzrpy(T: np.ndarray) -> np.ndarray:
        """4x4 matrix -> [x, y, z, roll, pitch, yaw]"""
        pos = T[:3, 3]
        rpy = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=False)
        return np.concatenate([pos, rpy])
    
    @staticmethod
    def xyzrpy_to_T(xyzrpy: np.ndarray) -> np.ndarray:
        """ [x, y, z, roll, pitch, yaw] -> 4x4 matrix"""
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = xyzrpy[:3]
        T[:3, :3] = R.from_euler('xyz', xyzrpy[3:], degrees=False).as_matrix()
        return T
    
    def gripper_to_flange(self, T_base_gripper: np.ndarray) -> np.ndarray:
        """convert gripper pose to flange pose"""
        T_flange_gripper = np.eye(4, dtype=np.float64)
        T_flange_gripper[:3, 3] = self.GRIPPER_OFFSET
        return T_base_gripper @ np.linalg.inv(T_flange_gripper)
    
    def flange_to_gripper(self, T_base_flange: np.ndarray) -> np.ndarray:
        """convert flange pose to gripper pose"""
        T_flange_gripper = np.eye(4, dtype=np.float64)
        T_flange_gripper[:3, 3] = self.GRIPPER_OFFSET
        return T_base_flange @ T_flange_gripper
    
    # ==================== core rendering methods ====================
    
    def render(
        self,
        T_camlink_ee: np.ndarray,
        gripper_width: float = 0.02,
        is_gripper_pose: bool = True
    ) -> Real2SimResult:
        """
        render single frame
        
        Args:
            T_camlink_ee: 4x4 transformation matrix of end-effector in camera link coordinate system
                         if is_gripper_pose=True, this is gripper pose
                         if is_gripper_pose=False, this is flange pose
            gripper_width: gripper opening width [0, 0.044]
            is_gripper_pose: whether input is gripper pose (needs conversion to flange)
        
        Returns:
            Real2SimResult contains rgb, mask, joint angles and other information
        """
        # check if input contains nan
        if np.isnan(T_camlink_ee).any():
            if self.verbose:
                print("[Real2Sim] Warning: T_camlink_ee contains nan, skipping frame.")
            H, W = self.height, self.width
            return Real2SimResult(
                rgb=np.zeros((H, W, 3), dtype=np.uint8),
                mask=np.zeros((H, W), dtype=np.uint8),
                joint_angles=np.zeros(6),
                T_world_flange=np.eye(4),
                T_world_gripper=np.eye(4),
                ik_success=False
            )
        
        # convert to world coordinate system
        T_world_ee = self.T_world_camlink @ T_camlink_ee
        
        # if input is gripper pose, convert to flange pose
        if is_gripper_pose:
            T_world_flange_target = self.gripper_to_flange(T_world_ee)
        else:
            T_world_flange_target = T_world_ee
        
        # convert to initial flange coordinate system (needed for IK solving)
        T_flange_init_inv = np.linalg.inv(self.T_world_flange_init)
        T_flange_init_flange_target = T_flange_init_inv @ T_world_flange_target
        
        # IK solving
        xyzrpy_target = self.T_to_xyzrpy(T_flange_init_flange_target)
        
        try:
            joint_angles = bimanual.inverse_kinematics(xyzrpy_target)
            if joint_angles is None:
                ik_success = False
                joint_angles = self._last_valid_joints.copy()
            else:
                ik_success = True
                self._last_valid_joints = joint_angles.copy()
        except Exception as e:
            if self.verbose:
                print(f"[WARN] IK failed: {e}")
            ik_success = False
            joint_angles = self._last_valid_joints.copy()
        
        # set joint angles
        full_joints = np.zeros(8)
        full_joints[:6] = joint_angles
        full_joints[6:8] = np.clip(gripper_width, 0.0, 0.044)
        
        self.arm.set_joint_angles(full_joints)
        self.arm.forward()
        
        # get actual pose
        T_world_flange = self.arm.get_body_pose("link6")
        T_world_gripper = self.flange_to_gripper(T_world_flange)
        
        # set target marker for debugging
        self.arm.set_target_marker(T_world_ee)
        self.arm.forward()
        
        # render
        rgb, mask = self.arm.render("render_camera", self.width, self.height, with_mask=True)
        
        return Real2SimResult(
            rgb=rgb,
            mask=mask,
            joint_angles=joint_angles,
            T_world_flange=T_world_flange,
            T_world_gripper=T_world_gripper,
            ik_success=ik_success
        )
    
    def render_batch(
        self,
        T_camlink_ee_list: List[np.ndarray],
        gripper_widths: Optional[List[float]] = None,
        is_gripper_pose: bool = True,
        show_progress: bool = True
    ) -> List[Real2SimResult]:
        """
        batch render
        
        Args:
            T_camlink_ee_list: end-effector pose list, each element is 4x4 transformation matrix
            gripper_widths: gripper width list, None means all use 0.02
            is_gripper_pose: whether input is gripper pose
            show_progress: whether to show progress
        
        Returns:
            Real2SimResult list
        """
        n = len(T_camlink_ee_list)
        
        if gripper_widths is None:
            gripper_widths = [0.02] * n
        
        assert len(gripper_widths) == n, "gripper_widths length must be same as pose list"
        
        results = []
        ik_fail_count = 0
        
        for i, (T_ee, gw) in enumerate(zip(T_camlink_ee_list, gripper_widths)):
            result = self.render(T_ee, gw, is_gripper_pose)
            results.append(result)
            
            if not result.ik_success:
                ik_fail_count += 1
            
            if show_progress and (i + 1) % 10 == 0:
                print(f"[Real2Sim] Rendered {i+1}/{n} frames, IK fails: {ik_fail_count}")
        
        if show_progress:
            print(f"[Real2Sim] Done. Total: {n}, IK fails: {ik_fail_count}")
        
        return results
    
    # ==================== auxiliary methods ====================
    
    def get_arrays(self, results: List[Real2SimResult]) -> Tuple[np.ndarray, np.ndarray]:
        """
        extract RGB and Mask arrays from render results
        
        Returns:
            rgb_array: (N, H, W, 3) uint8
            mask_array: (N, H, W) uint8
        """
        rgb_list = [r.rgb for r in results]
        mask_list = [r.mask for r in results]
        return np.stack(rgb_list), np.stack(mask_list)
    
    def start_viewer(self):
        """start MuJoCo Viewer"""
        self.arm.start_viewer()
    
    def spin(self, rate_hz: float = 60.0):
        """continuously update Viewer until closed"""
        self.arm.spin(rate_hz)
    
    def close(self):
        """close renderer"""
        self.arm.stop_viewer()


# ==================== test ====================

if __name__ == "__main__":
    import cv2

    # parameters
    XML_PATH = "SDK/R5a/meshes/R5a_R5master.xml"
    EYE_TO_HAND_PATH = "real/camera/eye_to_hand_result_left_latest.json"
    CAMERA_INTRINSICS_PATH = "real/camera/camera_intrinsics_d435i.json"
    
    # load calibration data
    T_flange_init_camlink = load_eye_to_hand_matrix(EYE_TO_HAND_PATH)
    _, K, dist, v_fov = load_camera_intrinsics(CAMERA_INTRINSICS_PATH)
    
    print(f"[INFO] T_flange_init_camlink:\n{T_flange_init_camlink}")
    print(f"[INFO] v_fov: {v_fov}")
    
        # create renderer
    r2s = Real2Sim(
        xml_path=XML_PATH,
        T_flange_init_camlink=T_flange_init_camlink,
        width=640,
        height=480,
        fov=v_fov,
        verbose=True
    )
    
        # generate test sequence: end-effector moves linearly along camera link X-axis (forward)
    # pose remains unchanged (identity matrix)
    T_camlink_ee_list = []
    start_x = 0.5   # starting distance 0.2m
    end_x = 0.8     # ending distance 0.5m
    num_frames = 30
    
    for i in range(num_frames):
        T = np.eye(4, dtype=np.float64)
        # position: linear movement along X-axis (forward)
        x = start_x + (end_x - start_x) * i / (num_frames - 1)
        T[:3, 3] = [x, 0.0, 0.0]
        # pose: remain unchanged (identity matrix)
        T_camlink_ee_list.append(T)
    
    print(f"[INFO] generated {num_frames} frame test sequence, end-effector moves from x={start_x}m to x={end_x}m")
    
        # batch render
    results = r2s.render_batch(T_camlink_ee_list, show_progress=True)
    
    # display results
    for i, res in enumerate(results):
        vis = cv2.cvtColor(res.rgb, cv2.COLOR_RGB2BGR)
        
        # overlay mask
        overlay = vis.astype(np.float32)
        overlay[res.mask > 0] = overlay[res.mask > 0] * 0.5 + np.array([0, 255, 0], dtype=np.float32) * 0.5
        
        # display IK status
        status = "IK OK" if res.ik_success else "IK FAIL"
        cv2.putText(overlay, f"Frame {i}: {status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Real2Sim", overlay.astype(np.uint8))
        key = cv2.waitKey(100)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    # start viewer
    r2s.spin()
