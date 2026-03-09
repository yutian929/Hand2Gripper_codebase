#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MuJoCo single arm visualization script
Use MuJoCo 3.4.0 to read XML file, set joint angles and visualize
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation as R


class MujocoSingleArm:
    """
    MuJoCo single arm model wrapper class
    supports loading model, setting joint angles, forward kinematics calculation, camera rendering and visualization
    """
    
    def __init__(self, xml_path: str, verbose: bool = True):
        self.xml_path = xml_path
        self.viewer = None
        self.renderer = None
        self.render_width = 640
        self.render_height = 480
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.data.qpos[:] = 0
        self.forward()

        if verbose:
            self._print_model_info()
            self.print_body_positions()
    
    def _print_model_info(self):
        print(f"[INFO] Model: {self.xml_path}")
        print(f"  nq={self.model.nq}, njnt={self.model.njnt}, nbody={self.model.nbody}")
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            jtype = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}[self.model.jnt_type[i]]
            print(f"    Joint {i}: {name} ({jtype})")
    
    # ==================== joint operations ====================
    
    def set_joint_angles(self, angles: np.ndarray):
        angles = np.asarray(angles, dtype=np.float64)
        assert len(angles) == self.model.nq, f"Expected {self.model.nq} angles, got {len(angles)}"
        self.data.qpos[:] = angles
    
    def get_joint_angles(self) -> np.ndarray:
        return self.data.qpos.copy()
    
    def forward(self):
        mujoco.mj_forward(self.model, self.data)
    
    # ==================== body pose query ====================
    
    def get_body_pose(self, body_name: str) -> np.ndarray:
        """get body's 4x4 homogeneous transformation matrix"""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
        T[:3, 3] = self.data.xpos[body_id]
        return T
    
    def print_body_positions(self):
        print("\n[INFO] Body positions:")
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            pos = self.data.xpos[i]
            print(f"  {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    
    # ==================== Target Marker ====================
    
    def set_target_marker(self, pos: np.ndarray, marker_name: str = "target_marker"):
        """
        set target marker position (for debug visualization)
        
        Args:
            pos: position [x, y, z] or 4x4 transformation matrix
            marker_name: mocap body name (needs to be predefined as mocap="true" in XML)
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, marker_name)
        if body_id == -1:
            print(f"[WARN] Marker body '{marker_name}' not found in model.")
            return
        
        # get mocap id
        mocap_id = self.model.body_mocapid[body_id]
        if mocap_id == -1:
            print(f"[WARN] Body '{marker_name}' is not a mocap body.")
            return
        
        # support multiple input formats
        if isinstance(pos, np.ndarray):
            if pos.shape == (4, 4):
                # 4x4 transformation matrix
                self.data.mocap_pos[mocap_id] = pos[:3, 3]
                # optional: also set orientation
                rot_mat = pos[:3, :3]
                quat_scipy = R.from_matrix(rot_mat).as_quat()  # [x, y, z, w]
                self.data.mocap_quat[mocap_id] = [quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]
            elif pos.shape == (3,):
                # [x, y, z]
                self.data.mocap_pos[mocap_id] = pos
            elif pos.shape == (6,):
                # [x, y, z, roll, pitch, yaw]
                self.data.mocap_pos[mocap_id] = pos[:3]
                quat_scipy = R.from_euler('xyz', pos[3:], degrees=False).as_quat()
                self.data.mocap_quat[mocap_id] = [quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]
            else:
                print(f"[WARN] Invalid pos format: {pos.shape}")
        else:
            # assume it's list or tuple
            pos = np.array(pos)
            self.data.mocap_pos[mocap_id] = pos[:3]
    
    # ==================== camera rendering ====================
    
    def init_renderer(self, width: int = 640, height: int = 480):
        self.render_width = width
        self.render_height = height
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
    
    def set_camera_pose(self, cam_name: str, T_world_camlink: np.ndarray):
        """
        set camera pose (Link coordinate system: X forward, Y left, Z up)
        
        Args:
            cam_name: camera name (needs to be predefined in XML)
            T_world_camlink: camera 4x4 transformation matrix in world coordinate system
                            or (6,) [x, y, z, roll, pitch, yaw]
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id == -1:
            print(f"[WARN] Camera '{cam_name}' not found in model.")
            return
        
        # support two input formats
        if T_world_camlink.shape == (6,):
            # [x, y, z, roll, pitch, yaw] format
            pos = T_world_camlink[:3]
            r_input = R.from_euler('xyz', T_world_camlink[3:], degrees=False)
            mat_input = r_input.as_matrix()
        elif T_world_camlink.shape == (4, 4):
            # 4x4 matrix format
            pos = T_world_camlink[:3, 3]
            mat_input = T_world_camlink[:3, :3]
        else:
            print(f"[WARN] Invalid pose format: {T_world_camlink.shape}")
            return
        
        # set position
        self.model.cam_pos[cam_id] = pos
        
        # Link -> MuJoCo Camera conversion
        # refer to conversion matrix in code (from Link to MuJoCo camera coordinate system)
        # MuJoCo camera: -Z forward (looking direction), Y up, X right
        # Link: X forward, Y left, Z up
        # this matrix converts points from Link coordinate system to MuJoCo camera coordinate system
        mat_link_to_mjcam = np.array([
            [ 0,  0, -1],  # mjcam X = -link Z? wrong, re-derive
            [-1,  0,  0],  # mjcam Y = -link X? 
            [ 0,  1,  0],  # mjcam Z = link Y?
        ], dtype=np.float64)
        
        # apply conversion: R_world_mjcam = R_world_link @ R_link_mjcam
        mat_final = mat_input @ mat_link_to_mjcam
        
        r_final = R.from_matrix(mat_final)
        quat_scipy = r_final.as_quat()  # [x, y, z, w]
        
        # convert to MuJoCo order [w, x, y, z]
        self.model.cam_quat[cam_id] = [quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]

    def set_camera_fov(self, cam_name: str, fov_deg: float):
        """set camera vertical FOV (degrees)"""
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id != -1:
            self.model.cam_fovy[cam_id] = fov_deg

    def render(self, cam_name: str, width: int = 640, height: int = 480,
               with_mask: bool = False, exclude_geom_names: list = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        render RGB image, optionally return segmentation mask at the same time
        
        Args:
            cam_name: camera name
            width: image width
            height: image height
            with_mask: whether to return mask at the same time
            exclude_geom_names: list of geom names to exclude (not included in mask)
        
        Returns:
            if with_mask=False: (rgb, None)
            if with_mask=True: (rgb, mask)
            - rgb: RGB image (H, W, 3) uint8
            - mask: binary mask (H, W) uint8, robotic arm area is 255
        """
        if self.renderer is None or self.render_width != width or self.render_height != height:
            self.init_renderer(width, height)
        
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        
        # render RGB
        if cam_id != -1:
            self.renderer.update_scene(self.data, camera=cam_name)
        else:
            self.renderer.update_scene(self.data)
        rgb = self.renderer.render().copy()
        
        # if mask not needed, return directly
        if not with_mask:
            return rgb, None
        
        # render segmentation mask
        self.renderer.enable_segmentation_rendering()
        if cam_id != -1:
            self.renderer.update_scene(self.data, camera=cam_name)
        else:
            self.renderer.update_scene(self.data)
        seg = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        
        # process segmentation results
        # seg[:,:,0] is geom id
        geom_ids = seg[:, :, 0].astype(np.int32)
        
        # default excluded geoms
        if exclude_geom_names is None:
            exclude_geom_names = ['floor', 'stand']
        
        # get geom ids to exclude
        exclude_ids = set([-1])  # background
        for name in exclude_geom_names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid != -1:
                exclude_ids.add(gid)
        
        # create mask: all non-excluded geoms are considered robotic arm
        mask = np.zeros((height, width), dtype=np.uint8)
        for gid in np.unique(geom_ids):
            if gid not in exclude_ids and gid >= 0:
                mask[geom_ids == gid] = 255
        
        return rgb, mask
    
    # ==================== Viewer ====================
    
    def start_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
    
    def stop_viewer(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def is_viewer_running(self) -> bool:
        return self.viewer is not None and self.viewer.is_running()
    
    def update(self):
        self.forward()
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
    
    def spin(self, rate_hz: float = 60.0):
        if self.viewer is None:
            self.start_viewer()
        
        dt = 1.0 / rate_hz
        while self.is_viewer_running():
            self.update()
            time.sleep(dt)
        self.viewer = None
    
    def __del__(self):
        if self.viewer is not None:
            try: self.viewer.close()
            except: pass
        if self.renderer is not None:
            try: self.renderer.close()
            except: pass



if __name__ == "__main__":
    import cv2
    
    xml_path = "./submodules/arx_mujoco/SDK/R5a/meshes/R5a_R5master.xml"
    
    arm = MujocoSingleArm(xml_path)
    arm.set_joint_angles([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.02])
    arm.forward()
    arm.print_body_positions()
    
        # test camera rendering
    cam_pose = np.array([0.5, 0.0, 0.2, 0.0, 0.0, np.pi])
    arm.set_camera_pose("render_camera", cam_pose)
    arm.set_camera_fov("render_camera", 60.0)
    arm.forward()
    
        # render RGB and mask
    rgb, mask = arm.render("render_camera", 640, 480, with_mask=True)
    
    print(f"[INFO] RGB shape: {rgb.shape}, Mask shape: {mask.shape}")
    print(f"[INFO] Mask unique values: {np.unique(mask)}")
    
    # display
    cv2.imshow("RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imshow("Mask", mask)
    
        # overlay display
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0], dtype=np.float32) * 0.5
    cv2.imshow("Overlay", overlay.astype(np.uint8))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # test top view
    cam_pose_top = np.array([0.1, 0.0, 0.8, 0.0, np.pi/2, 0.0])
    arm.set_camera_pose("render_camera", cam_pose_top)
    arm.forward()
    
    rgb_top, mask_top = arm.render("render_camera", 640, 480, with_mask=True)
    cv2.imshow("Top RGB", cv2.cvtColor(rgb_top, cv2.COLOR_RGB2BGR))
    cv2.imshow("Top Mask", mask_top)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    arm.spin()