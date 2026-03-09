#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera parameter loading tool module
"""

import json
import numpy as np
from typing import Tuple, Optional


def load_camera_intrinsics(
    json_path: str,
    camera: str = "left"
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    load camera intrinsics from JSON file
    
    Args:
        json_path: JSON file path
        camera: camera selection ("left" or "right")
    
    Returns:
        raw_dict: raw dictionary data
        K: camera intrinsic matrix (3,3) numpy array
        dist: distortion coefficients (5,) numpy array
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cam_data = data.get(camera, data)
    
    # construct intrinsic matrix
    fx = cam_data["fx"]
    fy = cam_data["fy"]
    cx = cam_data["cx"]
    cy = cam_data["cy"]
    v_fov = cam_data.get("v_fov", {})
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # distortion coefficients (take first 5, convert to numpy array)
    disto = cam_data.get("disto", [0.0] * 5)
    dist = np.array(disto[:5], dtype=np.float64)
    
    return cam_data, K, dist, v_fov


def get_camera_intrinsics_from_dict(
    cam_data: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    construct camera intrinsic matrix and distortion coefficients from dictionary
    
    Args:
        cam_data: dictionary containing fx, fy, cx, cy, disto
    
    Returns:
        K: camera intrinsic matrix (3,3) numpy array
        dist: distortion coefficients (5,) numpy array
    """
    fx = cam_data["fx"]
    fy = cam_data["fy"]
    cx = cam_data["cx"]
    cy = cam_data["cy"]
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    disto = cam_data.get("disto", [0.0] * 5)
    dist = np.array(disto[:5], dtype=np.float64)
    
    return K, dist


def load_eye_to_hand_matrix(json_path: str) -> np.ndarray:
    """
    load hand-eye calibration matrix (camera link pose in robotic arm base coordinate system)
    
    Args:
        json_path: JSON file path
    
    Returns:
        T_base_camlink: (4,4) numpy array
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data["Mat_base_T_camera_link"], dtype=np.float64)


def T_optical_to_link() -> np.ndarray:
    """
    transformation matrix from optical coordinate system to link coordinate system
    
    Optical (OpenCV): X-Right, Y-Down, Z-Forward
    Link (ROS):       X-Forward, Y-Left, Z-Up
    
    Returns:
        T_link_optical: (4,4) transformation matrix, such that P_link = T @ P_optical
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([
        [ 0,  0,  1],  # link X = optical Z
        [-1,  0,  0],  # link Y = -optical X
        [ 0, -1,  0],  # link Z = -optical Y
    ], dtype=np.float64)
    return T


def T_link_to_optical() -> np.ndarray:
    """
    transformation matrix from link coordinate system to optical coordinate system
    
    Returns:
        T_optical_link: (4,4) transformation matrix
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([
        [ 0, -1,  0],  # optical X = -link Y
        [ 0,  0, -1],  # optical Y = -link Z
        [ 1,  0,  0],  # optical Z = link X
    ], dtype=np.float64)
    return T
