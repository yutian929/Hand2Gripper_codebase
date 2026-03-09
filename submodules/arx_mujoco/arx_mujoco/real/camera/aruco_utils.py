#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArUco code detection and visualization tool module
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


# =========================
# default parameters
# =========================
DEFAULT_MARKER_SIZE = 0.05  # 50mm -> meters
DEFAULT_MARKER_DICT = cv2.aruco.DICT_4X4_50


@dataclass
class ArucoResult:
    """ArUco detection result"""
    marker_id: int
    corners: np.ndarray          # (4, 2) image corners
    center: np.ndarray           # (2,) center pixel coordinates
    rvec: np.ndarray             # (3,) rotation vector
    tvec: np.ndarray             # (3,) translation vector (meters)
    T_cam_marker: np.ndarray     # (4, 4) marker->camera transformation matrix


def estimate_pose_from_corners(
    corners_4x2: np.ndarray,
    marker_size: float,
    K: np.ndarray,
    dist: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    estimate ArUco code pose from corners
    
    Args:
        corners_4x2: (4,2) image corners (top-left, top-right, bottom-right, bottom-left)
        marker_size: marker side length (meters)
        K: camera intrinsic matrix (3,3)
        dist: distortion coefficients (5,)
    
    Returns:
        rvec: (3,) rotation vector, return None on failure
        tvec: (3,) translation vector, return None on failure
    """
    L = float(marker_size)
    
    # ArUco corner order: top-left, top-right, bottom-right, bottom-left
    objp = np.array([
        [-L/2,  L/2, 0.0],
        [ L/2,  L/2, 0.0],
        [ L/2, -L/2, 0.0],
        [-L/2, -L/2, 0.0],
    ], dtype=np.float64)
    
    imgp = np.asarray(corners_4x2, dtype=np.float64).reshape(4, 2)
    
    flag = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=flag)
    
    if not ok:
        return None, None
    return rvec.reshape(3), tvec.reshape(3)


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    convert rvec/tvec to 4x4 transformation matrix T_cam_marker
    """
    from scipy.spatial.transform import Rotation as R
    
    T = np.eye(4, dtype=np.float64)
    Rm = R.from_rotvec(np.asarray(rvec, dtype=np.float64).reshape(3)).as_matrix()
    T[:3, :3] = Rm
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def detect_aruco(
    img: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    marker_size: float = DEFAULT_MARKER_SIZE,
    marker_dict: int = DEFAULT_MARKER_DICT,
    target_id: Optional[int] = None
) -> List[ArucoResult]:
    """
    detect ArUco codes in image and estimate pose
    
    Args:
        img: BGR image
        K: camera intrinsic matrix (3,3)
        dist: distortion coefficients (5,)
        marker_size: marker side length (meters)
        marker_dict: ArUco dictionary type
        target_id: specify to return only marker with certain ID, None means return all
    
    Returns:
        ArucoResult list
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    dictionary = cv2.aruco.getPredefinedDictionary(marker_dict)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    
    corners, ids, _ = detector.detectMarkers(gray)
    
    results = []
    
    if ids is None or len(ids) == 0:
        return results
    
    for i, marker_id in enumerate(ids.flatten()):
        if target_id is not None and marker_id != target_id:
            continue
        
        corner = corners[i].reshape(4, 2)
        center = corner.mean(axis=0)
        
        rvec, tvec = estimate_pose_from_corners(corner, marker_size, K, dist)
        
        if rvec is None:
            continue
        
        T_cam_marker = rvec_tvec_to_T(rvec, tvec)
        
        results.append(ArucoResult(
            marker_id=int(marker_id),
            corners=corner,
            center=center,
            rvec=rvec,
            tvec=tvec,
            T_cam_marker=T_cam_marker
        ))
    
    return results


def draw_aruco(
    img: np.ndarray,
    results: List[ArucoResult],
    K: np.ndarray,
    dist: np.ndarray,
    axis_length: float = 0.04,
    draw_corners: bool = True,
    draw_axes: bool = True,
    draw_info: bool = True
) -> np.ndarray:
    """
    visualize ArUco detection results on image
    
    Args:
        img: BGR image
        results: result list returned by detect_aruco
        K: camera intrinsic matrix
        dist: distortion coefficients
        axis_length: axis length (meters)
        draw_corners: whether to draw corners
        draw_axes: whether to draw axes
        draw_info: whether to draw ID and position info
    
    Returns:
        annotated image
    """
    vis = img.copy()
    
    for res in results:
        # draw corner box
        if draw_corners:
            pts = res.corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            
            # mark four corners
            for j, pt in enumerate(res.corners):
                color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)][j]
                cv2.circle(vis, tuple(pt.astype(int)), 5, color, -1)
        
        # draw coordinate axes
        if draw_axes:
            cv2.drawFrameAxes(vis, K, dist, res.rvec, res.tvec, axis_length)
        
        # draw ID and position information
        if draw_info:
            cx, cy = res.center.astype(int)
            cv2.circle(vis, (cx, cy), 5, (255, 0, 255), -1)
            
            # ID
            cv2.putText(vis, f"ID:{res.marker_id}", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # position information
            tx, ty, tz = res.tvec
            info_text = f"[{tx:.3f}, {ty:.3f}, {tz:.3f}]m"
            cv2.putText(vis, info_text, (cx + 10, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return vis


def get_single_aruco(
    img: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    marker_id: int = 0,
    marker_size: float = DEFAULT_MARKER_SIZE,
    marker_dict: int = DEFAULT_MARKER_DICT
) -> Optional[ArucoResult]:
    """
    convenience function: detect and return single ArUco code with specified ID
    
    Args:
        img: BGR image
        K: camera intrinsic matrix
        dist: distortion coefficients
        marker_id: target marker ID
        marker_size: marker side length (meters)
        marker_dict: ArUco dictionary type
    
    Returns:
        ArucoResult or None
    """
    results = detect_aruco(img, K, dist, marker_size, marker_dict, target_id=marker_id)
    return results[0] if results else None


# =========================
# test script
# =========================
if __name__ == "__main__":
    import os
    
    # =========================
    # set parameters directly
    # =========================
    IMAGE_PATH = "./submodules/arx_mujoco/assert/calib_screenshot_raw_20251228_091850.png"  # input image path
    MARKER_SIZE = 0.05             # Marker side length (meters)
    TARGET_MARKER_ID = 0           # specify marker ID to detect, None means detect all
    OUTPUT_PATH = None             # output image path, None means display directly
    
    # camera intrinsics (use actual calibration values, not image center approximation)
    FX = 606.0810546875
    FY = 605.1178588867188
    CX = 327.5788879394531
    CY = 245.88775634765625
    
    # =========================
    # construct camera intrinsics
    # =========================
    K = np.array([
        [FX, 0, CX],
        [0, FY, CY],
        [0, 0, 1]
    ], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    
    # =========================
    # read image
    # =========================
    if not os.path.exists(IMAGE_PATH):
        print(f"[ERROR] image does not exist: {IMAGE_PATH}")
        exit(1)
    
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"[ERROR] cannot read image: {IMAGE_PATH}")
        exit(1)
    
    print(f"[INFO] read image: {IMAGE_PATH}, size: {img.shape}")
    print(f"[INFO] using camera intrinsics: fx={FX}, fy={FY}, cx={CX}, cy={CY}")
    
    # Note: do not override K's cx/cy, should use actual calibrated intrinsic values
    
    # =========================
    # detect ArUco
    # =========================
    results = detect_aruco(img, K, dist, MARKER_SIZE, target_id=TARGET_MARKER_ID)
    
    if not results:
        print("[WARN] no ArUco codes detected")
    else:
        print(f"[INFO] detected {len(results)} ArUco codes:")
        for res in results:
            print(f"  - ID={res.marker_id}")
            print(f"    center(px): [{res.center[0]:.1f}, {res.center[1]:.1f}]")
            print(f"    tvec(m):    [{res.tvec[0]:.4f}, {res.tvec[1]:.4f}, {res.tvec[2]:.4f}]")
            print(f"    rvec(rad):  [{res.rvec[0]:.4f}, {res.rvec[1]:.4f}, {res.rvec[2]:.4f}]")
            print(f"    T_cam_marker:\n{res.T_cam_marker}")
    
    # =========================
    # visualize
    # =========================
    vis = draw_aruco(img, results, K, dist)
    
    # save or display
    if OUTPUT_PATH:
        cv2.imwrite(OUTPUT_PATH, vis)
        print(f"[OK] save result to: {OUTPUT_PATH}")
    else:
        cv2.imshow("ArUco Detection", vis)
        print("[INFO] press any key to exit")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
