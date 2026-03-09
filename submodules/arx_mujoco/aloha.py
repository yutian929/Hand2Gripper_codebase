#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALOHA-style master-slave teaching controller
4 robotic arms: master1, master2 (gravity compensation state), slave1, slave2 (replicate master arm pose)

Usage:
    python aloha.py

CAN port configuration:
    - master1: can0
    - master2: can1  
    - slave1: can2
    - slave2: can3
"""

import time
import signal
import sys
import numpy as np
from typing import Optional, Dict, Any
from arx_mujoco.real.real_single_arm import RealSingleArm  # use official low-level interface

class AlohaTeaching:
    """
    ALOHA master-slave teaching controller
    
    Master arm: gravity compensation state, manual drag teaching
    Follower arm: real-time replication of master arm pose
    """
    
    def __init__(self, 
                 master1_can: str = 'can0',
                 master2_can: str = 'can1',
                 follower1_can: str = 'can2',
                 follower2_can: str = 'can3',
                 arm_type: int = 0,
                 control_freq: float = 50.0):
        """
        Initialize ALOHA teaching system
        
        Args:
            master1_can: CAN port for master arm 1
            master2_can: CAN port for master arm 2
            follower1_can: CAN port for follower arm 1
            follower2_can: CAN port for follower arm 2
            arm_type: arm type (0: X5liteaa0, 1: R5_master)
            control_freq: control frequency (Hz)
        """
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self.running = False
        
        print("=" * 60)
        print("ALOHA master-slave teaching system initialization")
        print("=" * 60)
        
                # initialize 4 robotic arms
        print(f"\n[1/4] Initializing master arm 1 ({master1_can})...")
        self.master1 = self._create_arm(master1_can, arm_type, "Master1")
        
        print(f"\n[2/4] Initializing master arm 2 ({master2_can})...")
        self.master2 = self._create_arm(master2_can, arm_type, "Master2")
        
        print(f"\n[3/4] Initializing follower arm 1 ({follower1_can})...")
        self.follower1 = self._create_arm(follower1_can, arm_type, "Follower1")
        
        print(f"\n[4/4] Initializing follower arm 2 ({follower2_can})...")
        self.follower2 = self._create_arm(follower2_can, arm_type, "Follower2")
        
        print("\n" + "=" * 60)
        print("All robotic arms initialized!")
        print("=" * 60)
        
        # set signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _create_arm(self, can_port: str, arm_type: int, name: str) -> RealSingleArm:
        """
        Create single arm instance (using RealSingleArm)
        
        Args:
            can_port: CAN port
            arm_type: arm type
            name: arm name (for logging)
            
        Returns:
            SingleArm instance
        """
        try:
            arm = RealSingleArm(can_port=can_port, arm_type=arm_type, max_velocity=300, max_acceleration=800)
            print(f"  [{name}] connection successful")
            return arm
        except Exception as e:
            print(f"  [{name}] connection failed: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """signal handler, for graceful exit"""
        print("\n\nReceived exit signal, stopping...")
        self.running = False
    
    def enable_master_gravity_compensation(self):
        """
        Enable master arm gravity compensation (gravity compensation mode)
        Master arm can be freely dragged manually
        """
        print("\nEnabling master arm gravity compensation mode...")
        try:
            self.master1.arm.gravity_compensation()
            print("  [Master1] gravity compensation enabled")
        except Exception as e:
            print(f"  [Master1] enable failed: {e}")
        try:
            self.master2.arm.gravity_compensation()
            print("  [Master2] gravity compensation enabled")
        except Exception as e:
            print(f"  [Master2] enable failed: {e}")

    def sync_followers_to_masters(self):
        """
        Synchronize follower arms to master arms' current position
        Call before starting teaching to ensure follower arms are consistent with master arms
        """
        print("\nSynchronizing follower arms to master arms' position...")
        try:
            master1_pose = self.master1.get_gripper_pose()
            master2_pose = self.master2.get_gripper_pose()
            self.follower1.set_gripper_pose(master1_pose)
            self.follower2.set_gripper_pose(master2_pose)
            print("  Follower arms synchronized to master arms' position")
        except Exception as e:
            print(f"  Synchronization failed: {e}")

    def go_home_all(self):
        """All robotic arms go to zero position"""
        print("\nAll robotic arms go to zero position...")
        try:
            self.master1.go_home()
            self.master2.go_home()
            self.follower1.go_home()
            self.follower2.go_home()
            print("  Zero position command sent")
        except Exception as e:
            print(f"  Zero position failed: {e}")

    def start_teaching(self):
        """
        Starting master-slave teaching
        Master arm gravity compensation, follower arms follow in real-time (joint angles and gripper width 1:1 replication)
        """
        print("\n" + "=" * 60)
        print("Starting master-slave teaching")
        print("=" * 60)
        print("Operation instructions:")
        print("  - Master arm has entered gravity compensation mode, can be freely dragged")
        print("  - Follower arms will replicate master arm's joint angles and gripper width in real-time (1:1 replication)")
        print("  - Press Ctrl+C to stop teaching")
        print("=" * 60 + "\n")
        
        # Enable master arm gravity compensation
        self.enable_master_gravity_compensation()
        time.sleep(0.5)
        self.running = True
        loop_count = 0

        while self.running:
            t_start = time.time()
            try:
                # Read master arm joint angles and gripper width
                master1_joints = self.master1.get_joint_positions()
                master2_joints = self.master2.get_joint_positions()
                master1_gripper = self.master1.get_gripper_width(teacher=True)
                master2_gripper = self.master2.get_gripper_width(teacher=True)

                # Follower arms replicate master arm (joint angles and gripper width 1:1 replication)
                if master1_joints is not None and not np.isnan(master1_joints).any():
                    self.follower1.set_joint_positions(master1_joints)
                    self.follower1.set_gripper_width(master1_gripper)
                if master2_joints is not None and not np.isnan(master2_joints).any():
                    self.follower2.set_joint_positions(master2_joints)
                    self.follower2.set_gripper_width(master2_gripper)

                                # periodically print status
                loop_count += 1
                if loop_count % int(self.control_freq) == 0:
                    print(f"[Master1] joints: {np.round(master1_joints, 3) if master1_joints is not None else 'None'}, gripper: {master1_gripper:.4f} | [Master2] joints: {np.round(master2_joints, 3) if master2_joints is not None else 'None'}, gripper: {master2_gripper:.4f}")
            except Exception as e:
                print(f"Control loop error: {e}")
            elapsed = time.time() - t_start
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)
        print("\nTeaching stopped")
        self._stop_all()

    def _stop_all(self):
        """Stop all robotic arms"""
        print("\nStop all robotic arms...")
        try:
            # Can choose to let arms keep current position or go to zero
            self.go_home_all()
            pass
        except Exception as e:
            print(f"Stop failed: {e}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ALOHA master-slave teaching system')
    parser.add_argument('--master1', type=str, default='can0', 
                        help='CAN port for master arm 1')
    parser.add_argument('--master2', type=str, default='can2',
                        help='CAN port for master arm 2 ')
    parser.add_argument('--follower1', type=str, default='can3', 
                        help='CAN port for follower arm 1')
    parser.add_argument('--follower2', type=str, default='can1', 
                        help='CAN port for follower arm 2')
    
    args = parser.parse_args()
    
    # Create teaching system
    aloha = AlohaTeaching(
        master1_can=args.master1,
        master2_can=args.master2,
        follower1_can=args.follower1,
        follower2_can=args.follower2
    )

    aloha.go_home_all()
    print("Waiting for zero position completion...")
    
    # Start teaching
    aloha.start_teaching()


if __name__ == "__main__":
    main()
