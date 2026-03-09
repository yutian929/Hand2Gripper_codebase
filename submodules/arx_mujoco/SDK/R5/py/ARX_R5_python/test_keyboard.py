from bimanual import SingleArm
from typing import Dict, Any
import numpy as np
import numpy.typing as npt
import curses
import time

arm_config: Dict[str, Any] = {
    "can_port": "can1",
    "type": 0,
    # Add necessary configuration parameters for the left arm
}
single_arm = SingleArm(arm_config)

# use curses to capture keyboard input
def keyboard_control(stdscr):

    curses.curs_set(0)  # do not display cursor
    stdscr.nodelay(1)   # set to non-blocking mode
    stdscr.timeout(10)  # set keyboard read timeout
    global target_pose
    curses.mousemask(0)  # disable mouse events
    xyzrpy = np.zeros(6)
    gripper =0
    # “love needs no words"
    big_text = [
        "  AAAAA        RRRRR         X   X     ",
        " A     A       R    R         X X      ",
        " AAAAAAA       RRRRR           X       ",
        " A     A       R  R           X X      ",
        " A     A       R   RR        X   X     ",
        " A     A       R    R       X     X    "
    ]
    
    while True:
        key = stdscr.getch()  # get keyboard input
        stdscr.clear()
        # format each element in the return value
        ee_pose = single_arm.get_ee_pose_xyzrpy()
        joint_pos = single_arm.get_joint_positions()
        joint_vel = single_arm.get_joint_velocities()
        joint_curr = single_arm.get_joint_currents()

        # format and display
        stdscr.addstr(0, 0, f"EE_POSE: [{' '.join([f'{val:.3f}' for val in ee_pose])}]")
        stdscr.addstr(2, 0, f"JOINT_POS: [{' '.join([f'{val:.3f}' for val in joint_pos])}]")
        stdscr.addstr(4, 0, f"JOINT_VEL: [{' '.join([f'{val:.3f}' for val in joint_vel])}]")
        stdscr.addstr(6, 0, f"JOINT_CURR: [{' '.join([f'{val:.3f}' for val in joint_curr])}]")
        
        if key == ord('q'):  # press 'q' to exit program
            break
        if key == -1:  # press 'q' to exit program
            continue     
        elif key == ord('i'): 
            single_arm.gravity_compensation()
            value=single_arm.get_ee_pose_xyzrpy()

        elif key == ord('w'): 
            xyzrpy[0] += 0.005  # arm move forward
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == ord('s'): 
            xyzrpy[0] -= 0.005  # arm move backward
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == ord('a'):
            xyzrpy[1] += 0.005  # arm move left
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == ord('d'):  
            xyzrpy[1] -= 0.005  # arm move right
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == curses.KEY_UP: 
            xyzrpy[2] += 0.005  # arm move up
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == curses.KEY_DOWN: 
            xyzrpy[2] -= 0.005  # arm move down
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == curses.KEY_LEFT: 
            xyzrpy[1] += 0.005  # arm move left
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)  
        elif key == curses.KEY_RIGHT: 
             xyzrpy[1] -= 0.005  # arm move right
             single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)  
        elif key == ord(','): 
            xyzrpy[5] += 0.02  # arm yaw decrease
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == ord('/'): 
            xyzrpy[5] -= 0.02  # arm yaw increase
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == ord('m'): 
            xyzrpy[3] += 0.02  # arm roll increase
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == ord('n'): 
            xyzrpy[3] -= 0.02  # arm roll decrease
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == ord('l'): 
            xyzrpy[4] += 0.02  # arm pitch increase
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == ord('.'): 
            xyzrpy[4] -= 0.02  # arm pitch decrease
            single_arm.set_ee_pose_xyzrpy(xyzrpy=xyzrpy)
        elif key == ord('c'): 
            gripper -= 0.2  # close
            single_arm.set_catch_pos(pos=gripper)
        elif key == ord('o'): 
            gripper += 0.2  # open
            single_arm.set_catch_pos(pos=gripper)
        elif key == ord('r'): 
            xyzrpy = np.zeros(6)
            single_arm.go_home()
            print('return to origin\n')

        # height, width = stdscr.getmaxyx()

        # update screen to display current target pose
        # stdscr.addstr(0, 0, f"Current Target Pose: {xyzrpy}")
        # for i, line in enumerate(big_text):
        #     stdscr.addstr(height // 2 - 3 + i, (width - len(line)) // 2, line)

        stdscr.refresh()

if __name__ == "__main__":
    curses.wrapper(keyboard_control)
