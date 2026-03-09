import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_trajectory(traj_dir):
    """
    Load trajectory files (.npy).
    Assume file names are 0.npy, 1.npy, ...
    """
    if not os.path.exists(traj_dir):
        print(f"Directory does not exist: {traj_dir}")
        return None

    # sort by number
    traj_files = sorted(glob.glob(os.path.join(traj_dir, "*.npy")), 
                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    if not traj_files:
        print(f"No .npy files found in {traj_dir}")
        return None

    trajectory = [np.load(f) for f in traj_files]
    return np.array(trajectory)

def plot_trajectory(trajectory, title="Camera Trajectory"):
    """
    Plot camera trajectory and its orientation.
    trajectory: (N, 4, 4) numpy array
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # extract positions
    positions = trajectory[:, :3, 3]
    
    # plot trajectory line
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=title, linewidth=2)
    
    # label start and end points
    if len(positions) > 0:
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', marker='o', s=100, label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', marker='s', s=100, label='End')

    # plot camera pose directions (every few frames)
    if len(positions) > 1:
        max_range = np.max(np.max(positions, axis=0) - np.min(positions, axis=0))
        axis_length = max_range * 0.1 if max_range > 0 else 0.1
    else:
        axis_length = 0.1
    
    step = max(1, len(trajectory) // 20) # draw up to 20 coordinate systems
    for i in range(0, len(trajectory), step):
        T_w_c = trajectory[i]
        origin = T_w_c[:3, 3]
        R_w_c = T_w_c[:3, :3]
        # draw Z-axis (blue) indicating camera direction
        ax.quiver(origin[0], origin[1], origin[2], R_w_c[0, 2], R_w_c[1, 2], R_w_c[2, 2], color='b', length=axis_length, alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    
    # keep axis proportions consistent
    try:
        ax.set_aspect('equal', adjustable='box')
    except:
        pass # some matplotlib versions may not support equal aspect in 3d
        
    plt.show()

if __name__ == "__main__":
    # define trajectory folder path
    TRAJ_DIR = "./ros2_ws/camera_data/traj"
    
    print(f"Loading trajectory: {TRAJ_DIR}")
    trajectory = load_trajectory(TRAJ_DIR)
    
    if trajectory is not None and len(trajectory) > 0:
        print(f"Loaded successfully, {len(trajectory)} frames.")
        plot_trajectory(trajectory, title="Recorded Trajectory")
    else:
        print("Unable to load trajectory or trajectory is empty.")
