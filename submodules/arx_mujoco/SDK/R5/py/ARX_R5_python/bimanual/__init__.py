import os
import sys

# get current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

def find_first_specific_so_file(root_dir,file_name):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # check if filename starts with 'arx_l5pro_python_api' and ends with '.so'
            if filename.startswith(file_name) and filename.endswith('.so'):
                # return the first found file path that meets the conditions
                return os.path.join(dirpath, filename)
    return None  # if no matching file is found, return None

# use os.path.join to concatenate paths
so_file = find_first_specific_so_file(os.path.join(current_dir, 'api', 'arx_r5_python'),'arx_r5_python.')

# ensure shared library path is in Python path
if os.path.exists(so_file):
    sys.path.append(os.path.dirname(so_file))  # add shared library directory to sys.path
else:
    raise FileNotFoundError(f"Shared library not found: {so_file}")

# use os.path.join to concatenate paths
so_file = find_first_specific_so_file(os.path.join(current_dir, 'api'),'kinematic_solver.')

# ensure shared library path is in Python path
if os.path.exists(so_file):
    sys.path.append(os.path.dirname(so_file))  # add shared library directory to sys.path
else:
    raise FileNotFoundError(f"Shared library not found: {so_file}")

# import Python module
try:
    from .script.dual_arm import *  # ensure these two classes are properly defined in their respective files
    from .script.single_arm import *
    from .script.solver import *
except ImportError as e:
    raise ImportError(f"Failed to import Python modules: {e}")

# optional: define __all__ to control module exports
# __all__ = ['BimanualArm', 'SingleArm', 'arx_r5_python']
