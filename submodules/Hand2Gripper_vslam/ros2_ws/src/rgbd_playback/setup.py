from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rgbd_playback'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ------------------- Add the following line -------------------
        # Meaning: Install all .py files in the 'launch' folder to 'share/package_name/launch' directory
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # -----------------------------------------------------
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'player = rgbd_playback.player_node:main',
            'player_mapper = rgbd_playback.player_mapper_node:main',
            'player_mapper_masker = rgbd_playback.player_mapper_masker_node:main',  # newly added executable file
        ],
    },
)
