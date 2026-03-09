from setuptools import setup, find_packages

setup(
    name='arx_mujoco',                          # package name
    version='0.1.1',                       # version number
    packages=find_packages(),             # auto-discover packages
    install_requires=[                    # package dependencies
        # 'numpy', 'torch', 'opencv-python',  # example dependencies
    ],
    include_package_data=True,             # include other files
    package_data={                         # additional files
    }
)