import setuptools

setuptools.setup(
    name="hand2gripper",
    version="0.1",
    packages=setuptools.find_packages(exclude=["submodules", "submodules.*"]),
)