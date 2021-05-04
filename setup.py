from setuptools import find_packages, setup

setup(
        name="trainer",
        version=0.1,
        install_requires=["pytorch-lightning", "torch", "webdataset", "google-cloud-storage==1.38.0", "numpy",
                          "torchvision"],
        packages=find_packages(),
        include_package_data=True,
        )
