from setuptools import setup, find_packages

PROJECT = "ks_machine_learning"
PACKAGES = find_packages("src")

setup(
    name=PROJECT,
    packages=PACKAGES,
    package_dir={"": "src"},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "ksml = ks_machine_learning.__main__:cli"
        ]
    }
)
