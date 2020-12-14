from setuptools import setup, find_packages
import os
import re
import subprocess
import sys
from pkg_resources import Requirement


directory = os.path.dirname(os.path.abspath(__file__))


# Extract version information
path = os.path.join(directory, 'odeon', '__init__.py')
with open(path) as read_file:
    text = read_file.read()
pattern = re.compile(r"^__version__ = ['\"]([^'\"]*)['\"]", re.MULTILINE)
version = pattern.search(text).group(1)

# Extract long_description
path = os.path.join(directory, 'README.md')
with open(path) as read_file:
    long_description = read_file.read()


try:
    gdal_version = subprocess.check_output(
        ['gdal-config', '--version']).decode('utf')
    gdal_config = os.environ.get('GDAL_CONFIG', 'gdal-config')

except Exception:
    sys.exit("GDAL must be installed to use `odeon`.")


# Extract package requirements from Conda environment.yml
install_requires = []
dependency_links = []
path = os.path.join(directory, 'environment.yml')
with open(path) as read_file:
    state = "PREAMBLE"
    for line in read_file:
        line = line.rstrip().lstrip(" -")
        if line == "dependencies:":
            state = "CONDA_DEPS"
        elif line == "pip:":
            state = "PIP_DEPS"
        elif state == "CONDA_DEPS":
            # PyTorch requires substituting the recommended pip dependencies
            requirement = Requirement(line)
            if requirement.key == "pytorch":
                install_requires.append(line.replace("pytorch", "torch", 1))
                install_requires.append("torchvision")
            else:
                # Appends to dependencies
                install_requires.append(line)
        elif state == "PIP_DEPS":
            # Appends to dependency links
            dependency_links.append(line)
            # Adds package name to dependencies
            install_requires.append(line.split("/")[-1].split("@")[0])

setup(
    name='odeon',
    version=version,
    author='ODEON Team',
    description="python toolkit for Object Delineation on Earth Observations with Neural network",
    long_description_content_type='text/markdown',
    long_description=long_description,
    license='MIT',
    url='https://gitlab.com/dai-projets/odeon-landcover',
    keywords='machine-learning, semantic segmentation, earth observation',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License'
    ],
    packages=find_packages(),
    include_package_data=True,
    entry_points={'console_scripts': ['odeon = odeon.main:main']},
    install_requires=install_requires
)
