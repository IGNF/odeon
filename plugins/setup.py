from setuptools import find_packages, setup

with open('VERSION', 'r') as f:
    version = f.read().strip()

setup(
    name='odeonpluginexample',
    version=version,
    author='samy khelifi',
    author_email='samy.khelifi@ign.fr',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition'
    ],
    packages=find_packages(),
)
