# setup.py

from setuptools import setup, find_packages

setup(
    name='diffusion_laboratory',
    version='0.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
        'pyyaml',
        # Add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'diffusion_laboratory=diffusion_laboratory.cli:main',
        ],
    },
)