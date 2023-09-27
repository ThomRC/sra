#!/usr/bin/env python

from setuptools import setup, find_packages
from codecs import open
from os import path
import os

working_dir = path.abspath(path.dirname(__file__))
ROOT = os.path.abspath(os.path.dirname(__file__))

# Read the README.
with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
    README = f.read()

setup(name='sra',
      version='1.0.0',
      description='Implementation of experiments from Smoothed Robustness Analysis paper.',
      long_description=README,
      long_description_content_type='text/markdown',
      packages=find_packages(exclude=['tests*']),
      setup_requires=["chainer==7.8.0", "cupy==7.8.0", "cudatoolkit=11.0"],
      install_requires=["chainer==7.8.0", "cupy==7.8.0", "cudatoolkit=11.0", "ffmpeg-python==0.2.0", "dlib==19.22.0",
                        "scikit-learn==0.23.2", "statsmodels==0.12.1", "openturns==1.16", "matplotlib==3.0.3",
                        "future==0.18.2", "scipy==1.7.1", "threadpoolctl==3.0.0", "seaborn==0.11.0",
                        "pingouin"],
      )
