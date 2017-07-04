#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='bb_backend',
    version='0.9',
    description='Beesbook backend',
    url='https://github.com/BioroboticsLab/beesbook_backend/',
    py_modules=['bb_backend.__init__', 'bb_backend.api'],
    package_dir={'bb_backend': 'plotter'},
)
