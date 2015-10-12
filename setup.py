#!/usr/bin/env python
'''module mimopy'''
import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "mimopy",
    version = "0.0.1",
    author = "Espen Hagen",
    author_email = "e.hagen@fz-juelich.de",
    description = ("Multiple-input multiple-output (MIMO) transfer function estimation in Python"),
    license = "BSD",
    keywords = "example documentation tutorial",
    url = "http://github.com/espenhgn/mimopy",
    packages=['mimo'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
