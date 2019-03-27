#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup
from itertools import chain

pkg_name = 'netodesys'
license = 'MIT'
version = '0.1.0'

tests = ['netodesys.tests']
tests_require = ['pytest>=2.9.2', 'wurlitzer', 'numpy']

extras_req = {
    'testing': ['pytest', 'pytest-cov', 'pytest-flakes', 'pytest-pep8',
                'rstcheck', 'pyodesys[all]']
}
extras_req['all'] = list(chain(*extras_req.values()))

classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics"
]

if __name__ == '__main__':
    setup(
        name=pkg_name.lower(),
        version=version,
        description="Dynamical systems on networks",
        author="Sean P. Cornelius",
        author_email="spcornelius@gmail.com",
        license=license,
        packages=[pkg_name],
        install_requires=['networkx>=2.0', 'pyodesys', 'paramnet>=2.0.0'],
        extras_require=extras_req,
        python_requires='>=3.6',
        classifiers=classifiers)
