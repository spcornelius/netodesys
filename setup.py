#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distutils.core import setup
from itertools import chain

pkg_name = 'netodesys'
license = 'BSD'
version = '0.1'

tests = ['netodesys.tests']
tests_require = ['pytest>=2.9.2']

extras_req = {
    'testing': ['pytest', 'pytest-cov', 'pytest-flakes', 'pytest-pep8', 'rstcheck']
}
extras_req['all'] = list(chain(extras_req.values()))

if __name__ == '__main__':
    setup(
        name=pkg_name.lower(),
        version=version,
        description="NetworkX + ODE dynamics",
        author="Sean P. Cornelius",
        author_email="spcornelius@gmail.com",
        license=license,
        packages=[pkg_name],
        install_requires=['networkx>=2.0', 'indexed.py', 'pyodesys[all]'],
        tests_require=['pytest>=2.9.2', 'numpy', 'sympy'],
        extras_require=extras_req,
        python_requires='>=3.6')
