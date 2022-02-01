#!python
#cython: language_level=3

"""
Cython Make File - Contains instructions for the compiler
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("rdf_calc.pyx")
)
