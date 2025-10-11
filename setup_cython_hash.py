"""
Setup script for building the Cython array hashing extension.

Usage:
    python setup_cython_hash.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "array_hash_cython",
        ["array_hash_cython.pyx"],
        language="c",
    )
]

setup(
    name="array_hash_cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        }
    ),
)
