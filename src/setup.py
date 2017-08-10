from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

include_path = [numpy.get_include()]

extensions = [
    Extension('ldsinference', ['ldsinference.pyx'],
              include_dirs=[numpy.get_include()]),
    Extension('utils', ['utils.pyx'],
              include_dirs=[numpy.get_include()])
]

setup(
  ext_modules=cythonize(extensions, gdb_debug=True)
)
