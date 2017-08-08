from distutils.core import setup
from Cython.Build import cythonize
    
setup(
  name = 'project-eva',
  ext_modules = cythonize("CDM_SubHalo_Potential.pyx"),
)
