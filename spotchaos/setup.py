# to run: python setup.py build_ext --inplace
#from setuptools import setup,Extension
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

#setup(ext_modules=cythonize('EightBitTransit/cGridFunctions.pyx'))
#setup(ext_modules=cythonize('EightBitTransit/cTransitingImage.pyx'))
#setup(ext_modules=cythonize('EightBitTransit/misc.pyx'))
#setup(ext_modules=cythonize('EightBitTransit/deprecated.pyx'),include_dirs=[np.get_include()])
#setup(ext_modules=cythonize('EightBitTransit/inversion.pyx'),include_dirs=[np.get_include()])

extensions = [Extension('spotchaos.syntheticSignals',['spotchaos/syntheticSignals.pyx'])]

setup(name='spotchaos',
      version='0.0',
      description='',
      author='Emily Sandford',
      author_email='es835@cam.ac.uk',
      url='',
      license='MIT',
      packages=['spotchaos'],
      include_dirs=[np.get_include()],
      #install_requires=['numpy','matplotlib','warnings','scipy','copy','math','itertools','collections'],
      ext_modules=cythonize(extensions))
