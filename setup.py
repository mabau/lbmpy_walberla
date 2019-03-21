import os
import sys
from setuptools import setup, find_packages
sys.path.insert(0, os.path.abspath('..'))
from custom_pypi_index.pypi_index import get_current_dev_version_from_git


setup(name='lbmpy_walberla',
      version=get_current_dev_version_from_git(),
      description='Code Generation for Lattice Boltzmann Methods in the walberla framework',
      author='Martin Bauer',
      license='AGPLv3',
      author_email='martin.bauer@fau.de',
      url='https://i10git.cs.fau.de/software/pystencils/',
      packages=['lbmpy_walberla'] + ['lbmpy_walberla.' + s for s in find_packages('lbmpy_walberla')],
      install_requires=['lbmpy', 'pystencils_walberla'],
      package_dir={'lbmpy_walberla': 'lbmpy_walberla'},
      package_data={'lbmpy_walberla': ['templates/*']},
      )
