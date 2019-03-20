from setuptools import setup, find_packages

setup(name='lbmpy_walberla',
      description='Code Generation for Lattice Boltzmann Methods in the walberla framework',
      author='Martin Bauer',
      license='AGPLv3',
      author_email='martin.bauer@fau.de',
      url='https://i10git.cs.fau.de/software/pystencils/',
      packages=['lbmpy_walberla'] + ['lbmpy_walberla.' + s for s in find_packages('lbmpy_walberla')],
      install_requires=['lbmpy', 'pystencils_walberla'],
      package_dir={'lbmpy_walberla': 'lbmpy_walberla'},
      package_data={'lbmpy_walberla': ['templates/*']},
      version_format='{tag}.dev{commits}+{sha}',
      )
