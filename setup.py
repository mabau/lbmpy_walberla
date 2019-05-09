from setuptools import setup, find_packages
import subprocess


def version_number_from_git(tag_prefix='release/', sha_length=10, version_format="{version}.dev{commits}+{sha}"):
    def get_released_versions():
        tags = sorted(subprocess.getoutput('git tag').split('\n'))
        versions = [t[len(tag_prefix):] for t in tags if t.startswith(tag_prefix)]
        return versions

    def tag_from_version(v):
        return tag_prefix + v

    def increment_version(v):
        parsed_version = [int(i) for i in v.split('.')]
        parsed_version[-1] += 1
        return '.'.join(str(i) for i in parsed_version)

    latest_release = get_released_versions()[-1]
    commits_since_tag = subprocess.getoutput('git rev-list {}..HEAD --count'.format(tag_from_version(latest_release)))
    sha = subprocess.getoutput('git rev-parse HEAD')[:sha_length]
    is_dirty = len(subprocess.getoutput("git status --untracked-files=no -s")) > 0

    if int(commits_since_tag) == 0:
        version_string = latest_release
    else:
        next_version = increment_version(latest_release)
        version_string = version_format.format(version=next_version, commits=commits_since_tag, sha=sha)

    if is_dirty:
        version_string += ".dirty"
    return version_string


setup(name='lbmpy_walberla',
      version=version_number_from_git(),
      description='Code Generation for Lattice Boltzmann Methods in the walberla framework',
      author='Martin Bauer',
      license='AGPLv3',
      author_email='martin.bauer@fau.de',
      url='https://i10git.cs.fau.de/pycodegen/lbmpy_walberla/',
      packages=['lbmpy_walberla'] + ['lbmpy_walberla.' + s for s in find_packages('lbmpy_walberla')],
      install_requires=['lbmpy', 'pystencils_walberla'],
      package_dir={'lbmpy_walberla': 'lbmpy_walberla'},
      package_data={'lbmpy_walberla': ['templates/*']},
      test_suite='lbmpy_walberla_tests',
      )
