#!/usr/bin/env python3
import os
from os.path import join, dirname, realpath
from setuptools import find_packages, setup
from distutils.command.build import build as DistutilsBuild


with open(join("mujoco_py", "version.py")) as version_file:
     exec(version_file.read())


def read_requirements_file(filename):
     req_file_path = '%s/%s' % (dirname(realpath(__file__)), filename)
     with open(req_file_path) as f:
         return [line.strip() for line in f]


packages = find_packages()


class Build(DistutilsBuild):
     def run(self):
         os.environ['MUJOCO_PY_FORCE_REBUILD'] = 'True'
         os.environ['MUJOCO_PY_SKIP_ACTIVATE'] = 'True'
         import mujoco_py  # noqa: force build
         DistutilsBuild.run(self)


setup(
     name='mujoco-py',
     version=__version__,  # noqa
     author='OpenAI Robotics Team',
     author_email='robotics@openai.com',
     url='https://github.com/openai/mujoco-py',
     packages=packages,
     include_package_data=True,
     cmdclass={'build': Build},
     package_dir={'mujoco_py': 'mujoco_py'},
     package_data={'mujoco_py': ['generated/*.so']},
     install_requires=read_requirements_file('requirements.txt'),
     tests_require=read_requirements_file('requirements.dev.txt'),
 )


setup(
   name = 'grasp',
   packages = [
       package for package in find_packages() if package.startswith('grasp')
   ],
   eager_resources =['*'],
   include_package_data=True,
   python_requires='>=3',
   description="grasp suite: standard benchmark",
   author='jiali duan',
   url='http://mcl-lab.usc.edu:1313/',
   author_email='jialidua@usc.edu',
   version='1.0'
)
