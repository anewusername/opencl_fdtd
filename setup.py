#!/usr/bin/env python3

from setuptools import setup, find_packages
import opencl_fdtd

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='opencl_fdtd',
      version=opencl_fdtd.version,
      description='OpenCL FDTD solver',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Jan Petykiewicz',
      author_email='anewusername@gmail.com',
      url='https://mpxd.net/code/jan/opencl_fdtd',
      packages=find_packages(),
      package_data={
          'opencl_fdfd': ['kernels/*']
      },
      install_requires=[
            'numpy',
            'pyopencl',
            'jinja2',
            'fdfd_tools>=0.3',
      ],
      extras_require={
      },
      )

