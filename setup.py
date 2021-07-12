#!/usr/bin/env python3

from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

with open('opencl_fdtd/VERSION.py', 'rt') as f:
    version = f.readlines()[2].strip()

setup(name='opencl_fdtd',
      version=version,
      description='OpenCL FDTD solver',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Jan Petykiewicz',
      author_email='jan@mpxd.net',
      url='https://mpxd.net/code/jan/opencl_fdtd',
      packages=find_packages(),
      package_data={
          'opencl_fdfd': ['kernels/*']
      },
      install_requires=[
            'numpy',
            'pyopencl',
            'jinja2',
            'meanas>=0.3',
      ],
      extras_require={
      },
      )

