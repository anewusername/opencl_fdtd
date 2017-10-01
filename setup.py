#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='opencl_fdtd',
      version='0.4',
      description='OpenCL FDTD solver',
      author='Jan Petykiewicz',
      author_email='anewusername@gmail.com',
      url='https://mpxd.net/gogs/jan/opencl_fdtd',
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

