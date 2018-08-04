#! /usr/bin/env python

"""Genetic Programming in Python, with a scikit-learn inspired API"""

from setuptools import setup, find_packages
import gplearn

DESCRIPTION = __doc__
VERSION = gplearn.__version__

setup(name='gplearn',
      version=VERSION,
      description=DESCRIPTION,
      long_description=open("README.rst").read(),
      classifiers=['Development Status :: 3 - Alpha',
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7'],
      author='Trevor Stephens',
      author_email='trev.stephens@gmail.com',
      url='https://github.com/trevorstephens/gplearn',
      license='new BSD',
      packages=find_packages(),
      test_suite='nose.collector',
      zip_safe=False,
      package_data={'': ['LICENSE'],
                    'gplearn': ['tests/*.py']},
      install_requires=['scikit-learn>=0.18.1'],
      extras_require={'testing': ['nose'],
                      'docs': ['Sphinx']})
