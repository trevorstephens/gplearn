#! /usr/bin/env python

"""Genetic Programming in Python, with a scikit-learn inspired API"""

import sys
from setuptools import setup, find_packages
import gplearn

DESCRIPTION = __doc__
VERSION = gplearn.__version__

setup_options = dict(
    name='gplearn',
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
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 2',
                 'Programming Language :: Python :: 2.6',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.3',
                 'Programming Language :: Python :: 3.4'],
    author='Trevor Stephens',
    author_email='trev.stephens@gmail.com',
    license='new BSD',
    packages=find_packages(),
    test_suite='nose.collector',
    zip_safe=False,
    install_requires=['scikit-learn>=0.15'],
    extras_require={'testing': ['nose'],
                    'docs': ['Sphinx']}
)

# For these actions, NumPy is not required. We want them to succeed without,
# for example when pip is used to install seqlearn without NumPy present.
NO_NUMPY_ACTIONS = ('--help-commands', 'egg_info', '--version', 'clean')
if not ('--help' in sys.argv[1:]
        or len(sys.argv) > 1 and sys.argv[1] in NO_NUMPY_ACTIONS):
    import numpy as np
    setup_options['include_dirs'] = [np.get_include()]

setup(**setup_options)
