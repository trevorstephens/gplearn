import sys
from setuptools import setup


setup(
    name='gplearn',
    version='0.0.1',
    description='Genetic Programming in Python, with a scikit-learn inspired API',
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
    packages=['gplearn'],
    test_suite='nose.collector',
    install_requires=['scikit-learn>=0.15'],
    tests_require=['nose'],
    zip_safe=False
)
