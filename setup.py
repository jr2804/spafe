# -*- coding: utf-8 -*-
import numpy
import pathlib
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

# The directory containing this file
path = pathlib.Path(__file__).parent

# get readme text
readme = (path / "README.md").read_text()

# define requirements
requires = ["numpy>=1.17.2", "scipy>=1.3.1"]


setup (
        name         = 'spafe',
        version      = '0.1.2',
        author       = 'SuperKogito',
        author_email = 'superkogito@gmail.com',
        description  = 'Simplified python Audio Features Extraction',
        long_description = readme,
        long_description_content_type = "text/markdown",
        license      = 'BSD',
        url          = 'https://github.com/SuperKogito/spafe',
        packages     = find_packages(),
        classifiers  = [
                        'Development Status :: 3 - Alpha',
                        'Environment :: Console',
                        'Environment :: Web Environment',
                        'Intended Audience :: Developers',
                        'License :: OSI Approved :: BSD License',
                        'Operating System :: OS Independent',
                        'Programming Language :: Python',
                        'Topic :: Documentation',
                        'Topic :: Utilities',
                      ],
        platforms            = 'any',
        include_package_data = True,
        install_requires     = requires,

        ext_modules = cythonize(["spafe/cutils/cythonfuncs.pyx"],
                                 annotate=True,
                                 language_level=3),
        include_dirs=[numpy.get_include()]
)
