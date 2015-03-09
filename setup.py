#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Pedro Tome <pedro.tome@idiap.ch>
# Tue 25 Mar 18:18:08 2014 CEST
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# This file contains the python (distutils/setuptools) instructions so your
# package can be installed on **any** host system. It defines some basic
# information like the package name for instance, or its homepage.
#
# It also defines which other packages this python package depends on and that
# are required for this package's operation. The python subsystem will make
# sure all dependent packages are installed or will install them for you upon
# the installation of this package.
#
# The 'buildout' system we use here will go further and wrap this package in
# such a way to create an isolated python working environment. Buildout will
# make sure that dependencies which are not yet installed do get installed, but
# **without** requiring adminstrative privileges on the host system. This
# allows you to test your package with new python dependencies w/o requiring
# administrative interventions.

from setuptools import setup, find_packages

# Define package version
version = open("version.txt").read().rstrip()

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    # This is the basic information about your project. Modify all this
    # information before releasing code publicly.
    name='bob.fingervein',
    version=version,
    description='Fingervein recognition based on Bob and the facereclib',

    url='https://github.com/bioidiap/bob.fingervein',
    license='LICENSE.txt',
    
    author='Pedro Tome',
    author_email='pedro.tome@idiap.ch',
        
    keywords = "Fingervein recognition, fingervein verification, reproducible research, facereclib",

    # If you have a better, long description of your package, place it on the
    # 'doc' directory and then hook it here
    long_description=open('README.rst').read(),

    # This line is required for any distutils based packaging.
    packages=find_packages(),
    include_package_data=True,
    zip_safe = False,

    
    install_requires=[
      'setuptools',
      'bob.io.base',
      'bob.core',
      'bob.ip.base',
      'bob.sp',
      'bob.io.matlab',
      'facereclib',
      'bob.db.vera',
      'bob.db.utfvp',

    ],

    namespace_packages = [
      'bob',
    ],

    entry_points={

      # scripts should be declared using this entry:
      'console_scripts': [
	      'fingerveinverify.py = bob.fingervein.script.fingerveinverify:main',
        'scores2spoofingfile.py = bob.fingervein.script.scores2spoofingfile:main',		
        #'scoresanalysis.py = bob.fingervein.script.scoresanalysis:main',
        #'scoresfusion.py = bob.fingervein.script.scoresfusion:main',
        #'plot_scatter_fusion.py = bob.fingervein.script.plot_scatter_fusion:main',
      ],
      
      # registered database short cuts
      'facereclib.database': [
        'utfvp             = bob.fingervein.configurations.databases.utfvp:database',
        'vera              = bob.fingervein.configurations.databases.vera:database',
      ],

      # registered preprocessings
      'facereclib.preprocessor': [
        'none = bob.fingervein.configurations.preprocessing.finger_crop_None_None:preprocessor',
        'histeq = bob.fingervein.configurations.preprocessing.finger_crop_None_HE:preprocessor',
        'highfreq = bob.fingervein.configurations.preprocessing.finger_crop_None_HFE:preprocessor',
        'circGabor = bob.fingervein.configurations.preprocessing.finger_crop_None_CircGabor:preprocessor',
        
      ],


      # registered feature extractors
      'facereclib.feature_extractor': [
        'ncc-normalisedcrosscorr    = bob.fingervein.configurations.features.normalised_crosscorr:feature_extractor',
        'mc-maximumcurvature        = bob.fingervein.configurations.features.maximum_curvature:feature_extractor',
        'rlt-repeatedlinetracking   = bob.fingervein.configurations.features.repeated_line_tracking:feature_extractor',
        'wld-widelinedetector       = bob.fingervein.configurations.features.wide_line_detector:feature_extractor',
        'lbp-localbinarypatterns    = bob.fingervein.configurations.features.lbp:feature_extractor',
        
      ],

      # registered fingervein recognition algorithms
      'facereclib.tool': [
        'match-wld      = bob.fingervein.configurations.tools:huangwl_tool',
        'match-wld-gpu  = bob.fingervein.configurations.tools:huangwl_gpu_tool',
        'match-mc       = bob.fingervein.configurations.tools:miuramax_tool',
        'match-mc-gpu   = bob.fingervein.configurations.tools:miuramax_gpu_tool',
        'match-rlt      = bob.fingervein.configurations.tools:miurarlt_tool',
        'match-rlt-gpu  = bob.fingervein.configurations.tools:miurarlt_gpu_tool',
        'match-lbp      = facereclib.configurations.tools.lgbphs:tool',
       ], 

      # registered SGE grid configuration files
      'facereclib.grid': [
        'gpu               = bob.fingervein.configurations.grid.gpu:grid',
        'gpu2              = bob.fingervein.configurations.grid.gpu2:grid',
        'gpu3              = bob.fingervein.configurations.grid.gpu3:grid',
        'grid              = bob.fingervein.configurations.grid.grid:grid',
        'demanding         = bob.fingervein.configurations.grid.demanding:grid',
        'very-demanding    = bob.fingervein.configurations.grid.very_demanding:grid',
        'gbu               = bob.fingervein.configurations.grid.gbu:grid',
        'small             = bob.fingervein.configurations.grid.small:grid',        
      ],

      # tests that are _exported_ (that can be executed by other packages) can
      # be signalized like this:
      'bob.test': [
        'tests = bob.fingervein.tests.test:FingerveinTests',
        #'preprocessors       = bob.fingervein.tests.test_preprocessing:PreprocessingTest',
        #'feature_extractors  = bob.fingervein.tests.test_features:FeatureExtractionTest',
        #'matching            = bob.fingervein.tests.test_matching:MatchingTest',
        
      ],
   
      },

    # Classifiers are important if you plan to distribute this package through
    # PyPI. You can find the complete list of classifiers that are valid and
    # useful here (http://pypi.python.org/pypi?%3Aaction=list_classifiers).
    classifiers = [
      'Framework :: Bob',
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'Topic :: Software Development :: Libraries :: Python Modules',
      ],
)
