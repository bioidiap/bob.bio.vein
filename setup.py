#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from setuptools import setup, dist
dist.Distribution(dict(setup_requires=['bob.extension']))

from bob.extension.utils import load_requirements, find_packages
install_requires = load_requirements()

setup(

    name='bob.bio.vein',
    version=open("version.txt").read().rstrip(),
    description='Vein Recognition Library',

    url='https://gitlab.idiap.ch/bob/bob.bio.vein',
    license='GPLv3',

    author='Andre Anjos,Pedro Tome',
    author_email='andre.anjos@idiap.ch,pedro.tome@idiap.ch',

    keywords = "bob, biometric recognition, evaluation, vein",

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,
    zip_safe = False,

    install_requires=install_requires,

    entry_points={

      'bob.bio.database': [
        'verafinger = bob.bio.db.default_configs.verafinger:database',
        'utfvp = bob.bio.db.default_configs.utfvp:database',
        ],

      'bob.bio.preprocessor': [
        'nopp = bob.bio.vein.configurations.preprocessors:none',
        'histeq = bob.bio.vein.configurations.preprocessors:he',
        'highfreq = bob.bio.vein.configurations.preprocessors:hfe',
        'circgabor = bob.bio.vein.configurations.preprocessors:circgabor',
        ],

      'bob.bio.extractor': [
        'normalisedcrosscorr = bob.bio.vein.configurations.extractors.normalised_crosscorr:feature_extractor',
        'maximumcurvature = bob.bio.vein.configurations.extractors.maximum_curvature:feature_extractor',
        'repeatedlinetracking = bob.bio.vein.configurations.extractors.repeated_line_tracking:feature_extractor',
        'widelinedetector = bob.bio.vein.configurations.extractors.wide_line_detector:feature_extractor',
        'localbinarypatterns = bob.bio.vein.configurations.extractors.lbp:feature_extractor',
        ],

      'bob.bio.algorithm': [
        'match-wld = bob.bio.vein.configurations.algorithms:huangwl',
        'match-mc = bob.bio.vein.configurations.algorithms:miuramax',
        'match-rlt = bob.bio.vein.configurations.algorithms:miurarlt',
        #'match-lbp = bob.bio.face.configurations.algorithms.lgbphs:tool',
        ],

      'bob.bio.grid': [
        'idiap = bob.bio.vein.configurations.grid:default',
        ],

      },

    classifiers = [
      'Framework :: Bob',
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'Topic :: Software Development :: Libraries :: Python Modules',
      ],

)
