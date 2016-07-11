#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from setuptools import setup, dist
dist.Distribution(dict(setup_requires=['bob.extension']))

from bob.extension.utils import load_requirements, find_packages
install_requires = load_requirements()

setup(

    name='bob.bio.vein',
    version=open("version.txt").read().rstrip(),
    description='Vein recognition based on Bob and the bob.bio framework',

    url='https://gitlab.idiap.ch/biometric/bob.bio.vein',
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

      # registered database short cuts
      'bob.bio.database': [
        'utfvp = bob.bio.vein.configurations.databases.utfvp:database',
        'vera = bob.bio.vein.configurations.databases.vera:database',
      ],

      # registered preprocessors
      'bob.bio.preprocessor': [
        'none = bob.bio.vein.configurations.preprocessors.finger_crop_None_None:preprocessor',
        'histeq = bob.bio.vein.configurations.preprocessors.finger_crop_None_HE:preprocessor',
        'highfreq = bob.bio.vein.configurations.preprocessors.finger_crop_None_HFE:preprocessor',
        'circGabor = bob.bio.vein.configurations.preprocessors.finger_crop_None_CircGabor:preprocessor',

      ],

      # registered feature extractors
      'bob.bio.extractor': [
        'ncc-normalisedcrosscorr = bob.bio.vein.configurations.extractors.normalised_crosscorr:feature_extractor',
        'mc-maximumcurvature = bob.bio.vein.configurations.extractors.maximum_curvature:feature_extractor',
        'rlt-repeatedlinetracking = bob.bio.vein.configurations.extractors.repeated_line_tracking:feature_extractor',
        'wld-widelinedetector = bob.bio.vein.configurations.extractors.wide_line_detector:feature_extractor',
        'lbp-localbinarypatterns = bob.bio.vein.configurations.extractors.lbp:feature_extractor',
      ],

      # registered fingervein recognition algorithms
      'bob.bio.algorithm': [
        'match-wld = bob.bio.vein.configurations.algorithms:huangwl_tool',
        'match-wld-gpu = bob.bio.vein.configurations.algorithms:huangwl_gpu_tool',
        'match-mc = bob.bio.vein.configurations.algorithms:miuramax_tool',
        'match-mc-gpu = bob.bio.vein.configurations.algorithms:miuramax_gpu_tool',
        'match-rlt = bob.bio.vein.configurations.algorithms:miurarlt_tool',
        'match-rlt-gpu = bob.bio.vein.configurations.algorithms:miurarlt_gpu_tool',
        #'match-lbp = bob.bio.face.configurations.algorithms.lgbphs:tool',
       ],

      # registered SGE grid configuration files
      'facereclib.grid': [
        'gpu = bob.bio.vein.configurations.grid.gpu:grid',
        'gpu2 = bob.bio.vein.configurations.grid.gpu2:grid',
        'gpu3 = bob.bio.vein.configurations.grid.gpu3:grid',
        'grid = bob.bio.vein.configurations.grid.grid:grid',
        'demanding = bob.bio.vein.configurations.grid.demanding:grid',
        'very-demanding = bob.bio.vein.configurations.grid.very_demanding:grid',
        'gbu = bob.bio.vein.configurations.grid.gbu:grid',
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
