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

      'bob.bio.config': [
        # databases
        'verafinger = bob.bio.vein.configurations.verafinger',
        'utfvp = bob.bio.vein.configurations.utfvp',

        # baselines
        'mc = bob.bio.vein.configurations.maximum_curvature',
        'rlt = bob.bio.vein.configurations.repeated_line_tracking',
        'wld = bob.bio.vein.configurations.wide_line_detector',

        # other
        'parallel = bob.bio.vein.configurations.parallel',
        ],


      'console_scripts': [
        'compare_rois.py = bob.bio.vein.script.compare_rois:main',
        'view_mask.py = bob.bio.vein.script.view_mask:main',
        ]

      # registered database short cuts
      'bob.bio.database': [
        'utfvp = bob.bio.vein.configurations.databases.utfvp:database',
        'vera = bob.bio.vein.configurations.databases.vera:database',
	'biowave_test = bob.bio.vein.configurations.databases.biowave_test:database',
      ],

      # registered preprocessors
      'bob.bio.preprocessor': [
        'none = bob.bio.vein.configurations.preprocessors.finger_crop_None_None:preprocessor',
        'histeq = bob.bio.vein.configurations.preprocessors.finger_crop_None_HE:preprocessor',
        'highfreq = bob.bio.vein.configurations.preprocessors.finger_crop_None_HFE:preprocessor',
        'circGabor = bob.bio.vein.configurations.preprocessors.finger_crop_None_CircGabor:preprocessor',
        'topography-cut-roi = bob.bio.vein.configurations.preprocessors.topography_cut_roi:preprocessor', # topography cut roi
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
