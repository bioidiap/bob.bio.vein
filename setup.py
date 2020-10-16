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
        # protocols TODO: add all protocols
          # verafinger
          'Nom = bob.bio.vein.configurations.database.protocol.Nom',
          'Cropped-Nom = bob.bio.vein.configurations.database.protocol.cropped_Nom',
          'B = bob.bio.vein.configurations.database.protocol.b',
          'Cropped-B = bob.bio.vein.configurations.database.protocol.cropped_b',
          'Full = bob.bio.vein.configurations.database.protocol.full',
          'Cropped-Full = bob.bio.vein.configurations.database.protocol.cropped_full',
          'Fifty = bob.bio.vein.configurations.database.protocol.fifty',
          'Cropped-Fifty = bob.bio.vein.configurations.database.protocol.cropped_fifty',
          # utfvp
          # 'nom = bob.bio.vein.configurations.database.protocol.nom',
          # fv3d
          'central = bob.bio.vein.configurations.database.protocol.central',
          'left = bob.bio.vein.configurations.database.protocol.left',
          'right = bob.bio.vein.configurations.database.protocol.right',
          'stitched = bob.bio.vein.configurations.database.protocol.stitched',
          # putvein
          # 'wrist-LR_1 = bob.bio.vein.configurations.database.protocol.wristLR_1',
          # TODO That will be a LOT of protocols, for putvein...

        # legacy databases
        'verafinger = bob.bio.vein.configurations.database.verafinger',
        'utfvp = bob.bio.vein.configurations.database.utfvp',
        'fv3d = bob.bio.vein.configurations.database.fv3d',
        'putvein = bob.bio.vein.configurations.database.putvein',

        # legacy baselines
        'mc = bob.bio.vein.configurations.maximum_curvature',
        'rlt = bob.bio.vein.configurations.repeated_line_tracking',
        'wld = bob.bio.vein.configurations.wide_line_detector',

        # other (legacy)
        'parallel = bob.bio.vein.configurations.parallel',
        'gridio4g48 = bob.bio.vein.configurations.gridio4g48',
        'grid = bob.bio.vein.configurations.gridio4g48',
        ],

     'bob.bio.database': [
        'verafinger = bob.bio.vein.configurations.database.verafinger:database',
        'utfvp = bob.bio.vein.configurations.database.utfvp:database',
        'fv3d = bob.bio.vein.configurations.database.fv3d:database',
        'putvein = bob.bio.vein.configurations.database.putvein:database',
      ],

      'bob.bio.pipeline': [
        'wld = bob.bio.vein.configurations.wide_line_detector:pipeline',
        'mc = bob.bio.vein.configurations.maximum_curvature:pipeline',
        'rlt = bob.bio.vein.configurations.repeated_line_tracking:pipeline',
      ],

      'console_scripts': [
        'bob_bio_vein_compare_rois.py = bob.bio.vein.script.compare_rois:main',
        'bob_bio_vein_view_sample.py = bob.bio.vein.script.view_sample:main',
        'bob_bio_vein_blame.py = bob.bio.vein.script.blame:main',
        'bob_bio_vein_markdet.py = bob.bio.vein.script.markdet:main',
        'bob_bio_vein_watershed_mask.py = bob.bio.vein.script.watershed_mask:main',
        ]
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
