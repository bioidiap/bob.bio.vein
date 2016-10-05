#!/usr/bin/env python

"""
`BioWave V1`_ Database. This is a database of wrist vein images that are acquired using BIOWATCH biometric sensor. For each subject of the database there are 3 session images (sessions were held at least 24 hours apart). Each session consists of 5 attempts, in each attempt 5 images were acquired, meaning, that there are ``3 sessions x 5 attempts x 5 images = 75 images`` images per each person's hand, ``75 x 2 images`` per person.
Images were previously manually evaluated, and if any of the ``75`` one hand's images were unusable (too blurred, veins couldn't be seen, etc), than all hand data were discarded. That is way some persons has only 1 hand's images in the database.

Statistics of the data - in total 111 hands;

1) Users with both hands images - 53
2) Users with only R hand images - 4
3) Users with only L hand images - 1

You can find more information there - `BioWave V1`_ .

.. include:: links.rst
"""
from bob.bio.vein.database import BiowaveV1BioDatabase

biowave_v1_image_directory = "[YOUR_BIOWAVE_V1_IMAGE_DIRECTORY]"
"""Value of ``~/.bob_bio_databases.txt`` for this database"""

database = BiowaveV1BioDatabase(
           original_directory=biowave_v1_image_directory,
           original_extension = '.png',
           )
"""The :py:class:`bob.bio.base.database.BioDatabase` derivative with BioWave V1 settings

.. warning::

   This class only provides a programmatic interface to load data in an orderly
   manner, respecting usage protocols. It does **not** contain the raw
   datafiles. You should procure those yourself.

Notice that ``original_directory`` is set to ``[YOUR_BIOWAVE_V1_IMAGE_DIRECTORY]``.
You must make sure to create ``${HOME}/.bob_bio_databases.txt`` setting this
value to the place where you actually installed the Verafinger Database, as
explained in the section :ref:`bob.bio.vein.baselines`.
"""
protocol='Idiap_1_1_R'
"""The default protocol to use for tests

You may modify this at runtime by specifying the option ``--protocol`` on the
command-line of ``verify.py`` or using the keyword ``protocol`` on a
configuration file that is loaded **after** this configuration resource.
"""
