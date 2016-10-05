#!/usr/bin/env python
"""
`BioWave Test`_ Database. This is a small test database of wrist vein images that are acquired using BIOWATCH biometric sensor -- ``20`` subjects, both hands, only one session / attempt -- 5 images per hand -- all together ``20 x 2 x 5 = 200`` images. Images aren't mirrored / rotated, there is no such function at the moment.
You can find more information there - `BioWave Test`_ .

.. include:: links.rst
"""
from bob.bio.vein.database import BiowaveTestBioDatabase
"""Value of ``~/.bob_bio_databases.txt`` for this database"""


biowave_test_image_directory = "[YOUR_BIOWAVE_TEST_IMAGE_DIRECTORY]"

database = BiowaveTestBioDatabase(
    original_directory=biowave_test_image_directory,
    original_extension='.png',
)
"""The :py:class:`bob.bio.base.database.BioDatabase` derivative with BioWave Test settings

.. warning::

   This class only provides a programmatic interface to load data in an orderly
   manner, respecting usage protocols. It does **not** contain the raw
   datafiles. You should procure those yourself.

Notice that ``original_directory`` is set to ``[YOUR_BIOWAVE_TEST_IMAGE_DIRECTORY]``.
You must make sure to create ``${HOME}/.bob_bio_databases.txt`` setting this
value to the place where you actually installed the Verafinger Database, as
explained in the section :ref:`bob.bio.vein.baselines`.
"""

protocol = 'all'
"""The default protocol to use for tests

You may modify this at runtime by specifying the option ``--protocol`` on the
command-line of ``verify.py`` or using the keyword ``protocol`` on a
configuration file that is loaded **after** this configuration resource.
"""
