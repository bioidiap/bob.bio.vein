#!/usr/bin/env python
"""
Putvein Database.

**This is default** ``bob.db.putvein`` **database configuration.
More information -** :py:class:`bob.bio.vein.database.PutveinV1BioDatabase`.

.. include:: links.rst
"""


from bob.bio.vein.database import PutveinBioDatabase

putvein_image_directory = "[YOUR_PUTVEIN_IMAGE_DIRECTORY]"

"""Value of ``~/.bob_bio_databases.txt`` for this database"""

database = PutveinBioDatabase(
           original_directory=putvein_image_directory,
           original_extension='.bmp',
           )
"""The :py:class:`bob.bio.base.database.BioDatabase` derivative with
PUTVEIN settings

.. warning::

   This class only provides a programmatic interface to load data in an orderly
   manner, respecting usage protocols. It does **not** contain the raw
   datafiles. You should procure those yourself.

Notice that ``original_directory`` is set to ``[YOUR_PUTVEIN_IMAGE_DIRECTORY]``.
You must make sure to create ``${HOME}/.bob_bio_databases.txt`` setting this
value to the place where you actually installed the Verafinger Database, as
explained in the section :ref:`bob.bio.vein.baselines`.
"""
protocol = 'wrist-R_4'
"""The default protocol to use for tests

You may modify this at runtime by specifying the option ``--protocol`` on the
command-line of ``verify.py`` or using the keyword ``protocol`` on a
configuration file that is loaded **after** this configuration resource.
"""
