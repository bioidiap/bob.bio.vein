#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""`MMCBNU 6000 Database`_ is a dataset for biometric fingervein recognition

The `MMCBNU 6000 database`_ contains fingervein images collected by the
Multimedia Laboratory from Chonbuk Nation University in South Korea, from
January 29th to February 20th 2013. Notice this package does not contain the
raw data files from this dataset, which need to be obtained through the link
provided above.

You can download the raw data of the `MMCBNU 6000 Database`_ by following
the link.
"""


from ..database.mmcbnu6k import Database

_mmcbnu6k_directory = "[YOUR_MMCBNU6K_DIRECTORY]"
"""Value of ``~/.bob_bio_databases.txt`` for this database"""

database = Database(
    original_directory = _mmcbnu6k_directory,
    original_extension = '.bmp',
    )
"""The :py:class:`bob.bio.base.database.BioDatabase` derivative with mmcbnu6k
database settings

.. warning::

   This class only provides a programmatic interface to load data in an orderly
   manner, respecting usage protocols. It does **not** contain the raw
   datafiles. You should procure those yourself.

Notice that ``original_directory`` is set to ``[YOUR_MMCBNU6K_DIRECTORY]``. You
must make sure to create ``${HOME}/.bob_bio_databases.txt`` setting this value
to the place where you actually installed the `MMCBNU 6000 Database`_, as
explained in the section :ref:`bob.bio.vein.baselines`.
"""

protocol = 'default'
"""The default protocol to use for tests

You may modify this at runtime by specifying the option ``--protocol`` on the
command-line of ``verify.py`` or using the keyword ``protocol`` on a
configuration file that is loaded **after** this configuration resource.
"""
