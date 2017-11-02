#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Tsinghua University Finger Vein and Finger Dorsal Texture Database

The `Tsinghua University Finger Vein and Finger Dorsal Texture Database`_
contains data of finger vein and finger dorsal texture from 610 different
subjects.

You can download the raw data of the `Tsinghua University Finger Vein and
Finger Dorsal Texture Database`_ by following the link.
"""


from ..database.thu_fvfdt import Database

_thu_fvfdt_directory = "[YOUR_THU_FVFDT_DIRECTORY]"
"""Value of ``~/.bob_bio_databases.txt`` for this database"""

database = Database(
    original_directory = _thu_fvfdt_directory,
    original_extension = '.bmp',
    )
"""The :py:class:`bob.bio.base.database.BioDatabase` derivative with thu_fvfdt
database settings

.. warning::

   This class only provides a programmatic interface to load data in an orderly
   manner, respecting usage protocols. It does **not** contain the raw
   datafiles. You should procure those yourself.

Notice that ``original_directory`` is set to ``[YOUR_THU_FVFDT_DIRECTORY]``.
You must make sure to create ``${HOME}/.bob_bio_databases.txt`` setting this
value to the place where you actually installed the `Tsinghua University Finger
Vein and Finger Dorsal Texture Database`_, as explained in the section
:ref:`bob.bio.vein.baselines`.
"""

protocol = 'p3'
"""The default protocol to use for tests

You may modify this at runtime by specifying the option ``--protocol`` on the
command-line of ``verify.py`` or using the keyword ``protocol`` on a
configuration file that is loaded **after** this configuration resource.
"""
