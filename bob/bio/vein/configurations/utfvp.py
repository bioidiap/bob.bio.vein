#!/usr/bin/env python

"""`UTFVP`_ is a database for biometric fingervein recognition

The University of Twente Finger Vascular Pattern (UTFVP) Database is made
publically available in order to support and stimulate research efforts in the
area of developing, testing and evaluating algorithms for vascular patter
recognition. The University of Twente, Enschede, The Netherlands (henceforth,
UT) owns copyright of and serves as the source for the UTFVP database, which is
now distributed to any research group approved by the UTFVP principal
investigator. The reference citation is [TV13]_.

You can download the raw data of the `UTFVP`_ database by following the link.

.. include:: links.rst
"""

from bob.bio.vein.database import UtfvpBioDatabase

utfvp_directory = "[YOUR_UTFVP_DIRECTORY]"
"""Value of ``~/.bob_bio_databases.txt`` for this database"""

database = UtfvpBioDatabase(original_directory = utfvp_directory)
"""The :py:class:`bob.bio.base.database.BioDatabase` derivative with UTFVP settings

.. warning::

   This class only provides a programmatic interface to load data in an orderly
   manner, respecting usage protocols. It does **not** contain the raw
   datafiles. You should procure those yourself.

Notice that ``original_directory`` is set to ``[YOUR_UTFVP_DIRECTORY]``.
You must make sure to create ``${HOME}/.bob_bio_databases.txt`` setting this
value to the place where you actually installed the Verafinger Database, as
explained in the section :ref:`bob.bio.vein.baselines`.
"""

protocol = 'nom'
"""The default protocol to use for tests

You may modify this at runtime by specifying the option ``--protocol`` on the
command-line of ``verify.py`` or using the keyword ``protocol`` on a
configuration file that is loaded **after** this configuration resource.
"""
