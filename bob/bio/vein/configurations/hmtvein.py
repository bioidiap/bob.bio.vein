#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""`SDUMLA-HMT Database`_ is a dataset for multimodal biometric recognition

The `SDUMLA-HMT Database`_ (*Shadong University's Machine Learning and
Applications Laboratory - Homologous Multimodal Traits Database*) consists of
face images from 7 view angles, finger vein images of 6 fingers, gait videos
from 6 view angles, iris images from an iris sensor, and fingerprint images
acquired with 5 different sensors. The database includes real multimodal data
from 106 individuals.

.. warning::

   This package **only** includes bindings for the fingervein data.


   It does not contain the raw data, which you must procure yourself either
   from the link above or contacting the authors.

"""


from ..database.mmcbnu6k import Database

_hmtvein_directory = "[YOUR_HMTVEIN_DIRECTORY]"
"""Value of ``~/.bob_bio_databases.txt`` for this database"""

database = Database(
    original_directory = _hmtvein_directory,
    original_extension = '.bmp',
    )
"""The :py:class:`bob.bio.base.database.BioDatabase` derivative with hmtvein
database settings

.. warning::

   This class only provides a programmatic interface to load data in an orderly
   manner, respecting usage protocols. It does **not** contain the raw
   datafiles. You should procure those yourself.

Notice that ``original_directory`` is set to ``[YOUR_HMTVEIN_DIRECTORY]``. You
must make sure to create ``${HOME}/.bob_bio_databases.txt`` setting this value
to the place where you actually installed the `SDUMLA-HMT Database`_, as
explained in the section :ref:`bob.bio.vein.baselines`.
"""

protocol = 'default'
"""The default protocol to use for tests

You may modify this at runtime by specifying the option ``--protocol`` on the
command-line of ``verify.py`` or using the keyword ``protocol`` on a
configuration file that is loaded **after** this configuration resource.
"""
