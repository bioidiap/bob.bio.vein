#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""The Hong Kong Polytechnic University Finger Image Database

The `Hong Kong Polytechnic University Finger Image Database`_  consists of
simultaneously acquired finger vein and finger surface texture images from the
male and female volunteers. The currently available database has 6264 images
from the 156 subjects. Details of this database were published in [KV12]_.

You can download the raw data of the `Hong Kong Polytechnic University Finger Image Database`_ by following the link.
"""


from ..database.hkpu import Database

_hkpu_directory = "[YOUR_HKPU_DIRECTORY]"
"""Value of ``~/.bob_bio_databases.txt`` for this database"""

database = Database(
    original_directory = _hkpu_directory,
    original_extension = '.bmp',
    )
"""The :py:class:`bob.bio.base.database.BioDatabase` derivative with hkpu
database settings

.. warning::

   This class only provides a programmatic interface to load data in an orderly
   manner, respecting usage protocols. It does **not** contain the raw
   datafiles. You should procure those yourself.

Notice that ``original_directory`` is set to ``[YOUR_HKPU_DIRECTORY]``. You
must make sure to create ``${HOME}/.bob_bio_databases.txt`` setting this value
to the place where you actually installed the `Hong Kong Polytechnic University
Finger Image Database`_, as explained in the section
:ref:`bob.bio.vein.baselines`.
"""

protocol = 'A'
"""The default protocol to use for tests

You may modify this at runtime by specifying the option ``--protocol`` on the
command-line of ``verify.py`` or using the keyword ``protocol`` on a
configuration file that is loaded **after** this configuration resource.
"""


from ..preprocessor import NoCrop, WatershedMask, HuangNormalization, \
    NoFilter, Preprocessor

from os.path import join as _join
from pkg_resources import resource_filename as _filename
_model = _filename(__name__, _join('data', 'hkpu.hdf5'))

preprocessor = Preprocessor(
    crop=NoCrop(),
    mask=WatershedMask(
      model=_model,
      foreground_threshold=0.6,
      background_threshold=0.2,
      ),
    normalize=HuangNormalization(),
    filter=NoFilter(),
    )
"""Preprocessing using morphology and watershedding

Settings are optimised for the image quality of HKPU.
"""