#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Tsinghua University Finger Vein and Finger Dorsal Texture Database

The `Tsinghua University Finger Vein and Finger Dorsal Texture Database`_
contains data of finger vein and finger dorsal texture from 610 different
subjects. Details of part 1 of this dataset were originally published in
[YYL09]_. A number of studies were published based on this subset of the FVFDT
database. Authors of this dataset list [YRL11]_, [YHL12]_, [YMLL13]_ and
[YHZL13]_.

Subsequently, the dataset was augmented (part 2) to include more subjects and a
study based on fixed regions of interested [YMZL14]_. The raw images from which
the regions of interest in part 2 were extracted were then published to become
part 3 of this dataset, without further references.

You can download the raw data of the `Tsinghua University Finger Vein and
Finger Dorsal Texture Database`_ by following the link.
"""


from ..database.thufvdt import Database

_thufvdt_directory = "[YOUR_THUFVDT_DIRECTORY]"
"""Value of ``~/.bob_bio_databases.txt`` for this database"""

database = Database(
    original_directory = _thufvdt_directory,
    original_extension = '.bmp',
    )
"""The :py:class:`bob.bio.base.database.BioDatabase` derivative with thufvdt
database settings

.. warning::

   This class only provides a programmatic interface to load data in an orderly
   manner, respecting usage protocols. It does **not** contain the raw
   datafiles. You should procure those yourself.

Notice that ``original_directory`` is set to ``[YOUR_THUFVDT_DIRECTORY]``.
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


from ..preprocessor import NoCrop, WatershedMask, HuangNormalization, \
    NoFilter, Preprocessor

from os.path import join as _join
from pkg_resources import resource_filename as _filename
_model = _filename(__name__, _join('data', 'thufvdt.hdf5'))

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

Settings are optimised for the image quality of VERA-Fingervein.
"""
