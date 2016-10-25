#!/usr/bin/env python
"""
`BioWave V1`_ Database. This is a database of wrist vein images that are acquired using BIOWATCH biometric sensor. For each subject of the database there are 3 session images (sessions were held at least 24 hours apart). Each session consists of 5 attempts, in each attempt 5 images were acquired, meaning, that there are ``3 sessions x 5 attempts x 5 images = 75 images`` images per each person's hand, ``75 x 2 images`` per person.
Images were previously manually evaluated, and if any of the ``75`` one hand's images were unusable (too blurred, veins couldn't be seen, etc), than all hand data were discarded. That is way some persons has only 1 hand's images in the database.

Statistics of the data - in total 111 hands;

1) Users with both hands images - 53
2) Users with only R hand images - 4
3) Users with only L hand images - 1

Database have 6 protocols, as described there - `BioWave V1`_ .

**High level implementation**


In addition to the methods implemented in ``bob.bio.db.BioDatabase`` in the ``BIOWAVE_V1`` high level implementation there also are 2 extra flags that user can use:

- ``annotated_images`` -- by default this is set to ``False``. If set True, only subset of protocol images are returned - those images, that have annotations (``8% of all images``).
- ``extra_annotation_information`` = By default this is set to ``False``.

If set to ``True``, this automatically sets the flag ``annotated_images``
as ``True``, and now database interface returns not the original image,
but an :py:class:`dict` object containing fields ``image``,
``roi_annotations``, ``vein_annotations``. In this case ``preprocessor``
needs to act accordingly.
    
**Configurations / Entry points**

There are 3 ways how database can be used, for each of them there is a separate entry point as well as database default configuration file (hint - each configuration uses different ``BiowaveV1BioDatabase`` flags). To simply run the verification experiments, simply point to one of the entry points:

- ``biowave_v1`` -- all database data (unchanged operation);
- ``biowave_v1_a`` -- (``a`` denotes ``annotations``)  only annotated files. E.g., by default, for each *hand* there are ``3x25`` images (3 sessions, 25 images in each), then the number of annotated images for each hand are ``3x2`` images;
- ``biowave_v1_e`` -- (``a`` denotes ``extra``) this time also only data for images with annotations are returned, but this time not only original image is returned, but interface returns :py:class:`dict` object containing original image, ROI annotations and vein annotations; Note that when this database configuration is used, the ``preprocessor`` class needs to act accordingly. Currently there is implemented class :py:class:`bob.bio.vein.preprocessors.ConstructAnnotations` that works with such objects and can return constructed annotation image.

**This** ``bob.db.biowave_v1`` **database configuration that makes database to return extra information dor files (original images) that have annotations (for each hand in database ``3x2`` images).** More information - :py:class:`bob.bio.vein.database.BiowaveV1BioDatabase()`.

.. include:: links.rst
"""
from bob.bio.vein.database import BiowaveV1BioDatabase

biowave_v1_image_directory = "[YOUR_BIOWAVE_V1_IMAGE_DIRECTORY]"
"""Value of ``~/.bob_bio_databases.txt`` for this database"""

database = BiowaveV1BioDatabase(
           original_directory=biowave_v1_image_directory,
           original_extension = '.png',
           annotated_images = True,
           extra_annotation_information = True
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
