#!/usr/bin/env python
"""
`BioWave Test`_ Database. This is a small test database of wrist vein images that are acquired using BIOWATCH biometric sensor -- ``20`` subjects, both hands, only one session / attempt -- 5 images per hand -- all together ``20 x 2 x 5 = 200`` images. Images aren't mirrored / rotated, there is no such function at the moment.
You can find more information there - `BioWave Test`_ .

.. include:: links.rst
"""
from bob.bio.vein.database import BiowaveTestBioDatabase

biowave_test_image_directory = "[YOUR_BIOWAVE_TEST_IMAGE_DIRECTORY]"

database = BiowaveTestBioDatabase(
    original_directory=biowave_test_image_directory,
    original_extension='.png',
    protocol='all'
)
