#!/usr/bin/env python

from bob.bio.vein.database import BiowaveTestBioDatabase

biowave_test_image_directory = "[YOUR_BIOWAVE_TEST_IMAGE_DIRECTORY]"

database = BiowaveTestBioDatabase(
    original_directory=biowave_test_image_directory,
    original_extension='.png',
    protocol='all'
)
