#!/usr/bin/env python

from bob.bio.vein.database import BiowaveV1BioDatabase

biowave_v1_image_directory = "[YOUR_BIOWAVE_V1_IMAGE_DIRECTORY]"

database = BiowaveV1BioDatabase(
    original_directory=biowave_v1_image_directory,
    original_extension='.png',
    protocol='Idiap_1_1_R'
)
