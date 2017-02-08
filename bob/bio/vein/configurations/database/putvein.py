#!/usr/bin/env python

from bob.bio.vein.database import PutveinBioDatabase

putvein_image_directory = "[YOUR_PUTVEIN_IMAGE_DIRECTORY]"

database = PutveinBioDatabase(
    original_directory=putvein_image_directory,
    original_extension='.bmp',
    protocol='wrist-R_4'
)
