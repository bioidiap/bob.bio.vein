#!/usr/bin/env python

from bob.bio.vein.database import VerafingerBioDatabase

vera_finger_directory = "[YOUR_VERAFINGER_DIRECTORY]"

database = VerafingerBioDatabase(
    original_directory = vera_finger_directory,
    original_extension = '.png',
    )

