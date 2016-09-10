#!/usr/bin/env python

from bob.bio.vein.database import UtfvpBioDatabase

utfvp_directory = "[YOUR_UTFVP_DIRECTORY]"

database = UtfvpBioDatabase(
    original_directory = utfvp_directory,
    extension = ".png",
    )    

