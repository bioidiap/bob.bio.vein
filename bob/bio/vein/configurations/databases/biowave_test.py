#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from bob.bio.db import BiowaveTest

biowave_test_directory = "/idiap/user/onikisins/Databases/BIOWAVE/Database_25_04_2016"

database = BiowaveTest(
      original_directory = biowave_test_directory,
      original_extension = ".png")
