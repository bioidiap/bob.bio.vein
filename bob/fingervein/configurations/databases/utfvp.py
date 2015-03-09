#!/usr/bin/env python

import bob.db.utfvp
import facereclib

utfvp_directory = "/idiap/resource/database/UTFVP/data/"

database = facereclib.databases.DatabaseBob(
    database = bob.db.utfvp.Database(
      original_directory = utfvp_directory,
      original_extension = ".png"
    ),
    name = 'utfvp',    
)
