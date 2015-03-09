#!/usr/bin/env python

import bob.db.vera
import facereclib

vera_directory = "/idiap/project/vera"

database = facereclib.databases.DatabaseBob(
    database = bob.db.vera.Database(
      original_directory = vera_directory,
      original_extension = ".png",    
    ),
    name = 'vera',
)