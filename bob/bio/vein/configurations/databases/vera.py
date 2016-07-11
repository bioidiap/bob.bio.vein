#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.db.vera

from bob.bio.base.database import DatabaseBob

vera_directory = "/idiap/project/vera"

database = DatabaseBob(
    database = bob.db.vera.Database(
      original_directory = vera_directory,
      original_extension = ".png",
      ),
    name = 'vera',
    )
