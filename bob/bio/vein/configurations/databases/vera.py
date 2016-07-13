#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.db.vera

from bob.bio.base.database import DatabaseBob

directory = "/idiap/project/vera"
extension = ".png"

database = DatabaseBob(
    database = bob.db.vera.Database(
      original_directory = directory,
      original_extension = extension,
      ),
    name = 'vera',
    )
