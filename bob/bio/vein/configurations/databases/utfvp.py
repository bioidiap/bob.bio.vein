#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.db.utfvp

from bob.bio.base.database import DatabaseBob

directory = "/idiap/resource/database/UTFVP/data/"
extension = ".png"

database = DatabaseBob(
    database = bob.db.utfvp.Database(
      original_directory = directory,
      original_extension = extension,
      ),
    name = 'utfvp',
    )
