#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.db.utfvp

from bob.bio.base.database import DatabaseBob

utfvp_directory = "/idiap/resource/database/UTFVP/data/"

database = DatabaseBob(
    database = bob.db.utfvp.Database(
      original_directory = utfvp_directory,
      original_extension = ".png"
      ),
    name = 'utfvp',
    )
