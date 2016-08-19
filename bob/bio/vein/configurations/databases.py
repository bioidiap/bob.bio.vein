#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.bio.db

verafinger = bob.bio.db.VerafingerBioDatabase(
    original_directory = '/idiap/project/vera/databases/VERA-fingervein',
    original_extension = '.png',
    )

utfvp = bob.bio.db.UtfvpBioDatabase(
    original_directory = '/idiap/resource/database/UTFVP/data',
    extension = ".png",
    )
