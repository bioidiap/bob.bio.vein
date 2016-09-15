#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from .database import VeinBioFile
from bob.bio.base.database import BioDatabase, BioFile


class UtfvpBioDatabase(BioDatabase):
    """
    Implements verification API for querying UTFVP Fingervein database.
    """

    def __init__(self, **kwargs):

        super(UtfvpBioDatabase, self).__init__(name='utfvp',
            **kwargs)
        from bob.db.utfvp.query import Database as LowLevelDatabase
        self.__db = LowLevelDatabase()

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        protocol = protocol if protocol is not None else self.protocol
        return self.__db.model_ids(groups=groups, protocol=protocol)

    def objects(self, groups=None, protocol=None, purposes=None,
        model_ids=None, **kwargs):

        retval = self.__db.objects(groups=groups, protocol=protocol,
            purposes=purposes, model_ids=model_ids, **kwargs)
        return [VeinBioFile(client_id=f.client_id, path=f.path, file_id=f.file_id) for f in retval]
