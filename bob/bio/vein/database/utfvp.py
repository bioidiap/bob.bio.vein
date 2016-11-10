#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Fri 04 Nov 2016 14:46:53 CET

from bob.bio.base.database import BioFile, BioDatabase


class File(BioFile):
    """
    Implements extra properties of vein files for the UTFVP Fingervein database


    Parameters:

      f (object): Low-level file (or sample) object that is kept inside

    """

    def __init__(self, f):

        super(File, self).__init__(client_id=f.client_id, path=f.path,
            file_id=f.id)
        self.__f = f


class Database(BioDatabase):
    """
    Implements verification API for querying UTFVP Fingervein database.
    """

    def __init__(self, **kwargs):

        super(Database, self).__init__(name='utfvp', **kwargs)
        from bob.db.utfvp.query import Database as LowLevelDatabase
        self.__db = LowLevelDatabase()


    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        protocol = protocol if protocol is not None else self.protocol
        return self.__db.model_ids(groups=groups, protocol=protocol)


    def objects(self, groups=None, protocol=None, purposes=None,
        model_ids=None, **kwargs):

        retval = self.__db.objects(groups=groups, protocol=protocol,
            purposes=purposes, model_ids=model_ids, **kwargs)

        return [File(f) for f in retval]
