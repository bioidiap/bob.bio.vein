#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Teodors Eglitis <teodors.eglitis@idiap.ch>
# Wed 20 Jul

"""
  BIOWAVE_V1 database implementation of bob.bio.db.BioDatabase interface.
  It is an extension of an SQL-based database interface, which directly talks to BIOWAVE_V1 database for
  verification experiments (good to use in bob.bio.base framework).
"""

from .database import VeinBioFile
from bob.bio.base.database import BioDatabase, BioFile


class BiowaveV1BioDatabase(BioDatabase):
    """
    Implements verification API for querying BIOWAVE_V1 database.
    """

    def __init__(
            self,
            **kwargs
    ):

        super(BiowaveV1BioDatabase, self).__init__(name='biowave_v1', **kwargs)

        from bob.db.biowave_v1.query import Database as LowLevelDatabase
        self.__db = LowLevelDatabase()

    def client_id_from_model_id(self, model_id, group='dev'):
        """Required as ``model_id != client_id`` on this database"""
        return self.__db.client_id_from_model_id(model_id)

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        return self.__db.model_ids(protocol=protocol, groups=groups)

    def objects(self, protocol=None, groups=None, purposes=None, model_ids=None, **kwargs):
        retval = self.__db.objects(protocol=protocol, groups=groups, purposes=purposes, model_ids=model_ids, sessions=None, attempts=None, im_numbers=None)
        return [VeinBioFile(client_id=f.client_id, path=f.path, file_id=f.id) for f in retval]

