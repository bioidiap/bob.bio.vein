#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tue 27 Sep 2016 16:49:15 CEST

"""BIOWAVE_TEST database implementation of bob.bio.db.BioDatabase interface.

It is an extension of an SQL-based database interface, which directly talks to
BIOWAVE_TEST database for verification experiments (good to use in bob.bio.base
framework).
"""

from .database import VeinBioFile
from bob.bio.base.database import BioDatabase, BioFile


class BiowaveTestBioDatabase(BioDatabase):
    """
    Implements verification API for querying BIOWAVE_TEST database.
    """

    def __init__(
            self,
            **kwargs
    ):
        # before it was also "name" in the init.
        #
        # the BioDatabase class is defined in:
        # bob.bio.db/bob/bio/db/database.py
        #
        # In this -- the high level implementation we call base class constructors to
        # open a session to the database. We use **kwargs so that we could pass
        # the arguments later, e.g. from the default database configuration.

        super(BiowaveTestBioDatabase, self).__init__(name='biowave_test', **kwargs)

        from bob.db.biowave_test.query import Database as LowLevelDatabase
        self.__db = LowLevelDatabase()

    def client_id_from_model_id(self, model_id, group='dev'):
        """Required as ``model_id != client_id`` on this database"""
        return self.__db.client_id_from_model_id(model_id)

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        return self.__db.model_ids(protocol=protocol, groups=groups)

    def objects(self, protocol=None, groups=None, purposes=None, model_ids=None, **kwargs):
        retval = self.__db.objects(protocol=protocol, groups=groups, purposes=purposes, model_ids=model_ids)
        return [VeinBioFile(client_id=f.client_id, path=f.path, file_id=f.id) for f in retval]

    # the methodes are derived from:
    # bob.bio.db/bob/bio/db/database.py
    # this means that methodes defined there need to have certain arguments, e.g.:
    # model_ids_with_protocol:
    #    groups;
    #    protocol;
    # objects:
    #    groups;
    #    protocol;
    #    purposes;
    #    model_ids;
    # If you have some other arguments to pass, use **kwargs, if your methods doesn't have some
    # arguments, just don't pass them (e.g. see the model_ids).
