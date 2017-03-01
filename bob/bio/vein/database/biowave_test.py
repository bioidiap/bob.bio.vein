#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Teodors Eglitis <teodors.eglitis@idiap.ch>
# Wed 20 Jul

"""
  BIOWAVE_TEST database implementation of bob.bio.db.BioDatabase interface.
  It is an extension of an SQL-based database interface, which directly talks to BIOWAVE_TEST database for
  verification experiments (good to use in bob.bio.base framework).
"""

from bob.bio.base.database import BioFile, BioDatabase


class File(BioFile):
  """
  Implements extra properties of vein files for the BIOWAVE V1 database


  Parameters:

    f (object): Low-level file (or sample) object that is kept inside

  """

  def __init__(self, f):

    super(File, self).__init__(client_id=f.client_id, path=f.path, file_id=f.id)
    self.__f = f



class BiowaveTestBioDatabase(BioDatabase):
    """
    Implements verification API for querying BIOWAVE_TEST database.
    """

    def __init__(
            self,
            **kwargs
    ):

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
        return [File(f) for f in retval]

    def annotations(self, file):
        return None

