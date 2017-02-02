# vim: set fileencoding=utf-8 :

"""
  PUTVEIN database implementation of bob.bio.db.BioDatabase interface.
  It is an extension of low level database interface, which directly talks to 
  PUTVEIN database for verification experiments (good to use in bob.bio.base 
  framework).
"""

from bob.bio.base.database import BioFile, BioDatabase


class File(BioFile):
  """
  Implements extra properties of vein files for the BIOWAVE V1 database


  Parameters:

    f (object): Low-level file (or sample) object that is kept inside

  """

  def __init__(self, f):

    super(File, self).__init__(client_id=f.client_id)
    # normally "file_id=f.id", but this database dosn't have any;
    #
    # normally "path=f.path", but closest there is the f.make_path() that returns
    # also th eextension.
    self.__f = f



class PutveinBioDatabase(BioDatabase):
    """
    Implements verification API for querying PUTVEIN database.
    """

    def __init__(
            self,
            **kwargs
    ):

        super(PutveinBioDatabase, self).__init__(name='putvein', **kwargs)

        from bob.db.putvein.query import Database as LowLevelDatabase
        self.__db = LowLevelDatabase()

    #def client_id_from_model_id(self, model_id, group='dev'):
    #    """Required as ``model_id != client_id`` on this database"""
    #    return self.__db.client_id_from_model_id(model_id)

    #def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
    #    return self.__db.model_ids(protocol=protocol, groups=groups)

    def objects(self, protocol=None, groups=None, purposes=None, model_ids=None, kinds=None, **kwargs):
        retval = self.__db.objects(protocol=protocol, groups=groups, purposes=purposes, kinds=kinds)
        return [File(f) for f in retval]

