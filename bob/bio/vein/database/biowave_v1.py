
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
from bob.bio.base.database import BioDatabase


class BiowaveV1BioFile(VeinBioFile):
    def __init__(self, low_level_file, client_id, path, file_id, protocol):
        """
        Initializes this File object with an File equivalent from the low level
        implementation. The load function depends on the low level database 
        protocol.
        If the protocol name ends with letter ``na`` (the ``n`` stands for ``Not 
        centered`` and the ``a`` -- for ``annotations``) than the not centered 
        / centered annotations are loaded respectively instead of the original
        image.
        """
        super(BiowaveV1BioFile, self).__init__(client_id=client_id, path=path, file_id=file_id)
        self.protocol = protocol
        self.low_level_file = low_level_file

    def load(self, directory=None, extension='.png'):
      if self.protocol.endswith("na"):
          return self.low_level_file.construct_vein_image(self, directory=directory, center=False)
      elif self.protocol.endswith("ca"):
          return self.low_level_file.construct_vein_image(self, directory=directory, center=True)
      else:
        super(BiowaveV1BioFile, self).load(directory=directory, extension=extension)


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
        return [BiowaveV1BioFile(f, client_id=f.client_id, path=f.path, file_id=f.id, protocol=protocol) for f in retval]

