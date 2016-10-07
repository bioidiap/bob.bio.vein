#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tue 27 Sep 2016 16:48:57 CEST


from .database import VeinBioFile
from bob.bio.base.database import BioDatabase


class VerafingerBioDatabase(BioDatabase):
    """
    Implements verification API for querying Vera Fingervein database.
    """

    def __init__(self, **kwargs):

        super(VerafingerBioDatabase, self).__init__(name='verafinger',
            **kwargs)
        from bob.db.verafinger.query import Database as LowLevelDatabase
        self.__db = LowLevelDatabase()

        self.low_level_group_names = ('train', 'dev')
        self.high_level_group_names = ('world', 'dev')

    def groups(self):

        return self.convert_names_to_highlevel(self.__db.groups(),
            self.low_level_group_names, self.high_level_group_names)

    def client_id_from_model_id(self, model_id, group='dev'):
        """Required as ``model_id != client_id`` on this database"""

        return self.__db.finger_name_from_model_id(model_id)

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):

        groups = self.convert_names_to_lowlevel(groups,
            self.low_level_group_names, self.high_level_group_names)
        return self.__db.model_ids(groups=groups, protocol=protocol)

    def objects(self, groups=None, protocol=None, purposes=None,
        model_ids=None, **kwargs):

        groups = self.convert_names_to_lowlevel(groups,
            self.low_level_group_names, self.high_level_group_names)
        retval = self.__db.objects(groups=groups, protocol=protocol,
            purposes=purposes, model_ids=model_ids, **kwargs)
        return [VeinBioFile(client_id=f.model_id, path=f.path, file_id=f.id) for f in retval]
