#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tue 07 Nov 2017 12:29:26 CET


import numpy

from bob.bio.base.database import BioFile, BioDatabase

from . import AnnotatedArray
from ..preprocessor.utils import poly_to_mask


class File(BioFile):
    """
    Implements extra properties of vein files for the MMCBNU_6000 Fingervein
    database


    Parameters:

      f (object): Low-level file (or sample) object that is kept inside

    """

    def __init__(self, f):

        super(File, self).__init__(client_id=f.finger.unique_name, path=f.path,
            file_id=f.id)
        self._f = f


    def load(self, *args, **kwargs):
        """(Overrides base method) Loads both image and mask"""

        image = self._f.load(*args, **kwargs)

        if not self._f.has_roi():
          return image

        else:
          roi = self._f.roi()

        return AnnotatedArray(image, metadata=dict(roi=roi))


class Database(BioDatabase):
    """
    Implements verification API for querying the 3D Fingervein database.
    """

    def __init__(self, **kwargs):

        super(Database, self).__init__(name='mmcbnu6k', **kwargs)
        from bob.db.mmcbnu6k.query import Database as LowLevelDatabase
        self._db = LowLevelDatabase()

        self.low_level_group_names = ('train', 'dev', 'eval')
        self.high_level_group_names = ('world', 'dev', 'eval')


    def groups(self):

        return self.convert_names_to_highlevel(self._db.groups(),
            self.low_level_group_names, self.high_level_group_names)


    def client_id_from_model_id(self, model_id, group='dev'):
        """Required as ``model_id != client_id`` on this database"""

        return self._db.finger_name_from_model_id(model_id)


    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):

        groups = self.convert_names_to_lowlevel(groups,
            self.low_level_group_names, self.high_level_group_names)
        return self._db.model_ids(groups=groups, protocol=protocol)


    def objects(self, groups=None, protocol=None, purposes=None,
        model_ids=None, **kwargs):

        groups = self.convert_names_to_lowlevel(groups,
            self.low_level_group_names, self.high_level_group_names)
        retval = self._db.objects(groups=groups, protocol=protocol,
            purposes=purposes, model_ids=model_ids, **kwargs)

        return [File(f) for f in retval]


    def annotations(self, file):
        return None
