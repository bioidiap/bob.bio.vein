#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import numpy

from bob.bio.base.database import BioFile, BioDatabase

from . import AnnotatedArray
from ..preprocessor.utils import poly_to_mask


class File(BioFile):
    """
    Implements extra properties of vein files for the 3D Fingervein database


    Parameters:

      f (object): Low-level file (or sample) object that is kept inside

    """

    def __init__(self, f):

        super(File, self).__init__(client_id=f.client_id, path=f.path,
            file_id=f.id)
        self.__f = f


    def load(self, *args, **kwargs):
        """(Overrides base method) Loads both image and mask"""

        image = super(File, self).load(*args, **kwargs)
        image = numpy.rot90(image, 1)

        if not self.__f.has_roi():
          return image

        else:
          roi = self.__f.roi()

          # calculates the 90 degrees clockwise rotated RoI points
          w, h = image.shape
          roi = [(w-x,y) for (y,x) in roi]

        return AnnotatedArray(image, metadata=dict(roi=roi))


class Database(BioDatabase):
    """
    Implements verification API for querying the 3D Fingervein database.
    """

    def __init__(self, **kwargs):

        super(Database, self).__init__(name='thufvdt', **kwargs)
        from bob.db.thufvdt.query import Database as LowLevelDatabase
        self.__db = LowLevelDatabase()

        self.low_level_group_names = ('train', 'dev', 'eval')
        self.high_level_group_names = ('world', 'dev', 'eval')


    def groups(self):

        return self.convert_names_to_highlevel(self.__db.groups(),
            self.low_level_group_names, self.high_level_group_names)


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

        return [File(f) for f in retval]


    def annotations(self, file):
        return None
