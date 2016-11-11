#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


from .database import VeinBioFile
from bob.bio.base.database import BioDatabase


class BiowaveV1BioFile(VeinBioFile):
    def __init__(self,
                 low_level_file,
                 client_id, path,
                 file_id, protocol,
                 extra_annotation_information):
        """
        Initializes this File object with an File equivalent from the low level
        implementation. The load function depends on the low level database
        protocol.
        """
        super(BiowaveV1BioFile, self).__init__(client_id=client_id,
                                               path=path,
                                               file_id=file_id)
        self.protocol = protocol
        self.low_level_file = low_level_file
        self.extra_annotation_information = extra_annotation_information

    def load(self, directory=None, extension='.png'):
      if self.extra_annotation_information == True:
        image = self.low_level_file.load(directory=directory,
                                         extension=extension)
        roi_annotations = self.low_level_file.\
            roi_annotations(directory=directory)
        vein_annotations = self.low_level_file.\
            vein_annotations(directory=directory)
        alignment_annotations = self.low_level_file.\
            alignment_annotations(directory=directory)
        return {"image": image,
                "roi_annotations": roi_annotations,
                "vein_annotations": vein_annotations,
                "alignment_annotations": alignment_annotations}
      else:
        return self.low_level_file.load(directory=directory,
                                        extension=extension)


class BiowaveV1BioDatabase(BioDatabase):
    """
    `BioWave V1`_ Database. This is a database of wrist vein images that are
    acquired using BIOWATCH biometric sensor. For each subject of the database
    there are 3 session images (sessions were held at least 24 hours apart).
    Each session consists of 5 attempts, in each attempt 5 images were
    acquired, meaning, that there are ``3 sessions x 5 attempts x 5 images =
    75 images`` images per each person's hand, ``75 x 2 images`` per person.

    Images were previously manually evaluated, and if any of the ``75`` one
    hand's images were unusable (too blurred, veins couldn't be seen, etc),
    than all hand data were discarded. That is way some persons has only 1
    hand's images in the database.

    Statistics of the data - in total 111 hands;

    1) Users with both hands images - 53
    2) Users with only R hand images - 4
    3) Users with only L hand images - 1

    Database have 6 protocols, as described there - `BioWave V1`_ .

    **High level implementation**


    In addition to the methods implemented in ``bob.bio.db.BioDatabase`` in the
    ``BIOWAVE_V1`` high level implementation there also are 2 extra flags that
    user can use:

    - ``annotated_images`` -- by default this is set to ``False``. If set True,
      only subset of protocol images are returned - those images, that have
      annotations (``8% of all images``).
    - ``extra_annotation_information`` = By default this is set to ``False``.

    If set to ``True``, this automatically sets the flag ``annotated_images``
    as ``True``, and now database interface returns not the original image,
    but an :py:class:`dict` object containing fields ``image``,
    ``roi_annotations``, ``vein_annotations``. In this case ``preprocessor``
    needs to act accordingly.

    **Configurations / Entry points**

    There are 3 ways how database can be used, for each of them there is a
    separate entry point as well as database default configuration file (hint -
    each configuration uses different ``BiowaveV1BioDatabase`` flags). To
    simply run the verification experiments, simply point to one of the entry
    points:

    - ``biowave_v1`` -- all database data (unchanged operation);
    - ``biowave_v1_a`` -- (``a`` denotes ``annotations``)  only annotated
      files. E.g., by default, for each *hand* there are ``3x25`` images (3
      sessions, 25 images in each), then the number of annotated images for
      each hand are ``3x2`` images;
    - ``biowave_v1_e`` -- (``a`` denotes ``extra``) this time also only data
      for images with annotations are returned, but this time not only original
      image is returned, but interface returns :py:class:`dict` object
      containing original image, ROI annotations, vein annotations and
      alignment annotations; Note that when this database configuration is
      used, the ``preprocessor`` class needs to act accordingly. Currently
      there is implemented class
      :py:class:`bob.bio.vein.preprocessors.ConstructAnnotations` that works
      with such objects and can return constructed annotation image.

    .. include:: links.rst
    """
    def __init__(
            self,
            annotated_images = False,
            extra_annotation_information = False,
            **kwargs
    ):

        super(BiowaveV1BioDatabase, self).__init__(name='biowave_v1', **kwargs)

        from bob.db.biowave_v1.query import Database as LowLevelDatabase
        self.__db = LowLevelDatabase()
        self.extra_annotation_information = extra_annotation_information
        if extra_annotation_information == True:
          self.annotated_images = True
        else:
          self.annotated_images = annotated_images

    def client_id_from_model_id(self, model_id, group='dev'):
        """Required as ``model_id != client_id`` on this database"""
        return self.__db.client_id_from_model_id(model_id)

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        return self.__db.model_ids(protocol=protocol,
                                   groups=groups,
                                   annotated_images=self.annotated_images,
                                   imagedir = self.original_directory)

    def objects(self, protocol=None,
                groups=None,
                purposes=None,
                model_ids=None,
                **kwargs):

        retval = self.__db.objects(protocol=protocol,
                                   groups=groups,
                                   purposes=purposes,
                                   model_ids=model_ids,
                                   sessions=None,
                                   attempts=None,
                                   im_numbers=None,
                                   annotated_images=self.annotated_images,
                                   imagedir = self.original_directory)
        return [BiowaveV1BioFile(f,
                                 client_id=f.client_id,
                                 path=f.path,
                                 file_id=f.id,
                                 protocol=protocol,
                                 extra_annotation_information=
                                   self.extra_annotation_information)
                for f in retval]
