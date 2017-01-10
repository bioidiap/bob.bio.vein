#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


from bob.bio.base.extractor import Extractor
import bob.io.base
import numpy as np

class ExtNone( Extractor ):
    """
    An empty extracror class that returns an input image
    """

    def __init__(self):

        Extractor.__init__( self)

    def __call__( self, data):
        """
        Returns the input data (does absolutely nothing)
        """
        return data

#  def write_feature(self, feature, feature_file):
#    """Writes the given *extracted* feature to a file with the given name.
#    In this base class implementation, we simply use :py:func:`bob.bio.base.save` for that.
#    If you have a different format, please overwrite this function.
#
#    **Parameters:**
#
#    feature : object
#      The extracted feature, i.e., what is returned from `__call__`.
#
#    feature_file : str or :py:class:`bob.io.base.HDF5File`
#      The file open for writing, or the name of the file to write.
#    """
#    utils.save(feature, feature_file)

    def write_feature(self, data, file_name):
        """
        Writes the given data (that has been generated using the __call__
        function of this class) to file.
        This method overwrites the write_data() method of the Preprocessor
        class.

        **Parameters:**

        data :
            data returned by the __call__ method of the class.

        file_name : :py:class:`str`
            name of the file.
        """

        f = bob.io.base.HDF5File(file_name, 'w')
        f.set('image', data[0])
        f.set('alignment annotations', data[1])
        del f

#  def read_feature(self, feature_file):
#    """Reads the *extracted* feature from file.
#    In this base class implementation, it uses :py:func:`bob.bio.base.load` to do that.
#    If you have different format, please overwrite this function.
#
#    **Parameters:**
#
#    feature_file : str or :py:class:`bob.io.base.HDF5File`
#      The file open for reading or the name of the file to read from.
#
#    **Returns:**
#
#    feature : object (usually :py:class:`numpy.ndarray`)
#      The feature read from file.
#    """
#    return utils.load(feature_file)

    def read_feature(self, file_name):
        """
        Reads the preprocessed data from file.
        his method overwrites the read_data() method of the Preprocessor class.

        **Parameters:**

        file_name : :py:class:`str`
            name of the file.

        **Returns:**

        output : ``tuple``
            a tuple containing the image and it's alignment annotations
        """
        f = bob.io.base.HDF5File(file_name, 'r')
        image = f.read('image')
        alignment_annotations = f.read('alignment annotations')
        del f
        output = (image, alignment_annotations)
        # output = np.squeeze(output)
        return output
