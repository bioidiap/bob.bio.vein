#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from bob.bio.base.preprocessor import Preprocessor
from .utils import ConstructVeinImage
from .utils import NormalizeImageRotation
import bob.io.base


class ConstructAnnotations(Preprocessor):
    """
    Constract annotations.
    TO DO- explaning what and how
    """
    def __init__(self,
                 postprocessing=None,
                 center=False,
                 rotate=False,
                 **kwargs):
        Preprocessor.__init__(self,
                              postprocessing=postprocessing,
                              center=center,
                              rotate=rotate,
                              **kwargs)
        self.center = center
        self.rotate = rotate

    def __call__(self, annotation_dictionary, annotations=None):
        """
        """
        vein_image = ConstructVeinImage(annotation_dictionary,
                                        center=self.center)
        if self.rotate is True:
            vein_image = NormalizeImageRotation(vein_image, dark_lines=False)

        alignment_annotations = annotation_dictionary["alignment_annotations"]

        return (vein_image, alignment_annotations)

    def write_data(self, data, file_name):
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

    def read_data(self, file_name):
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
        return output
