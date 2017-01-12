#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from bob.bio.base.preprocessor import Preprocessor
from .utils import ConstructVeinImage
from .utils import NormalizeImageRotation
import bob.io.base
from bob.bio.vein.preprocessors.utils import ManualRoiCut


class ManualData(Preprocessor):
    """
    ManualData preprocessor.
    Preprocessor is an improvement over preprocesors:
    
        * ConstructAnnotations;
        * ManualRoi;
        * PreNone.
        
    These preprocesors should be excluded from bob.bio.vein.
    Preprocesor returns:
        
        [original image or manual veins] + ROI mask
        
    or:
        
        [original image or manual veins] + ROI mask + alignment annotations
        
    """
    def __init__(self,
                 vein_annotations=False,
                 alignment_annotations=False,
                 erode_size=0,
                 **kwargs):
        Preprocessor.__init__(self,
                              vein_annotations=vein_annotations,
                              alignment_annotations=alignment_annotations,
                              erode_size=erode_size,
                              **kwargs)

        self.vein_annotations = vein_annotations
        self.alignment_annotations = alignment_annotations
        self.erode_size = erode_size

    def __call__(self, annotation_dictionary, annotations=None):
        if self.vein_annotations is True:
            image = ConstructVeinImage(annotation_dictionary,
                                       center=False) * 255
        else:
            image = annotation_dictionary["image"]

        mask = ManualRoiCut(annotation_dictionary["roi_annotations"]).\
            roi_mask(erode_size=self.erode_size)

        if self.alignment_annotations is True:
            alignment_annotations =\
                annotation_dictionary["alignment_annotations"]
            return(image, mask, alignment_annotations)
        else:
            return(image, mask)

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
        f.set('mask', data[1])
        if self.alignment_annotations is True:
            f.set('alignment annotations', data[2])

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
        mask = f.read('mask')
        if self.alignment_annotations is True:
            alignment_annotations = f.read('alignment annotations')
            output = (image, mask, alignment_annotations)
        else:
            output = (image, mask)
        del f

        return output
