#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:22:52 2016

@author: teglitis
"""

from bob.bio.base.preprocessor import Preprocessor
import bob.io.image
from bob.bio.vein.preprocessors.utils import ManualRoiCut

class ManualRoi(Preprocessor):
    """ManualRoi preprocessor class - preprocessor that returns both - original
       image and manual vein annotations. In case of ``bob.bio.vein`` database
       needs to be used with entry-point ``biowave_v1_e`` (that returns ``dict``
       object insetead of image).
    """
    def __init__(self, **kwargs):
  
      Preprocessor.__init__(self, **kwargs)

    def __call__(self, annotation_dictionary, annotations = None):
        """
          Call method. 
          
          Args:
            annotation_dictionary (:py:class:`dict`): Dictionary containing image
                and annotation data. Such :py:class:`dict` can be returned by the 
                high level ``bob.db.biowave_v1``  implementation of the ``bob.db.biowave_v1``
                database. It is supposed to contain fields (as can be returned by the
                ``bob.db.biowave_v1`` high level implementation using ``setup`` entry
                point ``biowave_v1_e``):
                
                    - ``image``
                    - ``roi_annotations``
                    - ``vein_annotations``
                  
                Although only the variables ``image`` and ``roi_annotations`` are
                used.
                ``vein_annotations`` are used.
            annotations : This argument isn't used (Default - ``None``).
          
          Returns:
            :py:object:`tuple` : A tuple object containing variables:
            
                ``image`` -- the original image;
                ``roi_mask`` -- manual roi region converted to an image.
        """
        
        image            = annotation_dictionary["image"]
        roi_mask = ManualRoiCut(annotation_dictionary["roi_annotations"]).roi_mask()
        return (image, roi_mask)
    
    #==========================================================================
    def write_data( self, data, file_name ):
        """
        Writes the given data (that has been generated using the __call__ function of this class) to file.
        This method overwrites the write_data() method of the Preprocessor class.
        
        Args:
        data : data returned by the ``__call__`` method of the class,
        file_name : name of the file
        """
        
        f = bob.io.base.HDF5File( file_name, 'w' )
        f.set( 'image', data[ 0 ] )
        f.set( 'mask', data[ 1 ] )
        del f
        
    #==========================================================================
    def read_data( self, file_name ):
        """
        Reads the preprocessed data from file.
        his method overwrites the read_data() method of the Preprocessor class.
        
        Args:
        file_name : name of the file.
        
        Return:
        
        """
        f = bob.io.base.HDF5File( file_name, 'r' )
        image = f.read( 'image' )
        mask = f.read( 'mask' )
        del f
        return ( image, mask )
        