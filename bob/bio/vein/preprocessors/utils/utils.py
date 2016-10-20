# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 17:12:41 2016
"""

# import what is needed:
import numpy as np
from PIL import Image, ImageDraw
import os
import six


class ManualRoiCut():
    """Class for manual roi extraction -- ManualRoiCut.
    Use examples:
       
       * To generate ROI mask:
         
         ```
         from bob.bio.vein.preprocessors.utils import ManualRoiCut
         roi = ManualRoiCut(roi_annotation_points).roi_mask()
         ```
       * To replace outside-ROI regins with ``pixel_value``:
         
         ```
         from bob.bio.vein.preprocessors.utils import ManualRoiCut
         image_cutted = ManualRoiCut(roi_annotation_points, image).roi_image(pixel_value=0)
         ```
....
  Parameters:
  
  annotation (File, list): The name of annotation file, with full path containing 
    annotation data (Bob format, (x, y)) 
    OR
    the list of annotation points (tuples) in Bob format -- (x, y)
    
  image (File, numpy.ndarray), optional: The name of the image to be annotation,
    with full path,
    OR
    image data as numpy.ndarray
    image is an optional parameter, because it isn't needed to generate binary 
    mask.
    
  sizes (tuple): Optional parameter - a tuple of image size in Bob format (x,y).
    This parameter is used IF no image is given to generate binary mask.
  """

    def __init__(self,annotation, image = None, sizes = (480, 480)):
      if isinstance(annotation, six.string_types):
          if os.path.exists(annotation):
              with open(annotation,'r') as f:
                  retval = np.loadtxt(f, ndmin=2)
              self.annotation = list([tuple([k[1], k[0]]) for k in retval])
          else:
              raise IOError("Doesn' t exist file: {}".format(annotation))
              return 1
      else :
          # Convert from Bob format(x,y) to regular (y, x)
          self.annotation = list([tuple([k[1], k[0]]) for k in annotation])
      
      #load image:
      if image is not None:
            if isinstance(image, six.string_types):
                if os.path.exists(image):
                    image = Image.open(image)
                    self.image = np.array(image)
                else:
                    raise IOError("Doesn't exist file: {}".format(annotation))
                    return 1
            else:
                self.image = np.array(image)
            self.size_y = self.image.shape[0]
            self.size_x = self.image.shape[1]
      else:
          self.image = None
          self.size_y = sizes[1]
          self.size_x = sizes[0]
    def roi_mask(self):
        
        """Method roi_mask - generates ROI mask.
          
        Returns: A uint8 image containing ROI mask. Value ``1`` determines ROI area,
        value ``0`` -- outside ROI area.
        """
        mask = Image.new('L', (self.size_x, self.size_y), 0)
        ImageDraw.Draw(mask).polygon(self.annotation, outline=1, fill=1)
        mask = np.array(mask)
        mask = 0 < mask
        return mask
    def roi_image(self, pixel_value = 0):
        """Method roi_image - replaces outside ROI pixel values with 0 or pixel_value
        
        pixel_value (integer): if given, outside-ROI region is replaced with this 
          value. By default replaced with 0.
          
        Returns: A copy of image that class was initialized with, with replaced
          outside ROI pixel values with ``pixel_value``.
        """
        if self.image is not None:
            mask = self.roi_mask()
            self.image[mask == 0] = pixel_value
            return self.image
        else:
            raise IOError("No input image given, can't perform non-ROI region removal")
            return 1
