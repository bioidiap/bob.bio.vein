# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 17:12:41 2016

@author: teglitis
"""

# import what is needed:
import numpy as np
from PIL import Image, ImageDraw
import os
import six


class ManualRoiCut():
    """Readme
....
  Parameters:
  
  annotation (File, list): The name of annotation file, with full path containing 
    anotation data (Bob format, (x, y)) 
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

    def __init__(self,annotation, image = None, sizes = (480, 840)):
        #load annotations
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
                self.image = image
            self.size_y = self.image.shape[0]
            self.size_x = self.image.shape[1]
      else:
          self.image = None
          self.size_y = sizes[1]
          self.size_x = sizes[0]
    def roi_mask(self):
        mask = Image.new('L', (self.size_x, self.size_y), 0)
        ImageDraw.Draw(mask).polygon(self.annotation, outline=1, fill=1)
        mask = np.array(mask)
        mask = 0 < mask
        return mask
    def roi_image(self, pixel_value = 0):
        """Replaces outside ROI pixel values with 0 or pixel_value
        
        pixel_value (integer): if given, outside-ROI region is replaced with this 
          value. By default replaced with 0
        """
        if self.image is not None:
            mask = self.roi_mask()
            self.image[mask == 0] = pixel_value
            return self.image
        else:
            raise IOError("No input image given, can't perform non-ROI region removal")
            return 1