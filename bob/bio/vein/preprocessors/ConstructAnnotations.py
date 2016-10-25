#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

#import math
#import numpy
#from PIL import Image

#import bob.io.base
#import bob.io.image
#import bob.ip.base
#import bob.sp
#import bob.core

from bob.bio.base.preprocessor import Preprocessor
from .utils import ConstructVeinImage
from .utils import NormalizeImageRotation
#from .. import utils


class ConstructAnnotations(Preprocessor):
  """PreNone preprocessor class - an empty preprocessor that only re-saves data
  """
  def __init__(self, postprocessing = None, center = False, rotate = False, **kwargs):

    Preprocessor.__init__(self,postprocessing = postprocessing, center = center, rotate = rotate, **kwargs)
    self.center = center
    self.rotate = rotate

  def __call__(self, annotation_dictionary, annotations=None):
    """An empty __call_method that returns the input image
    """
    vein_image = ConstructVeinImage(annotation_dictionary, center = self.center)
    if self.rotate == True:
      vein_image = NormalizeImageRotation(vein_image, dark_lines = False)
    return vein_image
