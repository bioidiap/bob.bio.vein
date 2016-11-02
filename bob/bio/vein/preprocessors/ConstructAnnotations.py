#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from bob.bio.base.preprocessor import Preprocessor
from .utils import ConstructVeinImage
from .utils import NormalizeImageRotation

class ConstructAnnotations(Preprocessor):
  """
  """
  def __init__(self, postprocessing = None, center = False, rotate = False, **kwargs):

    Preprocessor.__init__(self,postprocessing = postprocessing, center = center, rotate = rotate, **kwargs)
    self.center = center
    self.rotate = rotate

  def __call__(self, annotation_dictionary, annotations=None):
    """
    """
    vein_image = ConstructVeinImage(annotation_dictionary, center = self.center)
    if self.rotate == True:
      vein_image = NormalizeImageRotation(vein_image, dark_lines = False)
    return vein_image
