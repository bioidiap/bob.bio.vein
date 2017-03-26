#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from bob.bio.base.preprocessor import Preprocessor


class PreNone(Preprocessor):
  """PreNone preprocessor class - an empty preprocessor that only re-saves data
  """
  def __init__(self, postprocessing=None, **kwargs):

    Preprocessor.__init__(self, postprocessing=postprocessing, **kwargs)

  def __call__(self, image, annotations=None):
    """An empty __call_method that returns the input image
    """
    return image