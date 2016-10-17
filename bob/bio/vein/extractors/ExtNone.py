#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


from bob.bio.base.extractor import Extractor


class ExtNone( Extractor ):
    """
    An empty extracror class that returns an input image
    """

    def __init__( self):

        Extractor.__init__( self)

    def __call__( self, image):
        """
        Returns the input image (does absolutely nothing)
        """
        return image