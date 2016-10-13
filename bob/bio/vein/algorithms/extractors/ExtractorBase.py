#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 17:06:06 2016

@author: onikisins
"""

class ExtractorBase( object ):
    """
    Base class for the classes extracting the feature vectors from enroll or probe data containers.
    Used in the AlignedMatching class.
    """
    
    def __init__( self ):
        
        pass
        
        
    #==========================================================================
    def get_feature_vector( self, data ):
        """
        Extract feature vector from the data container representing the enroll or probe sample.
        
        **Parameters:**
        
        ``data`` : object
            A container with data (usually a tuple or a list) representing the enroll or probe.
            
        **Returns:**
        
        ``feature_vector`` : object
            A feature vector obtained from the ``data`` object.
        """
        
        raise NotImplementedError("This function must be overwritten in the derived classes.")
