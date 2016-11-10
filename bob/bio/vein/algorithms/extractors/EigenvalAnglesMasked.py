#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 09:10:16 2016

@author: onikisins
"""


#==============================================================================
# Extract what is needed here:
from . import ExtractorBase

#==============================================================================
# Main body of the class
class EigenvalAnglesMasked(ExtractorBase):
    """
    This class is designed to return the masked images of eigenvalues and angles.
    """
    
    def __init__( self ):
        
        ExtractorBase.__init__( self )        
        
    #==========================================================================
    def get_feature_vector( self, data ):
        """
        Extract feature vector from the data container representing the enroll or probe sample.
        
        **Parameters:**
        
        ``data`` : object
            A container with data (usually a tuple or a list) representing the enroll or probe.
            In this case this is a tuple containing the following data: (eigenvalues, angles, roi_mask)
            
        **Returns:**
        
        ``feature_vector`` : object
            A feature vector obtained from the ``data`` object.
            In this case a tuple containing the following data: (eigenvalues, angles)
        """
        
        eigenvalues = data[0]
        
        angles = data[1]
        
        return (eigenvalues, angles)
