#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 17:06:06 2016

@author: onikisins
"""

class TransformerBase( object ):
    """
    Base class for the classes transforming the enroll and probe features/masks given transformation matrix for the probe. 
    Used in the AlignedMatching class.
    """
    
    def __init__( self ):
        
        pass
        
        
    #==========================================================================
    def allign_enroll_probe( self, enroll, probe, M ):
        """
        Transform the enroll and probe features given transformation matrix for the probe.
        
        **Parameters:**
        
        ``enroll`` : object
            A container with data (usually a tuple or a list) representing the enroll.
        
        ``probe`` : object
            A container with data (usually a tuple or a list) representing the probe.
        
        ``M`` : 2D :py:class:`numpy.ndarray` of type :py:class:`float`
            Transformation matrix of the size (3, 3).
            
        **Returns:**
        
        ``enroll_updated`` : :py:class:`tuple`
            A tuple with transformed data representing the enroll.
        
        ``probe_updated`` : :py:class:`tuple`
            A tuple with transformed data representing the probe.
        """
        
        raise NotImplementedError("This function must be overwritten in the derived classes.")
