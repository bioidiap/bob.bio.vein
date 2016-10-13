#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 17:06:06 2016

@author: onikisins
"""

class AlgorithmBase( object ):
    """
    Base class for the matching algorithms used in the AlignedMatching class.
    """
    
    def __init__( self ):
        
        pass
    
    def score( self, enroll, probe ):
        """
        score(model, probe) -> score
        
        This function will compute the score between the given model and probe.     
        
        It must be overwritten by derived classes.     
        
        **Parameters:**
        
        ``enroll`` : :py:class:`object`
            The enroll to compare the probe with.
            
        ``probe`` : :py:class:`object`
            The probe object to compare the model with.
            
        **Returns:**
            
        ``score`` : :py:class:`float`
            A similarity between ``model`` and ``probe``.       
            Higher values define higher similarities.
        """
        
        raise NotImplementedError("score method is not implemented, must overwrite in the child class")
