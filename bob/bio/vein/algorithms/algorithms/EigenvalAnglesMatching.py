#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:46:30 2016

@author: onikisins
"""

#==============================================================================
# Extract what is needed here:
from . import AlgorithmBase

from skimage.feature import register_translation

#==============================================================================
# Main body of the class
class EigenvalAnglesMatching(AlgorithmBase):
    """ The class to match the images of eigenvalues or angles using cross-corelation.
    
    **Parameters:**
    
    ``features_name`` : :py:class:`str`
        The name of the features to use for matching. Possible values: "eigenvalues", "angles".
    """


    def __init__(self, features_name):
        
        AlgorithmBase.__init__(self)
        
        self.features_name = features_name
        self.available_features_names = ["eigenvalues", "angles"]

    #==========================================================================
    def score(self, enroll, probe):
        """
        score(model, probe) -> score
        
        Computes the similarity score for the enroll and probe data.
        
        **Parameters:**
        
        ``enroll`` : :py:class:`list`
            Enroll data.
        ``probe`` : :py:class:`list`
            Probe data.
        
        **Returns:**
        
        ``score`` : :py:class:`float`
            The resulting similarity score.         
        """
        
        if not( self.features_name in self.available_features_names ):
            raise Exception("Specified features_name is not in the list of available_features_names")
        
        if self.features_name == "eigenvalues":
            
            shift, error, diffphase = register_translation( enroll[0], probe[0] )
            
            score = - error
        
        if self.features_name == "angles":
            
            shift, error, diffphase = register_translation( enroll[1], probe[1] )
            
            score = - error
        
        return score











































