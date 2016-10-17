#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:46:30 2016

@author: onikisins
"""

#==============================================================================
# Extract what is needed here:
import numpy as np

from . import AlgorithmBase

from skimage.feature import register_translation

#==============================================================================
# Main body of the class
class SpatEnhancEigenvalMatching( AlgorithmBase ):
    """ The class to match the eigenvalues splitted into 4 subregions.
    
    **Parameters:**
    
    ``similarity_metrics_name`` : :py:class:`str`
        The name of the similarity metrics to use for matching. Possible values: "shift_std", "error_mean".
    """


    def __init__( self, similarity_metrics_name ):
        
        AlgorithmBase.__init__( self )
        
        self.similarity_metrics_name = similarity_metrics_name
        self.available_similarity_metrics = [ "shift_std", "error_mean" ]

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
        
        if not( self.similarity_metrics_name in self.available_similarity_metrics ):
            raise Exception("Specified similarity metrics is not in the list of available_similarity_metrics")
        
        shift_list = []

        error_list = []
        
        for idx in range(4):
            
            shift, error, diffphase = register_translation( enroll[idx], probe[idx] )
            
            shift_list.append( shift )
            
            error_list.append( error )
        
        shift_array = np.vstack( shift_list )
        
        if self.similarity_metrics_name == "shift_std":
            
            score = np.mean( np.std( shift_array, 0 ) )
            
            score = - score
        
        if self.similarity_metrics_name == "error_mean":
            
            score = np.mean( error_list )
            
            score = - score
        
        return score











































