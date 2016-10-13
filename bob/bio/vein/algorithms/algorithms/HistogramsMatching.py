#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
@author: Olegs Nikisins
"""

import bob.math

from . import AlgorithmBase

class HistogramsMatching( AlgorithmBase ):
    """ The class to match the histograms / sets of histograms.
    
    **Parameters:**
    
    ``similarity_metrics_name`` : :py:class:`str`
        The name of the similarity metrics to use in matching. Possible values: "chi_square", "histogram_intersection".
        Default value: "chi_square".
    """


    def __init__( self, similarity_metrics_name = "chi_square" ):
        
        AlgorithmBase.__init__( self )
        
        self.similarity_metrics_name = similarity_metrics_name
        self.available_similarity_metrics = [ "chi_square", "histogram_intersection" ]

    #==========================================================================
    def score(self, enroll, probe):
        """
        score(model, probe) -> score
        
        Computes the similarity score for the enroll and probe histograms.
        
        **Parameters:**
        
        ``enroll`` : 1D :py:class:`numpy.ndarray`
            Enroll data.
        ``probe`` : 1D :py:class:`numpy.ndarray`
            Probe data.
        
        **Returns:**
        
        ``score`` : :py:class:`float`
            The resulting similarity score.         
        """
        
        
        if not( self.similarity_metrics_name in self.available_similarity_metrics ):
            raise Exception("Specified similarity metrics is not in the list of available_similarity_metrics")
            
        score = getattr( bob.math, self.similarity_metrics_name ) ( enroll, probe )
        
        if self.similarity_metrics_name == "chi_square":
            
            score = -score # invert the sign of the score to make genuine scores large and impostor scores small
        
        return score




