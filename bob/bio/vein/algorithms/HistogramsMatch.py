#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
@author: Olegs Nikisins
"""

import numpy as np

from bob.bio.base.algorithm import Algorithm

import bob.math

class HistogramsMatch( Algorithm ):
    """ The class to match the histograms / sets of histograms.
    
    **Parameters:**
    
    similarity_metrics_name : str
        The name of the similarity metrics to use in matching. Possible values: "chi_square", "histogram_intersection".
        Default value: "chi_square".
    """


    def __init__( self, similarity_metrics_name = "chi_square" ):

        Algorithm.__init__( self, 
                           similarity_metrics_name = similarity_metrics_name )

        self.similarity_metrics_name = similarity_metrics_name
        self.available_similarity_metrics = [ "chi_square", "histogram_intersection" ]

    #==========================================================================
    def __select_first_hist__( self, data ):
        """
        This method selects the first histogram from the 2D array of the fistograms.
        
        **Parameters:**
        
        enroll_features : 1D or 2D :py:class:`numpy.ndarray`
            An array of histograms (or only one histogram / vector).
        
        **Returns:**
        
        features : 1D :py:class:`numpy.ndarray`
            The first histogram from the input array.
        """
        
        if len( data.shape ) == 3 and data.shape[0] == 1:
            data = np.squeeze( data ) # remove single-dimensional entries from the shape of an array if needed
        
        if len( data.shape )  == 2:
            
            data = data[ 0, : ] # select the first histogram from the array
        
        return data
        
    #==========================================================================
    def enroll(self, enroll_features):
        """enroll(enroll_features) -> model
        
        This function will enroll and return the model from the given list of features.
        It must be overwritten by derived classes.
        
        **Parameters:**
        
        enroll_features : [object]
            A list of features used for the enrollment of one model.
        
        **Returns:**
        
        model : object
            The model enrolled from the ``enroll_features``.
            Must be writable with the :py:meth:`write_model` function and readable with the :py:meth:`read_model` function.
        """
        
        features = []
        
        for feature in enroll_features:
            
            features.append( self.__select_first_hist__( feature ) )
        
        return features # Do nothing in our case


    #==========================================================================
    def score(self, model, probe):
        """score(model, probe) -> score
        
        Computes the score of the probe and the model using Miura matching algorithm.
        Prealignment with selected method is performed before matching if "alignment_flag = True".
        Score has a value between 0 and 0.5, larger value is better match.
        
        **Parameters:**
        
        model : a list of 1D or 2D `numpy.ndarray` arrays
            The model enrolled by the :py:meth:`enroll` function.
        probe : 2D :py:class:`numpy.ndarray`
            The probe read by the :py:meth:`read_probe` function.
        
        **Returns:**
        
        score : float
            The resulting similarity score.         
        """
        
        if len( probe.shape ) != 1:
            probe = self.__select_first_hist__( probe )
        
        if not( self.similarity_metrics_name in self.available_similarity_metrics ):
            raise Exception("Specified alignment method is not in the list of available_alignment_methods")
        
        scores = []
        
        for model_unit in model:
            
            scores.append( getattr( bob.math, self.similarity_metrics_name ) ( model_unit, probe ) )
        
        score = np.mean( scores )
        
        if self.similarity_metrics_name == "chi_square":
            
            score = -score # invert the sign of the score to make genuine scores large and impostor scores small
        
        return score


































