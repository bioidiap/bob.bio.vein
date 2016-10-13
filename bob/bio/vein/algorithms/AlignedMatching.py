#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
@author: Olegs Nikisins
"""

#==============================================================================
# Import what is needed here:

import numpy as np

from bob.bio.base.algorithm import Algorithm
#
#from scipy import ndimage
#
#import bob.math

#==============================================================================
# Main body of the class:

class AlignedMatching( Algorithm ):
    """
    This class is designed to compute the similarity score of the enroll and probe after the alignment. 
    
    The score computation pipeline is composed of the following steps:
    
    1. Alignment :
        Transformation matrix allowing to register the probe to the enroll is computed.
        
    2. Transformation :
        Data describing the enroll and probe is modified given the transformation matrix.
        
    3. Feature extraction :
        Feature vectors are computed for the transformed enroll and probe.
        
    4. Score computation : 
        The similarity score is computed given feature vectors of the enroll and probe.
    
    **Parameters:**
    
    ``aligner`` : object
        Instance of the Aligner class.
        Aligner classes are designed to compute the transformation matrix allowing to register probe to the enroll.
    
    ``transformer`` : object
        Instance of Transformer class.
        Transformer classes modify the descriptive data of the enroll and probe given the transformation matrix.
    
    ``extractor`` : object
        Instance of the Extractor class.
        Extractor classes allow to extract the feature vectors of the enroll and probe.
    
    ``algorithm`` : object
        Instance of the matching Algorithm class.
        Algorithm classes are designed to compute the similarity score given feature vectors of the enroll and probe.
    """
    
    def __init__( self, aligner, transformer, extractor, algorithm ):
        
        Algorithm.__init__( self, 
                           aligner = aligner, 
                           transformer = transformer, 
                           extractor = extractor,
                           algorithm = algorithm )
        
        self.aligner = aligner # instance of alignment class
        self.transformer = transformer # instance of feature transformation class
        self.extractor = extractor # instance of the feature extraction class
        self.algorithm = algorithm # instance of the matching algorithm
        
        
    #==========================================================================
    def get_features_after_align( self, enroll, probe ):
        """
        Finds the transformation matrix to allign the probe to the enroll
        and transforms the enroll and probe features using this matrix.
        
        **Parameters:**
        
        ``enroll`` : object
            A container with data (usually a tuple or a list) representing the enroll.
        
        ``probe`` : object
            A container with data (usually a tuple or a list) representing the probe.
        
        **Returns:**
        
        ``enroll_updated`` : object
            A container with data (usually a tuple or a list) representing the enroll after the alignment.
        
        ``probe_updated`` : object
            A container with data (usually a tuple or a list) representing the probe after the alignment.
        """
        
        M = self.aligner.get_transformation_matrix( enroll, probe ) # the transformation matrix
                
        enroll_updated, probe_updated = self.transformer.allign_enroll_probe( enroll, probe, M )
        
        return ( enroll_updated, probe_updated )
        
        
    #==========================================================================
    def get_enroll_probe_score( self, enroll, probe ):
        """
        Computes the similarity score of the enroll and probe after the alignment. 
        
        **Parameters:**
        
        ``enroll`` : object
            An object representing the enroll.
        
        ``probe`` : object
            An object representing the probe.
        
        **Returns:**
        
        ``score`` : :py:class:`float`
            The resulting similarity score.     
        """
        
        enroll_updated, probe_updated = self.get_features_after_align( enroll, probe )
        
        feature_vec_enroll = self.extractor.get_feature_vector( enroll_updated )
        
        feature_vec_probe = self.extractor.get_feature_vector( probe_updated )
        
        score = self.algorithm.score( feature_vec_enroll, feature_vec_probe )
        
        return score
        
        
    #==========================================================================
    def enroll( self, enroll_features ):
        """enroll(enroll_features) -> model
        
        This function will enroll and return the model from the given list of features.
        It must be overwritten by derived classes.
        
        **Parameters:**
        
        ``enroll_features`` : [object]
            A list of features used for the enrollment of one model.
        
        **Returns:**
        
        ``model`` : object
            The model enrolled from the ``enroll_features``.
            Must be writable with the :py:meth:`write_model` function and readable with the :py:meth:`read_model` function.
        """
        
        return enroll_features # Just leave as is
        
        
   #==========================================================================
    def score( self, model, probe ):
        """
        score(model, probe) -> score
        
        Computes the score of the probe and the model. 
        
        **Parameters:**
        
        ``model`` : :py:class:`list`
            A list of features representing the enroll.
        
        ``probe`` : object
            An object representing the probe.
        
        **Returns:**
        
        ``score`` : :py:class:`float`
            The resulting similarity score.         
        """
        
        scores = []
        
        for model_unit in model:
            
            scores.append( self.get_enroll_probe_score( model_unit, probe ) )
        
        score = np.mean( scores )
        
        return score
        
        
    #==========================================================================
    def read_probe( self, probe_file ):
        """
        read_probe(probe_file) -> probe
        
        Reads the probe feature from file.
        
        If this function is not overwritten in the aligner class the default :py:meth:`read_feature`
        method of the Algorithm class will be called. Otherwise the :py:meth:`read_feature` of the 
        aligner is called.
        
        **Parameters:**
        
        ``probe_file`` : :py:class:`str` or :py:class:`bob.io.base.HDF5File`
            The file open for reading, or the file name to read from.
        
        **Returns:**
        
        ``probe`` : object
            The probe that was read from file.
        """
        
        probe = self.aligner.read_probe( probe_file )
        
        return probe
        
        
    #==========================================================================
    def read_model( self, model_file ):
        """
        read_model(model_file) -> model
        
        Loads the enrolled model from file.
        
        If this function is not overwritten in the aligner class the default :py:meth:`read_model`
        method of the Algorithm class will be called. Otherwise the :py:meth:`read_model` of the 
        aligner is called.
        
        **Parameters:**
        
        ``model_file`` : :py:class:`str` or :py:class:`bob.io.base.HDF5File`
            The file open for reading, or the file name to read from.
        
        **Returns:**
        
        ``model`` : object
            The model that was read from file.
        """
        
        model = self.aligner.read_model( model_file )
        
        return model



