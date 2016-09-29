#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
@author: Olegs Nikisins
"""

#==============================================================================
# Import what is needed here:

import numpy as np

from bob.bio.base.algorithm import Algorithm

from scipy import ndimage

import bob.math

#==============================================================================
# Main body of the class:

class HessianHistMatchAligned( Algorithm ):
    """
    
    """
    
    
    def __init__( self, aligner, n_bins, eigenval_power, 
                 binarize_weights = False, 
                 similarity_metrics_name = "chi_square", 
                 alignment_method_name = "max_dot_product" ):

        Algorithm.__init__( self, 
                           aligner = aligner, 
                           n_bins = n_bins, 
                           eigenval_power = eigenval_power, 
                           binarize_weights = binarize_weights, 
                           similarity_metrics_name = similarity_metrics_name, 
                           alignment_method_name = alignment_method_name )
        
        self.aligner = aligner # instance of alignment class
        self.n_bins = n_bins
        self.eigenval_power = eigenval_power # raise to this power before histograms are computed
        
        self.binarize_weights = binarize_weights # binarize the weights
        
        self.similarity_metrics_name = similarity_metrics_name
        self.available_similarity_metrics = [ "chi_square", "histogram_intersection" ]
        
        self.alignment_method_name = alignment_method_name
        self.available_alignment_methods = [ "max_dot_product" ]



    #==========================================================================
    def get_normalized_hist_given_weights( self, data, weights, n_bins, data_min, data_max ):
        """
        This method computes the histogram of the input image/data taking pixel weights into account.
        Set the weights outside the ROI to zero if ROI is considered.
        The output histogram is also normalized making the sum of histogram entries equal to "1".
        
        **Parameters:**
        
        data : 2D :py:class:`numpy.ndarray`
            Input image.
        weights : 2D :py:class:`numpy.ndarray`
            Pixel weights to be substituted to the corresponding bins.
        n_bins : uint
            Number of bins in the output histogram.
        data_min : float
            The lower range of the bins.
        data_max : float
            The upper range of the bins.
        
        **Returns:**
        
        hist_mask_norm : 1D :py:class:`numpy.ndarray`
            Normalized histogram of the input image given weights.
        """
        
#        hist_mask = np.histogram( data, bins = ( data_max ) / n_bins * np.arange( n_bins + 1 ), weights = weights )[0]
        
        hist_mask = np.histogram( data, bins = n_bins, range = ( data_min, data_max ), weights = weights )[0]

        hist_mask = hist_mask.astype( np.float )
        
        hist_mask_norm = hist_mask / sum( hist_mask ) # make the sum of the histogram equal to 1
        
        return hist_mask_norm


    #==========================================================================
    def __unroll_data__( self, enroll, probe ):
        """
        
        """
	
        if ( len( enroll ) == 3 ) and ( len( probe ) == 3 ):
            eigenvalues_enroll = enroll[0]
            angles_enroll = enroll[1]
            mask_enroll = enroll[2]
            
            eigenvalues_probe = probe[0]
            angles_probe = probe[1]
            mask_probe = probe[2]
        else:
            raise Exception("Enroll and probe features must be a list of lenght 3")
        
        return ( eigenvalues_enroll, angles_enroll, mask_enroll, 
                eigenvalues_probe, angles_probe, mask_probe )


    #==========================================================================
    def __fill_aligner_kwargs__( self, enroll, probe, alignment_method_name ):
        """
        
        """
        
        ( eigenvalues_enroll, angles_enroll, mask_enroll, 
         eigenvalues_probe, angles_probe, mask_probe ) = self.__unroll_data__( enroll, probe )
        
        parameters_dict = {}
        if alignment_method_name == "max_dot_product":
            
            parameters_dict[ 'eigenvalues_enroll' ] = eigenvalues_enroll
            parameters_dict[ 'angles_enroll' ] = angles_enroll
            parameters_dict[ 'mask_enroll' ] = mask_enroll
            
            parameters_dict[ 'eigenvalues_probe' ] = eigenvalues_probe
            parameters_dict[ 'angles_probe' ] = angles_probe
            parameters_dict[ 'mask_probe' ] = mask_probe
        
        return parameters_dict
    

    #==========================================================================
    def allign_enroll_probe( self, enroll, probe, M ):
        """
        
        """
        
        # Unroll the input data:
        ( eigenvalues_enroll, angles_enroll, mask_enroll, 
                eigenvalues_probe, angles_probe, mask_probe ) = self.__unroll_data__( enroll, probe )
        
        relative_probe_shift = ( M[1,-1], M[0,-1] )
        
        # Shift the mask of the probe:
        mask_probe_transformed = ndimage.interpolation.shift( mask_probe, relative_probe_shift, cval = 0 )
        
        # make sure the values are correct: 0 or 1:
        mask_probe_transformed[ mask_probe_transformed<0.5 ] = 0
        mask_probe_transformed[ mask_probe_transformed>=0.5 ] = 1
        
        # The new enroll mask:
        mask_enroll_overlap = mask_enroll * mask_probe_transformed
        
        # The new probe mask:
        mask_probe_overlap = ndimage.interpolation.shift( mask_enroll_overlap, ( -relative_probe_shift[0], -relative_probe_shift[1] ), cval = 0 )
        
        mask_probe_overlap[ mask_probe_overlap<0.5 ] = 0
        mask_probe_overlap[ mask_probe_overlap>=0.5 ] = 1
        
        eigenvalues_enroll_updated = eigenvalues_enroll * mask_enroll_overlap
        angles_enroll_updated = angles_enroll * mask_enroll_overlap
        
        eigenvalues_probe_updated = eigenvalues_probe * mask_probe_overlap
        angles_probe_updated = angles_probe * mask_probe_overlap
        
        enroll_updated = ( eigenvalues_enroll_updated, angles_enroll_updated, mask_enroll_overlap )
        probe_updated = ( eigenvalues_probe_updated, angles_probe_updated, mask_probe_overlap )
        
        return ( enroll_updated, probe_updated )


    #==========================================================================
    def get_enroll_probe_hist( self, enroll, probe ):
        """
        
        """
        
        # Unroll the input data:
        ( eigenvalues_enroll, angles_enroll, mask_enroll, 
                eigenvalues_probe, angles_probe, mask_probe ) = self.__unroll_data__( enroll, probe )
        
        degrees_in_rad = 180 / np.pi
        
        angles_enroll_degrees = ( angles_enroll * degrees_in_rad + 90 ) * mask_enroll
        
        angles_probe_degrees = ( angles_probe * degrees_in_rad + 90 ) * mask_probe
        
        
        angle_min = 0 # minimum possible angle
        angle_max = 180 # maximum possible angle
        
        hist_enroll = self.get_normalized_hist_given_weights( angles_enroll_degrees, eigenvalues_enroll, self.n_bins, angle_min, angle_max )
        
        hist_probe = self.get_normalized_hist_given_weights( angles_probe_degrees, eigenvalues_probe, self.n_bins, angle_min, angle_max )
        
        return ( hist_enroll, hist_probe )


    #==========================================================================
    def get_masked_hessian_histogram( self, image, mask ):
        """
        Compute the normalized "Hessian Histogram" taking the binary mask of the ROI into account.
        
        **Parameters:**
        
        image : 2D :py:class:`numpy.ndarray`
            Input image.
        mask : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.
        
        **Returns:**
        
        hist_mask_norm : 1D :py:class:`numpy.ndarray`
            Normalized "Hessian Histogram" of the input image given ROI.
        """
        
        max_eigenvalues, angles = self.get_max_eigenvalues_and_angles( image, mask, self.sigma, self.power )
        
        angle_min = 0 # minimum possible angle
        angle_max = 180 # maximum possible angle
        
        hist_mask_norm = self.get_normalized_hist_given_weights( angles, max_eigenvalues, self.n_bins, angle_min, angle_max )
        
        return hist_mask_norm


    #==========================================================================
    def get_features_after_align( self, enroll, probe ):
        """
        
        """
        
        if not( self.alignment_method_name in self.available_alignment_methods ):
            raise Exception("Specified alignment method is not in the list of available_alignment_methods")
        
        parameters_dict = self.__fill_aligner_kwargs__( enroll, probe, self.alignment_method_name )
            
        M = self.aligner.get_transformation_matrix( **parameters_dict ) # the transformation matrix
        
        enroll_updated, probe_updated = self.allign_enroll_probe( enroll, probe, M )
        
        return ( enroll_updated, probe_updated )


    #==========================================================================
    def amplify_enroll_probe_eigenval( self, enroll, probe, eigenval_power ):
        """
        
        """
        
        enroll = ( enroll[0] ** eigenval_power, enroll[1], enroll[2] )
        
        probe = ( probe[0] ** eigenval_power, probe[1], probe[2] )
        
        return ( enroll, probe )


    #==========================================================================
    def binarize_enroll_probe_eigenval( self, enroll, probe ):
        """
        
        """
        
        enroll_binary = np.zeros( enroll[0].shape )
        
        probe_binary = np.zeros( probe[0].shape )
        
        enroll_binary[ enroll[0] > (np.max( enroll[0] )/2.0) ] = 1
        
        probe_binary[ probe[0] > (np.max( probe[0] )/2.0) ] = 1
        
        enroll = ( enroll_binary, enroll[1], enroll[2] )
        
        probe = ( probe_binary, probe[1], probe[2] )
        
        return ( enroll, probe )


    #==========================================================================
    def get_enroll_probe_score( self, enroll, probe ):
        """
        
        """
        
        if not( self.similarity_metrics_name in self.available_similarity_metrics ):
            raise Exception("Specified similarity metrics is not in the list of available_similarity_metrics")
        
        enroll_updated, probe_updated = self.get_features_after_align( enroll, probe )
        
        enroll_power, probe_power = self.amplify_enroll_probe_eigenval( enroll_updated, probe_updated, self.eigenval_power )
        
        if self.binarize_weights:
            
            enroll_power, probe_power = self.binarize_enroll_probe_eigenval( enroll_power, probe_power )
        
        hist_enroll, hist_probe = self.get_enroll_probe_hist( enroll_power, probe_power )
        
        score = getattr( bob.math, self.similarity_metrics_name ) ( hist_enroll, hist_probe )
        
        if self.similarity_metrics_name == "chi_square":
            
            score = -score # invert the sign of the score to make genuine scores large and impostor scores small
        
        return score


    #==========================================================================
    def enroll( self, enroll_features ):
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
        
        return enroll_features # Just leave as is


    #==========================================================================
    def score( self, model, probe ):
        """score(model, probe) -> score
        
        Computes the score of the probe and the model using Miura matching algorithm.
        Prealignment with selected method is performed before matching if "alignment_flag = True".
        Score has a value between 0 and 0.5, larger value is better match.
        
        **Parameters:**
        
        model : a list of 2D `numpy.ndarray` arrays
            The model enrolled by the :py:meth:`enroll` function.
        probe : tuple
            The probe read by read_probe function.
        
        **Returns:**
        
        score : float
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
        Reads the probe feature from file.
        
        **Parameters:**
        
        probe_file - name of the file.
        
        **Returns:**
        
        ( max_eigenvalues, angles, mask ) : tuple
            max_eigenvalues - 2D `numpy.ndarray` containing the maximum eigenvalues of Hessian matrices raised to the specified power.
            angles - 2D `numpy.ndarray` containing the angles (in radians) of eigenvectors with maximum eigenvalues.
            mask - 2D `numpy.ndarray` containing the binary mask of the ROI.
        """
        f = bob.io.base.HDF5File( probe_file, 'r' )
        max_eigenvalues = f.read( 'max_eigenvalues' )
        angles = f.read( 'angles' )
        mask = f.read( 'mask' )
        del f
        
        return ( max_eigenvalues, angles, mask )


    #==========================================================================
    def read_model( self, model_file ):
        """
        Loads the enrolled model from file.
        
        **Parameters:**
        
        model_file - name of the file.
        
        **Returns:**
        
        list_of_features : list
	    List list_of_features contains the tuples ( max_eigenvalues, angles, mask ), where:
            max_eigenvalues - 2D `numpy.ndarray` containing the maximum eigenvalues of Hessian matrices raised to the specified power.
            angles - 2D `numpy.ndarray` containing the angles (in radians) of eigenvectors with maximum eigenvalues.
            mask - 2D `numpy.ndarray` containing the binary mask of the ROI.
        """
        f = bob.io.base.HDF5File( model_file, 'r' )
        features = f.read( 'array' )
        del f
        
        list_of_features = []
        
        for feature in features:
            
            max_eigenvalues, angles, mask = np.vsplit( feature, 3 )
            list_of_features.append( ( max_eigenvalues, angles, mask ) )
        
        return list_of_features




