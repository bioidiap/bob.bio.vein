#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 13:39:41 2016

@author: onikisins
"""
#==============================================================================
# Extract what is needed here:
from . import ExtractorBase

import numpy as np

#==============================================================================
# Main body of the class
class HessianHistMasked(ExtractorBase):
    """
    This class is designed to compute the normalized "Hessian Histogram" of the input image taking the binary mask of the ROI into account. 
    The algorithm is composed of the following steps:
    
    1. Compute the histogram of angles considering eigenvalues as weights to be substituted to the corresponding bins.
       The weights outside the ROI are zeros.
    2. It is also possible to modify the weights by raising them to the specified power.
       By default the power is 1, meaning that weights are equal to eigenvalues.
    
    **Parameters:**
    
    ``n_bins`` : :py:class:`uint`
        Number of bins in the output histogram.
        
    ``eigenval_power`` : :py:class:`float`
        Raise the weights (eigenvalues) to the specified power. Default value: 1.
    """
    def __init__( self, n_bins, eigenval_power = 1 ):
        
        ExtractorBase.__init__( self )
        
        self.n_bins = n_bins
        self.eigenval_power = eigenval_power
        
        
    #==========================================================================
    def get_normalized_hist_given_weights( self, data, weights, n_bins, data_min, data_max ):
        """
        This method computes the histogram of the input image/data taking pixel weights into account.
        Set the weights outside the ROI to zero if ROI is considered.
        The output histogram is also normalized making the sum of histogram entries equal to "1".
        
        **Parameters:**
        
        ``data`` : 2D :py:class:`numpy.ndarray`
            Input image.
            
        ``weights`` : 2D :py:class:`numpy.ndarray`
            Pixel weights to be substituted to the corresponding bins.
            
        ``n_bins`` : :py:class:`uint`
            Number of bins in the output histogram.
            
        ``data_min`` : :py:class:`float`
            The lower range of the bins.
            
        ``data_max`` : :py:class:`float`
            The upper range of the bins.
        
        **Returns:**
        
        ``hist_mask_norm`` : 1D :py:class:`numpy.ndarray`
            Normalized histogram of the input image given weights.
        """
        
#        hist_mask = np.histogram( data, bins = ( data_max ) / n_bins * np.arange( n_bins + 1 ), weights = weights )[0]
        
        hist_mask = np.histogram( data, bins = n_bins, range = ( data_min, data_max ), weights = weights )[0]

        hist_mask = hist_mask.astype( np.float )
        
        hist_mask_norm = hist_mask / sum( hist_mask ) # make the sum of the histogram equal to 1
        
        return hist_mask_norm
        
        
    #==========================================================================
    def get_masked_hessian_histogram( self, eigenvalues, angles, mask, power ):
        """
        Compute the normalized "Hessian Histogram" taking the binary mask of the ROI into account.
        
        **Parameters:**
        
        ``eigenvalues`` : 2D :py:class:`numpy.ndarray`
            Array containing the magnitudes of maximum eigenvectors of Hessian matrices.
            
        ``angles`` : 2D :py:class:`numpy.ndarray`
            Array containing the orientations of maximum eigenvectors of Hessian matrices.
            
        ``mask`` : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.
            
        ``power`` : :py:class:`float`
            Raise the weights (eigenvalues) to the specified power.
            
        **Returns:**
        
        ``hist_mask_norm`` : 1D :py:class:`numpy.ndarray`
            Normalized "Hessian Histogram" of the input image given ROI.
        """
        
        eigenvalues = eigenvalues ** power
        
        degrees_in_rad = 180 / np.pi
        
        angles_degrees = ( angles * degrees_in_rad + 90 ) * mask # convert radians to degrees in the [0, 180] range
        
        angle_min = 0 # minimum possible angle
        angle_max = 180 # maximum possible angle
        
        hist_mask_norm = self.get_normalized_hist_given_weights( angles_degrees, eigenvalues, self.n_bins, angle_min, angle_max )
                
        return hist_mask_norm
        
        
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
        
        eigenvalues = data[0]
        
        angles = data[1]
        
        mask = data[2]
        
        feature_vector = self.get_masked_hessian_histogram( eigenvalues, angles, mask, self.eigenval_power )
        
        return feature_vector


