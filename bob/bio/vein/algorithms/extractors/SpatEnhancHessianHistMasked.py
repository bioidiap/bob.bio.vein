#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 09:10:31 2016

@author: onikisins
"""

#==============================================================================
# Extract what is needed here:
from . import ExtractorBase

import numpy as np

from scipy import ndimage

#==============================================================================
# Main body of the class
class SpatEnhancHessianHistMasked(ExtractorBase):
    """
    This class is designed to compute the normalized "Spatially Enhanced Hessian Histogram" 
    of the input image taking the binary mask of the ROI into account.
    
    The algorithm is composed of the following steps:
    
    1.  Split the binary mask of the ROI into 4 subregions. 
        The regioning grid is centered in the center of mass of the original binary mask.
        For each subregion do the following:
        
        1. Compute the histogram of angles considering eigenvalues as weights to be substituted to the corresponding bins.
           The weights outside the subregion are zeros.
        2. It is also possible to modify the weights by raising them to the specified power.
           By default the power is 1, meaning that weights are equal to eigenvalues.
           
    2.  Concatenate the histograms of the subregions into a single "Spatially Enhanced Hessian Histogram"
        
    **Parameters:**
    
    ``n_bins`` : :py:class:`uint`
        Number of bins in the histogram of the subregion.
        The length of the resulting histogram is ``4 * n_bins``
        
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
    def get_four_masks( self, mask ):
        """
        This method splits binary region into four subregions.
        The splitting grid is centered in the center of mass of the binary region.
        
        **Parameters:**
        
        ``mask`` : 2D :py:class:`numpy.ndarray`
            Input binary image.
        
        **Returns:**
        
        ``subregions`` : :py:class:`dict`
            Dictionary containing 4 binary subregions: mask1, ..., mask4. 
        """
        
        if np.isnan( ndimage.measurements.center_of_mass( mask )[0] ):
            
            coords = np.uint( ( mask.shape[0]/2, mask.shape[1]/2 ) )
            
        else:
            
            coords = np.uint( ndimage.measurements.center_of_mass( mask ) ) # (row, column) of the center of mass 
            
        subregions = {}
        
        for i in range(4):
            
            mask_copy = np.copy( mask )
            
            if i == 0:
                mask_copy[ coords[0]:, : ] = 0
                mask_copy[ :, coords[1]: ] = 0
                subregions[ "mask{}".format( i + 1 ) ] = mask_copy
            
            if i == 1:
                mask_copy[ coords[0]:, : ] = 0
                mask_copy[ :, :coords[1] ] = 0
                subregions[ "mask{}".format( i + 1 ) ] = mask_copy
            
            if i == 2:
                mask_copy[ :coords[0], : ] = 0
                mask_copy[ :, coords[1]: ] = 0
                subregions[ "mask{}".format( i + 1 ) ] = mask_copy
            
            if i == 3:
                mask_copy[ :coords[0], : ] = 0
                mask_copy[ :, :coords[1] ] = 0
                subregions[ "mask{}".format( i + 1 ) ] = mask_copy
            
        return subregions
        
        
    #==========================================================================
    def get_masked_spat_enhanc_hessian_histogram( self, eigenvalues, angles, mask, power ):
        """
        Compute the normalized "Spatially Enhanced Hessian Histogram" taking the binary mask of the ROI into account.
        
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
        
        ``spatial_hist_mask_norm`` : 1D :py:class:`numpy.ndarray`
            Normalized "Spatially Enhanced Hessian Histogram" of the input image given ROI.
        """
        
        eigenvalues = eigenvalues ** power
        
        degrees_in_rad = 180 / np.pi
        
        angles_degrees = ( angles * degrees_in_rad + 90 ) * mask # convert radians to degrees in the [0, 180] range
        
        angle_min = 0 # minimum possible angle
        angle_max = 180 # maximum possible angle
        
        subregions = self.get_four_masks( mask )
        
        hist_mask_norm_list = []
        
        for i in range(4):
            
            current_mask = subregions[ "mask{}".format( i + 1 ) ]
            
            current_eigenvalues = eigenvalues * current_mask
            
            current_angles_degrees = angles_degrees * current_mask
            
            hist_mask_norm = self.get_normalized_hist_given_weights( current_angles_degrees, current_eigenvalues, self.n_bins, angle_min, angle_max )
            
            hist_mask_norm_list.append( hist_mask_norm )
            
        spatial_hist_mask_norm = np.hstack( hist_mask_norm_list )
                
        return spatial_hist_mask_norm
        
        
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
        
        feature_vector = self.get_masked_spat_enhanc_hessian_histogram( eigenvalues, angles, mask, self.eigenval_power )
        
        return feature_vector




