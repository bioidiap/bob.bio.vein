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
class SpatEnhancEigenvalMasked(ExtractorBase):
    """
    This class is designed to split the image of eigenvalues into 4 subregions and save them in a list.
    The subregions are also mean normalized.
    """
    
    def __init__( self ):
        
        ExtractorBase.__init__( self )
        
        
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
    def get_masked_spat_enhanc_eigenval( self, eigenvalues, mask ):
        """
        Split the image of eigenvalues into 4 subregions and save them in a list.
        The subregions are also mean normalized.
        
        **Parameters:**
        
        ``eigenvalues`` : 2D :py:class:`numpy.ndarray`
            Array containing the magnitudes of maximum eigenvectors of Hessian matrices.
            
        ``mask`` : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.
            
            
        **Returns:**
        
        ``eigenvalues_list`` : :py:class:`list`
            List containing 4 subregions of eigenvalues.
        """
        
        subregions = self.get_four_masks( mask )
        
        eigenvalues_list = []
        
                
        for i in range(4):
            
            current_mask = subregions[ "mask{}".format( i + 1 ) ]
            
            current_eigenvalues = eigenvalues * current_mask
            
            current_eigenvalues_mean = np.sum( current_eigenvalues ) / np.sum( current_mask )
            
            current_eigenvalues = ( current_eigenvalues - current_eigenvalues_mean ) * current_mask
            
            eigenvalues_list.append( current_eigenvalues )
                
        return eigenvalues_list
        
        
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
        
#        angles = data[1]
        
        mask = data[2]
        
        feature_vector = self.get_masked_spat_enhanc_eigenval( eigenvalues, mask )
        
        return feature_vector




