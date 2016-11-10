#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:20:37 2016

@author: onikisins
"""

#==========================================================================
from . import TransformerBase

from scipy import ndimage

import numpy as np

#==========================================================================
# Main body of the class:

class ShiftEnrollProbeMasked( TransformerBase ):
    """
    This class is transforming the enroll and probe features and masks given transformation matrix for the probe. 
    Used in the AlignedMatching class.
    This class is able to compensate the translation only.
    """
    
    def __init__( self, center_data_flag = False ):
        """
        
        **Parameters:**
        
        ``center_data_flag`` : :py:class:`bool`
            If set to True the images/data representing the enroll and probe will
            be centered based on the centers of mass of the binary masks of the ROIs.
            Default value: False.
        """
        
        TransformerBase.__init__( self )
        
        self.center_data_flag = center_data_flag

    #==========================================================================
    def check_the_data( self, enroll, probe, M ):
        """
        Check if correct data is passed to the :py:meth:`allign_enroll_probe` method of this class.
        
        **Parameters:**
        
        ``enroll`` : :py:class:`tuple`
            A tuple representing the enroll.
            The last element of the ``enroll`` must be the binary mask of the ROI of the enroll.
        
        ``probe`` : :py:class:`tuple`
            A tuple representing the probe.
            The last element of the ``probe`` must be the binary mask of the ROI of the probe.
        
        ``M`` : 2D :py:class:`numpy.ndarray` of type :py:class:`float`
            Transformation matrix of the size (3, 3).
        """
        
        if M.shape != ( 3, 3 ):
            raise Exception( "The transformation matrix must be of the shape (3,3)." )
        
        if ( len( enroll ) != len( probe ) ) or ( len( enroll ) < 2 ) :
            raise Exception( "Enroll and probe lists must be of the same length and of their length must be at least 2." )
        
        if ( len( set(enroll[-1].flatten()) ) != 2 ) or ( len( set(probe[-1].flatten()) ) != 2 ):
            raise Exception( "The last elements in the enroll and probe lists must be the binary mask of the ROI." )
    
    #==========================================================================
    def __reformat_data( self, data ):
        """
        Reformat the data stored in the enroll or probe tuples.
        """
        mask = data[-1] # Last element must be the ROI
        
        features = [] # List of features to be transformed
        
        for item in data[:-1]:
            features.append( item )
        
        return ( features, mask )
        
    #==========================================================================
    def __mask_features( self, features, mask ):
        """
        Mask the features in the ``features`` list using the binary mask of the ROI specified in ``mask`` argument.
        """
        
        features_masked = []
        for item in features:
            
            features_masked.append( item * mask )
            
        return features_masked
        
        
    #==========================================================================
    def center_the_data( self, data ):
        """
        Center the images stored in the input ``data`` tuple based on the center
        of mass of the binary mask of the ROI. The ROI mask is the last element in the 
        input ``data`` tuple.
        
        **Parameters:**
        
        ``data`` : :py:class:`list`
            A tuple representing the enroll or probe. The following data is stored in the tuple:
            (eigenvalue, angles, roi_mask).
            
        **Returns:**
        
        ``data_centered`` : :py:class:`list`
            A tuple representing the enroll or probe after centering. The following data is stored in the tuple:
            (eigenvalue, angles, roi_mask).
        """
        
        mask_binary = data[-1]
        
        mask_center = np.round(np.array(ndimage.center_of_mass(mask_binary)))
        
        offset = np.array(data[0].shape)/2 - mask_center
        
        data_centered = []
        
        for item in data:
            
            data_centered.append(ndimage.interpolation.shift( item, ( offset[0], offset[1] ), cval = 0 ))
            
        data_centered[-1] = data_centered[-1].astype(np.uint8)
        
        for idx, item in enumerate(data_centered[:-1]):
            
            data_centered[idx] = item * data_centered[-1]
        
        data_centered[1][data_centered[1] > np.pi/2] = np.pi/2
        
        data_centered[1][data_centered[1] < -np.pi/2] = -np.pi/2
            
        return data_centered
        
        
        
        
    #==========================================================================
    def allign_enroll_probe( self, enroll, probe, M ):
        """
        Transform the enroll and probe features and masks given transformation matrix for the probe.
        Only the translation is compensated in this case.
        
        **Parameters:**
        
        ``enroll`` : :py:class:`tuple`
            A tuple representing the enroll.
            The last element of the ``enroll`` must be the binary mask of the ROI of the enroll.
        
        ``probe`` : :py:class:`tuple`
            A tuple representing the probe.
            The last element of the ``probe`` must be the binary mask of the ROI of the probe.
        
        ``M`` : 2D :py:class:`numpy.ndarray` of type :py:class:`float`
            Transformation matrix of the size (3, 3).
            Example: if (dx>0 and dy>0) in M, then probe must be shifted (right and down) to allign it with enroll.
            Coordinates (x, y) of the pixels in probe will increase by (dx, dy) values.
            
        **Returns:**
        
        ``enroll_updated`` : :py:class:`tuple`
            A tuple with transformed data and mask of the ROI representing the enroll.
        
        ``probe_updated`` : :py:class:`tuple`
            A tuple with transformed data and mask of the ROI representing the probe.
        """
        
        # Check if correct data is passed to this function:
        self.check_the_data( enroll, probe, M )
        
        features_enroll, mask_enroll = self.__reformat_data( enroll )
        features_probe, mask_probe = self.__reformat_data( probe )
           
        relative_probe_shift = ( M[1,-1], M[0,-1] )
        
        # Shift the mask of the probe:
        mask_probe_transformed = ndimage.interpolation.shift( mask_probe, relative_probe_shift, cval = 0 )
        
        # make sure the values are correct: 0 or 1:
        mask_probe_transformed[ mask_probe_transformed < 0.5 ] = 0
        mask_probe_transformed[ mask_probe_transformed >= 0.5 ] = 1
        
        # The new enroll mask:
        mask_enroll_overlap = mask_enroll * mask_probe_transformed
        
        # The new probe mask:
        mask_probe_overlap = ndimage.interpolation.shift( mask_enroll_overlap, ( -relative_probe_shift[0], -relative_probe_shift[1] ), cval = 0 )
        
        mask_probe_overlap[ mask_probe_overlap < 0.5 ] = 0
        mask_probe_overlap[ mask_probe_overlap >= 0.5 ] = 1
        
        features_enroll_updated = self.__mask_features( features_enroll, mask_enroll_overlap )
        
        features_probe_updated = self.__mask_features( features_probe, mask_probe_overlap )
        
        features_enroll_updated.append( mask_enroll_overlap )
        features_probe_updated.append( mask_probe_overlap )
        
        enroll_updated = tuple( features_enroll_updated )
        probe_updated = tuple( features_probe_updated )
        
        if self.center_data_flag:
            
            enroll_updated = self.center_the_data(enroll_updated)
            probe_updated = self.center_the_data(probe_updated)
        
        return ( enroll_updated, probe_updated )


    #==========================================================================
    def plot_enroll_probe( self, enroll, probe, M ):
        """
        **Use this method for debugging purposes only**
        
        Plot the enroll and probe data before and after alignment.
        
        **Parameters:**
        
        ``enroll`` : :py:class:`tuple`
            A tuple representing the enroll.
            The last element of the ``enroll`` must be the binary mask of the ROI of the enroll.
        
        ``probe`` : :py:class:`tuple`
            A tuple representing the probe.
            The last element of the ``probe`` must be the binary mask of the ROI of the probe.
        
        ``M`` : 2D :py:class:`numpy.ndarray` of type :py:class:`float`
            Transformation matrix of the size (3, 3).
        """
        
        # import only if this function is used:
        import matplotlib.pyplot as plt
        
        plt.figure()
        
        for idx, (item_enroll, item_probe) in enumerate( zip( enroll, probe ) ):
            
            plt.subplot( 2, len( enroll ), idx + 1 ),plt.imshow( item_enroll,cmap = 'gray')
            plt.title( 'features%d enroll'% (idx + 1) ), plt.xticks([]), plt.yticks([])
            plt.subplot( 2, len( probe ), idx + 1 + len( probe ) ),plt.imshow( item_probe,cmap = 'gray')
            plt.title( 'features%d probe'% (idx + 1) ), plt.xticks([]), plt.yticks([])
            
        plt.show()
        
        enroll_updated, probe_updated = self.allign_enroll_probe( enroll, probe, M )
        plt.figure()
        
        for idx, (item_enroll, item_probe) in enumerate( zip( enroll_updated, probe_updated ) ):
            
            plt.subplot( 2, len( enroll ), idx + 1 ),plt.imshow( item_enroll,cmap = 'gray')
            plt.title( 'features%d enroll'% (idx + 1) ), plt.xticks([]), plt.yticks([])
            plt.subplot( 2, len( probe ), idx + 1 + len( probe ) ),plt.imshow( item_probe,cmap = 'gray')
            plt.title( 'features%d probe'% (idx + 1) ), plt.xticks([]), plt.yticks([])
            
        plt.show()        





