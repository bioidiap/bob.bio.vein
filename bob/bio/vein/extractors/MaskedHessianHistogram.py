#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

#==============================================================================
# Import what is needed here:

from bob.bio.base.extractor import Extractor

from skimage import feature

import numpy as np

#==============================================================================
# Class implementation:

class MaskedHessianHistogram( Extractor ):
    """
    This class is designed to compute the normalized "Hessian Histogram" of the input image taking the binary mask of the ROI into account. 
    The algorithm is composed of the following steps:
    
    1) Compute Hessian matrix H for each pixel in the input image. The Hessian matrix is computed by convolving the image with 
    the second derivatives of the Gaussian kernel in the respective x- and y-directions.
    2) Perfom eigendecomposition of H finding the largest eigenvalue and the orientation/angle of the corresponding eigenvector.
    3) Compute the histogram of angles considering eigenvalues as weights to be substituted to the corresponding bins.
    The weights outside the ROI are zeros.
    4) It is also possible to modify the weights by raising them to the specified power.
    By default the power is 1, meaning that weights are equal to eigenvalues.
    
    **Parameters:**
    
    sigma : float
        Standard deviation used for the Gaussian kernel, which is used as weighting function for the auto-correlation matrix.
    n_bins : uint
        Number of bins in the output histogram.
    power : float
        Raise the weights to the specified power. Default value: 1.
    """
    
    def __init__( self, sigma, n_bins, power = 1 ):
        
        Extractor.__init__( self,
                           sigma = sigma,
                           n_bins = n_bins,
                           power = power )
        
        self.sigma = sigma
        self.n_bins = n_bins
        self.power = power

    #==========================================================================
    def get_max_eigenvalues_and_angles( self, image, mask, sigma, power = 1 ):
        """
        Compute Hessian matrices and perform their eigendecomposition:
        
        1) Compute Hessian matrix H for each pixel in the input image,
        2) Perfom eigendecomposition of H finding the largest eigenvalue and the orientation/angle of the corresponding eigenvector.
        3) Raise eigenvalues to the specified power.
        
        **Parameters:**
        
        image : 2D :py:class:`numpy.ndarray`
            Input image.
        mask : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.
        sigma : float
            Standard deviation used for the Gaussian kernel, which is used as weighting function for the auto-correlation matrix.
        power : float
            Raise the weights to the specified power. Default value: 1.
        
        **Returns:**
        
        ( max_eigenvalues, angles ) : tuple
            max_eigenvalues - 2D `numpy.ndarray` containing the maximum eigenvalues of Hessian matrices raised to the specified power.
            angles - 2D `numpy.ndarray` containing the angles of eigenvectors with maximum eigenvalues.
        """
        
        # Compute the components of the hessian matrix for each pixel in the image:
        (Hxx, Hxy, Hyy) = feature.hessian_matrix( image, sigma = sigma, mode = 'constant', cval = 0 )
        
        # Array to store the maximum eigenvalues of hessian matrices:
        max_eigenvalues = np.zeros( Hxx.shape )
        
        # Array to store the angles of eigenvectors corresponding to maximum eigenvalues of hessian matrices:
        angles = np.zeros( Hxx.shape )
        
        degrees_in_rad = 180 / np.pi
        
        for row_idx, row in enumerate( Hxx ):
            
            for col_idx, val in enumerate( row ):
                
                hxx = Hxx[ row_idx, col_idx ]
                hxy = Hxy[ row_idx, col_idx ]
                hyy = Hyy[ row_idx, col_idx ]
                
                H = np.array( [ [ hxx, hxy ], [ hxy, hyy ] ] ) # form a hessian matrix
                
                eigenval_eigenvec = np.linalg.eigh( H ) # compute eigenvalues and eigenvectors of Hessian matrix
                
                max_eigenval = eigenval_eigenvec[0][1] # get the maxumum eigenvalue
                
                selected_eigenvector = eigenval_eigenvec[1][:,1] # the eigenvector corresponding to maximum eigenvalue
                
                angle = np.arctan( selected_eigenvector[0]/selected_eigenvector[1] )*( degrees_in_rad ) # angle of eigenvector in degrees
                
                max_eigenvalues[ row_idx, col_idx ] = max_eigenval
                
                angles[ row_idx, col_idx ] = angle
        
        max_eigenvalues = max_eigenvalues * mask # apply mask
        
        max_eigenvalues = ( max_eigenvalues + np.abs( np.min(max_eigenvalues) ) ) * mask # make the values positive and mask again
        
        max_eigenvalues = max_eigenvalues / np.max( max_eigenvalues ) # normalize the values to 1.
        
        max_eigenvalues = max_eigenvalues ** power # raise eigenvalues to the power
        
        angles = ( angles + 90 ) * mask # convert angles to the range of [0, 180]
        
        return ( max_eigenvalues, angles )


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
    def __call__( self, input_data ):
        """
        Compute the normalized "Hessian Histogram" of the input image taking the binary mask of the ROI into account.
        The algorithm is composed of the following steps:
        
        1) Compute Hessian matrix H for each pixel in the input image. The Hessian matrix is computed by convolving the image with 
        the second derivatives of the Gaussian kernel in the respective x- and y-directions.
        2) Perfom eigendecomposition of H finding the largest eigenvalue and the orientation/angle of the corresponding eigenvector.
        3) Compute the histogram of angles considering eigenvalues as weights to be substituted to the corresponding bins.
        The weights outside the ROI are zeros.
        4) It is also possible to modify the weights by raising them to the specified power.
        By default the power is 1, meaning that weights are equal to eigenvalues.
        
        **Parameters:**
        
        input_data : tuple
            input_data[0] is an input image: 2D `numpy.ndarray`
            input_data[1] is the binary mask of the ROI: 2D `numpy.ndarray`
        
        **Returns:**
        
        hist_mask_norm : 1D :py:class:`numpy.ndarray`
            Normalized "Hessian Histogram" of the input image given ROI.
        """
        
        image = input_data[0] # Input image
        
        mask = input_data[1] # binary mask of the ROI
        
        return self.get_masked_hessian_histogram( image, mask )



