#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

#==============================================================================
# Import what is needed here:

from bob.bio.base.extractor import Extractor

from skimage import feature

import numpy as np

import bob.io.base

#==============================================================================
# Class implementation:

class MaxEigenvaluesAngles( Extractor ):
    """
    This class is designed to compute the maximum eigenvalues and angles of the Hessian matrices.
    The algorithm is composed of the following steps:
    
    1) Compute Hessian matrix H for each pixel in the input image. The Hessian matrix is computed by convolving the image with 
    the second derivatives of the Gaussian kernel in the respective x- and y-directions.
    2) Perfom eigendecomposition of H finding the largest eigenvalue and the orientation/angle of the corresponding eigenvector.
    3) Eigenvalues are then normalized to be in the range of [0,1]
    4) It is also possible to modify the normalized eigenvalues by raising them to the specified power.
    By default the power is 1, meaning that normalized eigenvalues remain unchanged.
    5) Eigenvalues and angles outside the ROI are set to zero.
    
    **Parameters:**
    
    sigma : float
        Standard deviation used for the Gaussian kernel, which is used as weighting function for the auto-correlation matrix.
    power : float
        Raise the weights to the specified power. Default value: 1.
    mean_normalization_flag : float
        Normalize the image of eigenvalues to it's mean value if set to True.
    """
    
    def __init__( self, sigma, power = 1, mean_normalization_flag = False ):
        
        Extractor.__init__( self,
                           sigma = sigma,
                           power = power,
                           mean_normalization_flag = mean_normalization_flag)
        
        self.sigma = sigma
        self.power = power
        self.mean_normalization_flag = mean_normalization_flag

    #==========================================================================
    def get_max_eigenvalues_and_angles( self, image, mask, sigma, power = 1, mean_normalization_flag = False ):
        """
        Compute Hessian matrices and perform their eigendecomposition:
        
        1) Compute Hessian matrix H for each pixel in the input image. The Hessian matrix is computed by convolving the image with 
        the second derivatives of the Gaussian kernel in the respective x- and y-directions.
        2) Perfom eigendecomposition of H finding the largest eigenvalue and the orientation/angle of the corresponding eigenvector.
        3) Eigenvalues are then normalized to be in the range of [0,1]
        4) It is also possible to modify the normalized eigenvalues by raising them to the specified power.
        By default the power is 1, meaning that normalized eigenvalues remain unchanged.
        5) Eigenvalues and angles outside the ROI are set to zero.
        
        **Parameters:**
        
        image : 2D :py:class:`numpy.ndarray`
            Input image.
        mask : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.
        sigma : float
            Standard deviation used for the Gaussian kernel, which is used as weighting function for the auto-correlation matrix.
        power : float
            Raise the weights to the specified power. Default value: 1.
        mean_normalization_flag : float
            Normalize the image of eigenvalues to it's mean value if set to True.
        
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
        
        for row_idx, row in enumerate( Hxx ):
            
            for col_idx, val in enumerate( row ):
                
                hxx = Hxx[ row_idx, col_idx ]
                hxy = Hxy[ row_idx, col_idx ]
                hyy = Hyy[ row_idx, col_idx ]
                
                H = np.array( [ [ hxx, hxy ], [ hxy, hyy ] ] ) # form a hessian matrix
                
                eigenval_eigenvec = np.linalg.eigh( H ) # compute eigenvalues and eigenvectors of Hessian matrix
                
                max_eigenval = eigenval_eigenvec[0][1] # get the maxumum eigenvalue
                
                selected_eigenvector = eigenval_eigenvec[1][:,1] # the eigenvector corresponding to maximum eigenvalue
                
                angle = np.arctan( selected_eigenvector[0]/selected_eigenvector[1] ) # angle of eigenvector in degrees
                
                max_eigenvalues[ row_idx, col_idx ] = max_eigenval
                
                angles[ row_idx, col_idx ] = angle
        
        max_eigenvalues = max_eigenvalues * mask # apply mask
        
        max_eigenvalues = ( max_eigenvalues + np.abs( np.min(max_eigenvalues) ) ) * mask # make the values positive and mask again
        
        if mean_normalization_flag:
            
            max_eigenvalues = max_eigenvalues / np.average(max_eigenvalues, weights = mask)
            
        else:
            
            max_eigenvalues = max_eigenvalues / np.max( max_eigenvalues ) # normalize the values to 1.
        
        max_eigenvalues = max_eigenvalues ** power # raise eigenvalues to the power
        
        angles = angles * mask # convert angles to the range of [0, 180]
        
        return ( max_eigenvalues, angles )


    #==========================================================================
    def __call__( self, input_data ):
        """
        Compute the maximum eigenvalues and angles of the Hessian matrices.
        The algorithm is composed of the following steps:
        
        1) Compute Hessian matrix H for each pixel in the input image. The Hessian matrix is computed by convolving the image with 
        the second derivatives of the Gaussian kernel in the respective x- and y-directions.
        2) Perfom eigendecomposition of H finding the largest eigenvalue and the orientation/angle of the corresponding eigenvector.
        3) Eigenvalues are then normalized to be in the range of [0,1]
        4) It is also possible to modify the normalized eigenvalues by raising them to the specified power.
        By default the power is 1, meaning that normalized eigenvalues remain unchanged.
        5) Eigenvalues and angles outside the ROI are set to zero.
        
        **Parameters:**
        
        input_data : tuple
            input_data[0] is an input image: 2D `numpy.ndarray`
            input_data[1] is the binary mask of the ROI: 2D `numpy.ndarray`
        
        **Returns:**
        
        ( max_eigenvalues, angles, mask ) : tuple
            max_eigenvalues - 2D `numpy.ndarray` containing the maximum eigenvalues of Hessian matrices raised to the specified power.
            angles - 2D `numpy.ndarray` containing the angles (in radians) of eigenvectors with maximum eigenvalues.
            mask - 2D `numpy.ndarray` containing the binary mask of the ROI.
        """
        
        image = input_data[0] # Input image
        
        mask = input_data[1] # binary mask of the ROI
        
        max_eigenvalues, angles = self.get_max_eigenvalues_and_angles(image, mask, self.sigma, self.power,
                                                                      self.mean_normalization_flag)
        
        return ( max_eigenvalues, angles, mask )

    #==========================================================================
    def write_feature( self, data, file_name ):
        """
        Writes the given data (that has been generated using the __call__ function of this class) to file.
        This method overwrites the write_feature() method of the Extractor class.
        
        **Parameters:**
        
        data - data returned by the __call__ method of the class,
        file_name - name of the file
        """
        
        f = bob.io.base.HDF5File( file_name, 'w' )        
        f.set( 'max_eigenvalues', data[ 0 ] )
        f.set( 'angles', data[ 1 ] )
        f.set( 'mask', data[ 2 ] )
        del f

    #==========================================================================
    def read_feature( self, file_name ):
        """
        Reads the preprocessed data from file.
        This method overwrites the read_feature() method of the Extractor class.
        
        **Parameters:**
        
        file_name - name of the file.
        
        **Returns:**
        
        features : 2D `numpy.ndarray`
	    The following arrays are stacked vertically into the features array:
            max_eigenvalues - 2D `numpy.ndarray` containing the maximum eigenvalues of Hessian matrices raised to the specified power.
            angles - 2D `numpy.ndarray` containing the angles (in radians) of eigenvectors with maximum eigenvalues.
            mask - 2D `numpy.ndarray` containing the binary mask of the ROI.
        """
        f = bob.io.base.HDF5File( file_name, 'r' )
        max_eigenvalues = f.read( 'max_eigenvalues' )
        angles = f.read( 'angles' )
        mask = f.read( 'mask' )
        del f
        
        features = np.vstack( ( max_eigenvalues, angles, mask ) ) # stack all data into a single array, more convinient for enrollment 
        
        return features




