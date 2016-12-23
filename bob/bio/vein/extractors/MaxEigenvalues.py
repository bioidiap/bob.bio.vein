#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 17:01:49 2016

@author: onikisins
"""

#==============================================================================
# Import what is needed here:

from bob.bio.base.extractor import Extractor

import numpy as np

import bob.io.base

from scipy import ndimage

#==============================================================================
# Class implementation:

class MaxEigenvalues( Extractor ):
    """
    This class is designed to compute the maximum eigenvalues of the Hessian matrices
    for each pixel in the input image.
    The algorithm is composed of the following steps:

    1. Compute Hessian matrix H for each pixel in the input image. The Hessian matrix is computed by convolving the image with
       the second derivatives of the Gaussian kernel in the respective x- and y-directions.
    2. Perfom eigendecomposition of H finding the largest eigenvalue of the corresponding eigenvector.
    3. It is possible to set negative eigenvalues to zero if ``set_negatives_to_zero`` set to ``True``
    4. The image of max eigenvalues can be mean-normalized if ``mean_normalization_flag`` is set to ``True``.
    5. Eigenvalues outside the ROI are set to zero.

    **Parameters:**

    ``sigma`` : :py:class:`float`
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.

    ``set_negatives_to_zero`` : :py:class:`bool`
        Set negative eigenvalues to zero if set to ``True``. This is done before
        normalization, which takes place if ``mean_normalization_flag`` is ``True``.

    ``mean_normalization_flag`` : :py:class:`bool`
        Perform mean normalization of the output image of eigenvalues if set to ``True``.
    """

    def __init__( self, sigma, set_negatives_to_zero, mean_normalization_flag ):

        Extractor.__init__( self,
                           sigma = sigma,
                           set_negatives_to_zero = set_negatives_to_zero,
                           mean_normalization_flag = mean_normalization_flag )

        self.sigma = sigma
        self.set_negatives_to_zero = set_negatives_to_zero
        self.mean_normalization_flag = mean_normalization_flag


    #==========================================================================
    def hessian_matrix_fast(self, image, sigma=1, mode='constant', cval=0):
        """
        Compute Hessian matrix (fast way) using the separability property
        of gaussian-based filters.

        The Hessian matrix is defined as::

            H = [Hxx Hxy]
                [Hxy Hyy]

        which is computed by convolving the image with the second derivatives
        of the Gaussian kernel in the respective x- and y-directions.

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            Input image.

        ``sigma`` : :py:class:`float`
            Standard deviation used for the Gaussian kernel, which is used as
            weighting function for the auto-correlation matrix.

        ``mode`` : :py:class:`str`
            {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
            How to handle values outside the image borders.

        ``cval`` : :py:class:`float`
            Optional.
            Used in conjunction with mode 'constant', the value outside
            the image boundaries.

        **Returns:**

        ``Hxx`` : 2D :py:class:`numpy.ndarray`
            Element of the Hessian matrix for each pixel in the input image.

        ``Hxy`` : 2D :py:class:`numpy.ndarray`
            Element of the Hessian matrix for each pixel in the input image.

        ``Hyy`` : 2D :py:class:`numpy.ndarray`
            Element of the Hessian matrix for each pixel in the input image.
        """

        image = image.astype(np.float)

        window_ext = max(1, np.ceil(3 * sigma))

        xy_range = np.arange(-window_ext, window_ext + 1, 1.0)

        gaussian_exp = np.exp(-(xy_range ** 2) / (2 * sigma ** 2))

        kernel_xx_1 = gaussian_exp
        kernel_xx_2 = 1 / (2 * np.pi * sigma ** 4) * (xy_range ** 2 / sigma ** 2 - 1)
        kernel_xx_2 *= gaussian_exp

        Hxx_x = ndimage.correlate1d(image, kernel_xx_1, axis = 0, mode=mode, cval=cval)
        Hxx = ndimage.correlate1d(Hxx_x, kernel_xx_2, axis = 1, mode=mode, cval=cval)

        Hyy_x = ndimage.correlate1d(image, kernel_xx_2, axis = 0, mode=mode, cval=cval)
        Hyy = ndimage.correlate1d(Hyy_x, kernel_xx_1, axis = 1, mode=mode, cval=cval)

        kernel_xy_1 = 1 / ( np.sqrt(2 * np.pi) * sigma ** 3) * (xy_range)
        kernel_xy_1 *= gaussian_exp

        Hxy_x = ndimage.correlate1d(image, kernel_xy_1, axis = 0, mode=mode, cval=cval)
        Hxy = ndimage.correlate1d(Hxy_x, kernel_xy_1, axis = 1, mode=mode, cval=cval)

        return Hxx, Hxy, Hyy


    #==========================================================================
    def mean_normalization(self, image, mask):
        """
        Perform mean normalization of the input image given weights in the mask
        array.

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            Input image.

        ``mask`` : 2D :py:class:`numpy.ndarray`
            Array with weights.

        **Returns:**

        ``image_normalized`` : 2D :py:class:`numpy.ndarray`
            Normalized image.
        """

        image_average = np.average(image, weights = mask)

        image_normalized = ( image - image_average ) * mask

        return image_normalized


    #==========================================================================
    def get_max_eigenvalues( self, image, mask, sigma, set_negatives_to_zero, mean_normalization_flag ):
        """
        Compute the maximum eigenvalues of the Hessian matrices
        for each pixel in the input image.
        The algorithm is composed of the following steps:

        1. Compute Hessian matrix H for each pixel in the input image. The Hessian matrix is computed by convolving the image with
           the second derivatives of the Gaussian kernel in the respective x- and y-directions.
        2. Perfom eigendecomposition of H finding the largest eigenvalue of the corresponding eigenvector.
        3. It is possible to set negative eigenvalues to zero if ``set_negatives_to_zero`` set to ``True``
        4. The image of max eigenvalues can be mean-normalized if ``mean_normalization_flag`` is set to ``True``.
        5. Eigenvalues outside the ROI are set to zero.

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            Input image.

        ``mask`` : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.

        ``sigma`` : :py:class:`float`
            Standard deviation used for the Gaussian kernel, which is used as weighting function for the auto-correlation matrix.

        ``set_negatives_to_zero`` : :py:class:`bool`
            Set negative eigenvalues to zero if set to ``True``. This is done before
            normalization, which takes place if ``mean_normalization_flag`` is ``True``.

        ``mean_normalization_flag`` : :py:class:`bool`
            Normalize the image of eigenvalues to it's mean value if set to ``True``.

        **Returns:**

        ``max_eigenvalues`` : 2D :py:class:`numpy.ndarray`
            Maximum eigenvalues of Hessian matrices.
        """

        # Compute the components of the hessian matrix for each pixel in the image:
        (Hxx, Hxy, Hyy) = self.hessian_matrix_fast( image, sigma = sigma, mode = 'constant', cval = 0 )

        T = Hxx + Hyy

        D = Hxx * Hyy - Hxy ** 2

        max_eigenvalues = T/2 + np.sqrt(T**2/4 - D)

        max_eigenvalues = max_eigenvalues * mask

        if mean_normalization_flag and not(set_negatives_to_zero):

            max_eigenvalues = self.mean_normalization(max_eigenvalues, mask)

        if set_negatives_to_zero:

            max_eigenvalues = self.mean_normalization(max_eigenvalues, mask)

            max_eigenvalues[max_eigenvalues < 0] = 0

            max_eigenvalues = self.mean_normalization(max_eigenvalues, mask)

        return max_eigenvalues


    #==========================================================================
    def __call__( self, input_data ):
        """
        Compute the maximum eigenvalues of the Hessian matrices
        for each pixel in the input image.
        The algorithm is composed of the following steps:

        1. Compute Hessian matrix H for each pixel in the input image. The Hessian matrix is computed by convolving the image with
           the second derivatives of the Gaussian kernel in the respective x- and y-directions.
        2. Perfom eigendecomposition of H finding the largest eigenvalue of the corresponding eigenvector.
        3. It is possible to set negative eigenvalues to zero if ``set_negatives_to_zero`` set to ``True``
        4. The image of max eigenvalues can be mean-normalized if ``mean_normalization_flag`` is set to ``True``.
        5. Eigenvalues outside the ROI are set to zero.

        **Parameters:**

        ``input_data`` : tuple
            input_data[0] is an input image: 2D `numpy.ndarray`
            input_data[1] is the binary mask of the ROI: 2D `numpy.ndarray`

        **Returns:**

        ( ``max_eigenvalues``, ``mask`` ) : tuple
            max_eigenvalues - 2D `numpy.ndarray` containing the maximum eigenvalues of Hessian matrices.
            mask - 2D `numpy.ndarray` containing the binary mask of the ROI.
        """

        image = input_data[0] # Input image

        mask = input_data[1] # binary mask of the ROI

        max_eigenvalues = self.get_max_eigenvalues( image, mask, self.sigma, self.set_negatives_to_zero,
                                                   self.mean_normalization_flag )

        return max_eigenvalues, mask

    #==========================================================================
    def write_feature( self, data, file_name ):
        """
        Writes the given data (that has been generated using the __call__ function of this class) to file.
        This method overwrites the write_feature() method of the Extractor class.

        **Parameters:**

        ``data`` : obj
            Data returned by the __call__ method of the class.

        ``file_name`` : :py:class:`str`
            Name of the file.
        """

        f = bob.io.base.HDF5File( file_name, 'w' )
        f.set( 'max_eigenvalues', data[ 0 ] )
        f.set( 'mask', data[ 1 ] )
        del f


    #==========================================================================
    def read_feature( self, file_name ):
        """
        Reads the preprocessed data from file.
        This method overwrites the read_feature() method of the Extractor class.

        **Parameters:**

        ``file_name`` : :py:class:`str`
            Name of the file.

        **Returns:**

        ``max_eigenvalues`` : 2D :py:class:`numpy.ndarray`
            Maximum eigenvalues of Hessian matrices.

        ``max_eigenvalues`` : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.
        """

        f = bob.io.base.HDF5File( file_name, 'r' )
        max_eigenvalues = f.read( 'max_eigenvalues' )
        mask = f.read( 'mask' )
        del f

        return max_eigenvalues, mask










