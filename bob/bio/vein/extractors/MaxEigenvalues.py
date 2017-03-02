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

from scipy import ndimage

from skimage import exposure

from skimage.transform import rotate

from skimage import morphology

from scipy.spatial.distance import pdist, squareform

from skimage.morphology import binary_closing

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
    3. It is possible to set non-vein pixels to zero if ``segment_veins_flag`` is set to True.

    **Parameters:**

    ``sigma`` : :py:class:`float`
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.

    ``segment_veins_flag`` : :py:class:`bool`
        Set non-vein pixels to zero is set to ``True``.
        Default value: ``False``.

    ``amplify_segmented_veins_flag`` : :py:class:`bool`
        Make the intensity of the veins even if set to ``True``. Only valid, when
        ``segment_veins_flag`` is set to ``True``.
        Default value: ``False``.

    ``two_layer_segmentation_flag`` : :py:class:`bool`
        Apply the segmentation algorithm twice if set to ``True``. Only valid, when
        ``segment_veins_flag`` is set to ``True``.
        Default value: ``False``.

    ``binarize_flag`` : :py:class:`bool`
        Binarize the resulting image.
        Only valid, when ``segment_veins_flag`` and ``amplify_segmented_veins_flag``
        are set to ``True``.
        Default value: ``False``.

    ``kernel_size`` : :py:class:`float`
        Erode the veins skeleton with kernel of this size. Only valid, when
        ``binarize_flag`` is set to ``True``.
        Default value: 3.

    ``norm_p2p_dist_flag`` : :py:class:`bool`
        If ``True`` normalize the mean distance between the point pairs in the
        input binary image to the ``selected_mean_dist`` value.
        Only valis when ``binarize_flag`` is set to ``True``.
        Default: ``False``.

    ``selected_mean_dist`` : :py:class:`float`
        Normalize the mean distance between the point pairs in the
        input binary image to this value.
        Default: 100.
    """

    def __init__( self, sigma, segment_veins_flag = False,
                 amplify_segmented_veins_flag = False,
                 two_layer_segmentation_flag = False,
                 binarize_flag = False,
                 kernel_size = 3,
                 norm_p2p_dist_flag = False,
                 selected_mean_dist = 100):

        Extractor.__init__(self,
                           sigma = sigma,
                           segment_veins_flag = segment_veins_flag,
                           amplify_segmented_veins_flag = amplify_segmented_veins_flag,
                           two_layer_segmentation_flag = two_layer_segmentation_flag,
                           binarize_flag = binarize_flag,
                           kernel_size = kernel_size,
                           norm_p2p_dist_flag = norm_p2p_dist_flag,
                           selected_mean_dist = 100)

        self.sigma = sigma
        self.segment_veins_flag = segment_veins_flag
        self.amplify_segmented_veins_flag = amplify_segmented_veins_flag
        self.two_layer_segmentation_flag = two_layer_segmentation_flag
        self.binarize_flag = binarize_flag
        self.kernel_size = kernel_size
        self.norm_p2p_dist_flag = norm_p2p_dist_flag
        self.selected_mean_dist = selected_mean_dist


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
    def segment_veins(self, image, mask):
        """
        In this function pixels considered as non-vein objects (less then
        threshold) are set to zero. The threshold is defined as a mean of all pixels
        in the ROI.

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            Input image.

        ``mask`` : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.

        **Returns:**

        ``output_image`` : 2D :py:class:`numpy.ndarray`
            Maximum eigenvalues of Hessian matrices.
        """

        image_coords = np.argwhere(mask==1) # Coordinates of the pixels in the ROI

        data_points = image[mask==1] # Values of the pixels in the ROI

        threshold = np.mean(data_points) # Threshold to distinguish veins from non-veins

        data_points[data_points<=threshold] = threshold # Set non-vein pixels to threshold value

        data_points = data_points - threshold # Set non-vein pixels to zero

        output_image = np.zeros(image.shape) # Output image

        output_image[image_coords[:,0], image_coords[:,1]] = data_points # Set vein pixels

        output_image = exposure.rescale_intensity(output_image, out_range = np.uint8)

        return output_image.astype(np.float)/255.


    #==========================================================================
    def get_max_eigenvalues(self, image, mask, sigma):
        """
        Compute the maximum eigenvalues of the Hessian matrices
        for each pixel in the input image.

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            Input image.

        ``mask`` : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.

        ``sigma`` : :py:class:`float`
            Standard deviation used for the Gaussian kernel, which is used as weighting function for the auto-correlation matrix.

        **Returns:**

        ``max_eigenvalues`` : 2D :py:class:`numpy.ndarray`
            Maximum eigenvalues of Hessian matrices.
        """

        # Compute the components of the hessian matrix for each pixel in the image:
        (Hxx, Hxy, Hyy) = self.hessian_matrix_fast( image, sigma = sigma, mode = 'constant', cval = 0 )

        T = Hxx + Hyy

        D = Hxx * Hyy - Hxy ** 2

        max_eigenvalues = T/2 + np.sqrt(T**2/4 - D)

        max_eigenvalues[np.isnan(max_eigenvalues)] = 0

        max_eigenvalues = max_eigenvalues * mask

        return max_eigenvalues


    #==========================================================================
    def amplify_segmented_veins_1d(self, image):
        """
        This function amplifies the segmented veins in one diamension

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            Input image with segmented veins.

        **Returns:**

        ``result`` : 2D :py:class:`numpy.ndarray`
            Output image with amplified veins in 1D.
        """

        result = np.zeros(image.shape)

        threshold = 1/20. * np.max(image)

        for vec_num, vec in enumerate(image):

            vec[vec < threshold] = 0

            zeros_coords = np.squeeze(np.argwhere(vec==0))

            peaks = zeros_coords[np.argwhere( (zeros_coords[1:] - zeros_coords[:-1])>1 )]

            peaks = peaks.tolist()
            peaks = [item[0] for item in peaks]
            peaks.append(len(vec))
            peaks = [0] + peaks

            ampl = np.zeros(len(vec))

            for idx, start in enumerate(peaks[:-1]):

                end = peaks[idx + 1]

                if np.sum(vec[start:end]):

                    ampl[start:end] = 1/np.max(vec[start:end])

            vec_ampl = ampl*vec

            result[vec_num, :] = vec_ampl

        return result


    #==========================================================================
    def amplify_segmented_veins(self, image):
        """
        This function amplifies segmented veins in both horizontal and vertical
        directions using ``amplify_segmented_veins_1d()`` function.

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            Input image with segmented veins.

        **Returns:**

        ``veins`` : 2D :py:class:`numpy.ndarray`
            Output image with amplified veins in 2D.
        """

        hor_result = self.amplify_segmented_veins_1d(image)

        vert_result = np.transpose(self.amplify_segmented_veins_1d(np.transpose(image)))

        diag_result_1 = rotate(self.amplify_segmented_veins_1d(rotate(image, 45, preserve_range = True)), -45, preserve_range = True)

        diag_result_2 = rotate(self.amplify_segmented_veins_1d(rotate(image, -45, preserve_range = True)), 45, preserve_range = True)

        veins = np.max(np.dstack([vert_result, hor_result, diag_result_1, diag_result_2]), axis = 2)

        return veins


    #==========================================================================
    def two_layer_segmentation(self, max_eigenvalues, mask, sigma):
        """
        In this method ``get_max_eigenvalues`` method is applied for the second
        time to the modified ``max_eigenvalues`` image.
        Non-vein objects (negative values) are then set to zero.

        **Parameters:**

        ``max_eigenvalues`` : 2D :py:class:`numpy.ndarray`
            Input image - maximum eigenvalues of Hessian matrices.

        ``mask`` : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.

        ``sigma`` : :py:class:`float`
            Standard deviation used for the Gaussian kernel, which is used as
            weighting function for the auto-correlation matrix.

        **Returns:**

        ``output_image`` : 2D :py:class:`numpy.ndarray`
            Output image with segmented veins.
        """

        max_eigenvalues = np.max(max_eigenvalues) - max_eigenvalues # make veins dark with this operation

        features_2 = self.get_max_eigenvalues(max_eigenvalues, mask, sigma) # apply filtering for the second time

        features_2[features_2<0] = 0 # set negatives to zero

        output_image = exposure.rescale_intensity(features_2, out_range = np.uint8)

        return output_image.astype(np.float)/255.


    #==========================================================================
    def binarize_image(self, image, kernel_size):
        """
        Binarize and erode the input image.

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            Image with segmented and amplified veins.

        ``kernel_size`` : :py:class:`float`
            Size of the square kernel for binary erosion and closing.

        **Returns:**

        ``output_image`` : 2D :py:class:`numpy.ndarray`
            Resulting binary image.
        """

        image_binary = np.zeros(image.shape)

        image_binary[image == 1] = 1

        kernel = np.ones((kernel_size, kernel_size))

        image_dilated = morphology.binary_dilation(image_binary, selem = kernel).astype(np.float)

        output_image = morphology.binary_closing(image_dilated, selem = kernel).astype(np.float)

        return output_image


    #==========================================================================
    def find_scale(self, binary_image, selected_mean_dist):
        """
        Find the scale normalizing the mean distance between the point pairs in the
        input binary image to the ``selected_mean_dist`` value.

        **Parameters:**

        ``binary_image`` : 2D :py:class:`numpy.ndarray`
            Input binary image.

        ``selected_mean_dist`` : :py:class:`float`
            Normalize the mean distance to this value.

        **Returns:**

        ``scale`` : :py:class:`float`
            The scale to be applied to the input binary image to
            normalize the mean distance between the point pairs.
        """

        X = np.argwhere(binary_image == 1)[::10,:]

        dist_mat = squareform(pdist(X, metric='euclidean'))

        dist_mat_mean = np.mean(dist_mat)

        scale = selected_mean_dist / dist_mat_mean

        return scale


    #==========================================================================
    def scale_binary_image(self, image, scale):
        """
        Scale the input binary image. The center of mass of the scaled/output binary
        image is aligned with the center of the input image.

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            Input binary image.

        ``scale`` : :py:class:`float`
            The scale to be applied to the input binary image.

        **Returns:**

        ``image_scaled_translated`` : 2D :py:class:`numpy.ndarray`
            The scaled image.
        """

        h, w = image.shape

        image_coords = np.argwhere(image) # centered coordinates of the vein (non-zero) pixels

        offset = np.mean(image_coords, axis=0)

        image_coords = image_coords - offset

        scale_matrix = np.array([[scale, 0],
                                 [0, scale]]) # scaling matrix

        center_offset = np.array(image.shape)/2.

        coords_scaled = np.round( np.dot( image_coords, scale_matrix ) ) + center_offset

        coords_scaled = coords_scaled.astype(np.int)

        coords_scaled[coords_scaled < 0] = 0
        coords_scaled[:, 0][coords_scaled[:, 0] >= h] = h-1
        coords_scaled[:, 1][coords_scaled[:, 1] >= w] = w-1

        image_scaled_centerd = np.zeros((h, w))

        image_scaled_centerd[coords_scaled[:,0], coords_scaled[:,1]] = 1

        image_scaled_centerd = binary_closing(image_scaled_centerd, selem = np.ones((2,2)))

        image_scaled_centerd[0, : ] = 0
        image_scaled_centerd[-1, :] = 0
        image_scaled_centerd[:, 0 ] = 0
        image_scaled_centerd[:, -1] = 0

        return image_scaled_centerd.astype(np.float64)



    #==========================================================================
    def __call__(self, input_data):
        """
        Compute the maximum eigenvalues of the Hessian matrices
        for each pixel in the input image. Also, the non-vein pixels
        (less then threshold) are set to zero in the
        output matrix if corresponding flag is set to ``True``.


        **Parameters:**

        ``input_data`` : tuple
            input_data[0] is an input image: 2D `numpy.ndarray`
            input_data[1] is the binary mask of the ROI: 2D `numpy.ndarray`

        **Returns:**

        ``max_eigenvalues`` : 2D :py:class:`numpy.ndarray`
            Maximum eigenvalues of Hessian matrices with optional vein
            segmentation if ``segment_veins_flag`` is set to ``True``.
        """

        image = input_data[0] # Input image

        mask = input_data[1] # binary mask of the ROI

        max_eigenvalues = self.get_max_eigenvalues(image, mask, self.sigma)

        if self.segment_veins_flag:

            if self.two_layer_segmentation_flag:

                max_eigenvalues = self.two_layer_segmentation(max_eigenvalues, mask, self.sigma)

            else:

                max_eigenvalues = self.segment_veins(max_eigenvalues, mask)

            if self.amplify_segmented_veins_flag:

                max_eigenvalues = self.amplify_segmented_veins(max_eigenvalues)

                if self.binarize_flag:

                    max_eigenvalues = self.binarize_image(max_eigenvalues, self.kernel_size) # the output is binary image

                    if self.norm_p2p_dist_flag:

                        scale = self.find_scale(max_eigenvalues, self.selected_mean_dist)

                        max_eigenvalues = self.scale_binary_image(max_eigenvalues, scale) # the output is binary image

        return max_eigenvalues



