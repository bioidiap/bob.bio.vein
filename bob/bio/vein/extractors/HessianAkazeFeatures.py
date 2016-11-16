#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:05:21 2016

@author: onikisins
"""

#==============================================================================
# Import what is needed here:

from bob.bio.base.extractor import Extractor

import numpy as np

from scipy.ndimage.measurements import center_of_mass

from scipy.ndimage.filters import gaussian_filter

from skimage import transform, exposure

import cv2

from scipy.misc.pilutil import imresize

#==============================================================================
# Class implementation:

class HessianAkazeFeatures(Extractor):
    """
    This class is designed to extract the AKAZE features from Hessian image.
    The pipeline is composed of the following steps:

        1. Compute Hessian matrix H for each pixel in the input image.
        2. Perfom eigendecomposition of H finding the largest eigenvalue and the orientation/angle of the corresponding eigenvector.
           The image of largest eigenvalues is called "Hessian image" further in text.
        3. The AKAZE features are next computed for the "Hessian image".

        **Parameters:**

        ``smoothing_sigma`` : :py:class:`float`
            Parameter for the Gaussian filter, used in the preprocessing of the
            input image.

        ``image_size`` : :py:class:`list`
            The input image and mask will be rescaled to the dimensions
            specified in the ``image_size``.

        ``hessian_sigma`` : :py:class:`float`
            Standard deviation used for the Gaussian kernel.

        ``hessian_threshold`` : :py:class:`float`
            Values above this threshold are set to zero in the Hessian image.

        ``hessian_saturation`` : :py:class:`float`
            Saturation value for the Hessian image.
    """

    def __init__(self, smoothing_sigma, image_size, hessian_sigma, hessian_threshold, hessian_saturation):

        Extractor.__init__(self,
                           smoothing_sigma = smoothing_sigma,
                           image_size = image_size,
                           hessian_sigma = hessian_sigma,
                           hessian_threshold = hessian_threshold,
                           hessian_saturation = hessian_saturation)

        self.smoothing_sigma = smoothing_sigma
        self.image_size = image_size
        self.hessian_sigma = hessian_sigma
        self.hessian_threshold = hessian_threshold
        self.hessian_saturation = hessian_saturation

    #==========================================================================
    def center_image(self, img, mask, b_is_mask=False):
        """
        Center the image bsed on the center of mass of the binary mask of the ROI.

        **Parameters:**

        ``img`` : 2D :py:class:`numpy.ndarray`
            Input image.

        ``mask`` : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.

        ``b_is_mask`` : :py:class:`bool`
            Return binary image if ``True``.

        **Returns:**

        ``norm_img`` : :py:class:`bool`
            The centered image.
        """
        if img.shape != mask.shape:
            return None

        ind = center_of_mass(mask)
        diff_from_center = (ind[1] - img.shape[1] / 2, ind[0] - img.shape[0] / 2)
        shift = transform.AffineTransform(translation=diff_from_center)

        if b_is_mask:
            norm_img = (transform.warp(np.float64(img), shift) > 0) * 255

        else:
            norm_img = transform.warp(np.float64(img), shift)

        return norm_img


    #==========================================================================
    def compute_hessian(self, img, hessian_sigma, hessian_boundaries = "mirror"):
        """
        Compute components of the Hessian matrices given input image.

        **Parameters:**

        ``img`` : 2D :py:class:`numpy.ndarray`
            Input image.

        ``hessian_sigma`` : :py:class:`float`
            Standard deviation used for the Gaussian kernel.

        ``hessian_boundaries`` : :py:class:`str`
            The `mode` parameter determines how the array borders are handled.
            Possible values: {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}.
            The default value: 'mirror'.

        **Returns:**

        ``output_array`` : 4D :py:class:`numpy.ndarray`
            The array containg the ``gxx``, ``gxy``, ``gyx``, ``gyy`` components of the Hessian
            matrices for each pixel in the input image.
        """
        # Image conversion
        img = img.astype(np.float64)

        # Inverting x- and y-axis with numpy array representation (for "order" parameter)
        gxx = gaussian_filter(img, sigma=hessian_sigma, order=(0, 2), mode=hessian_boundaries)
        gxy = gyx = gaussian_filter(img, sigma=hessian_sigma, order=(1, 1), mode=hessian_boundaries)
        gyy = gaussian_filter(img, sigma=hessian_sigma, order=(2, 0), mode=hessian_boundaries)

        return np.array([[gxx, gxy], [gyx, gyy]])


    #==========================================================================
    def compute_eigenvalues(self, tensor_field):
        """
        Compute the Hessian image given the components of the Hessian matrices.

        **Parameters:**

        ``tensor_field`` : 4D :py:class:`numpy.ndarray`
            The array containg the ``gxx``, ``gxy``, ``gyx``, ``gyy`` components of the Hessian
            matrices for each pixel in the input image.

        **Returns:**

        ``eig_img`` : 2D :py:class:`numpy.ndarray`
            Hessian image.
        """
        img_shape = tensor_field.shape[-2:]
        eig_img = np.zeros(img_shape)
        v_img = np.zeros((2,) + img_shape)

        for y in range(img_shape[0]):
            for x in range(img_shape[1]):
                # Eigendecomposition
                tensor = tensor_field[:, :, y, x]
                eigenvalues, eigenvectors = np.linalg.eigh(tensor)

                # Sort the eigenvalues and associated eigenvectors in decreasing order
                idx = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]

                # Biggest positive eigenvalue with dominant curvature
                eig_img[y, x] = max(eigenvalues[0], 0)

                # Keep tangential eigenvector if non-zero eigenvalue
                if eig_img[y, x] != 0:
                    if eigenvectors[1, 1] <= 0:
                        v_img[:, y, x] = eigenvectors[:, 1]

                    else:
                        v_img[:, y, x] = -eigenvectors[:, 1]

        return eig_img, v_img


    #==========================================================================
    def extract_keypoints(self, img):
        """
        Compute the AKAZE features given the Hessian image.

        **Parameters:**

        ``img`` : 2D :py:class:`numpy.ndarray`
            Input Hessian image.

        **Returns:**

        ``descriptors`` : 2D :py:class:`numpy.ndarray`
            Array conatining the AKAZE features. The dimensionality of the array:
            (N_features x Length_of_feature_vector).
        """
        # Image conversion
        img = np.uint8(img)

        # AKAZE descriptor
        detector = cv2.AKAZE_create(descriptor_type=4, nOctaves=3, nOctaveLayers=3, diffusivity=2)
        _, descriptors = detector.detectAndCompute(img, None)

        return descriptors


    #==========================================================================
    def get_akaze_features(self, img, mask,
                           smoothing_sigma, image_size, hessian_sigma, hessian_threshold, hessian_saturation):
        """
        Compute the AKAZE features given the input image and the binary mask of the ROI.

        **Parameters:**

        ``img`` : 2D :py:class:`numpy.ndarray`
            Input image.

        ``mask`` : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.

        ``smoothing_sigma`` : :py:class:`float`
            Parameter for the Gaussian filter, used in the preprocessing of the
            input image.

        ``image_size`` : :py:class:`list`
            The input image and mask will be rescaled to the dimensions
            specified in the ``image_size``.

        ``hessian_sigma`` : :py:class:`float`
            Standard deviation used for the Gaussian kernel.

        ``hessian_threshold`` : :py:class:`float`
            Values above this threshold are set to zero in the Hessian image.

        ``hessian_saturation`` : :py:class:`float`
            Saturation value for the Hessian image.

        **Returns:**

        ``descriptors`` : 2D :py:class:`numpy.ndarray`
            Array conatining the AKAZE features. The dimensionality of the array:
            (N_features x Length_of_feature_vector).
        """

        img = gaussian_filter(img, sigma = smoothing_sigma)

        img = imresize(img, image_size, interp="bicubic", mode="F")

        mask = imresize(mask, image_size, interp="bicubic", mode="F")
        mask[mask > 0.5] = 1
        mask[mask != 1] = 0
        mask = mask.astype(np.int)

        img = img.astype(np.float64)

        mask_img = self.center_image(mask, mask, b_is_mask=True)
#        roi_img = self.center_image(img, mask)
#        roi_img[mask_img == 0] = 0

        hessian = self.compute_hessian(img, hessian_sigma, "mirror")

        eig_img, _ = self.compute_eigenvalues(hessian)   # Vector field still not used for now

        eig_img = self.center_image(eig_img, mask)


        # Performing a thresholding and a saturation process
        eig_img[eig_img >= hessian_threshold] = 0
        max_pixel = hessian_saturation
        eig_img = exposure.rescale_intensity(eig_img, in_range=(0, max_pixel), out_range=(0, 255))
        eig_img[mask_img == 0] = 0


        # Extracting AKAZE feature
        descriptors = self.extract_keypoints(eig_img)

        return descriptors


    #==========================================================================
    def __call__(self, input_data):
        """
        Compute AKAZE features for the input data.

        **Parameters:**

        ``input_data`` : :py:class:`list`
            input_data[0] is an input image: 2D :py:class:`numpy.ndarray`
            input_data[1] is the binary mask of the ROI: 2D :py:class:`numpy.ndarray`

        **Returns:**

        ``descriptors`` : 2D :py:class:`numpy.ndarray`
            Array conatining the AKAZE features. The dimensionality of the array:
            (N_features x Length_of_feature_vector).
        """

        image = input_data[0] # Input image

        mask = input_data[1] # binary mask of the ROI

        descriptors = self.get_akaze_features(image, mask,
                                               self.smoothing_sigma, self.image_size,
                                               self.hessian_sigma, self.hessian_threshold, self.hessian_saturation)

        return descriptors



