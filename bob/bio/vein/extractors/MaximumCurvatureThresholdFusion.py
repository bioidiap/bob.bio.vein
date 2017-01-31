#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 09:24:13 2017

@author: onikisins
"""

from . import ThresholdExtractor

from . import MaximumCurvatureScaleRotation

import numpy as np

from bob.bio.base.extractor import Extractor

class MaximumCurvatureThresholdFusion(Extractor):
    """
    This class is a fusion of two vein extractors: ThresholdExtractor and
    MaximumCurvatureScaleRotation. The resulting vein image is a binary OR of the
    outputs of the above extractors.

    **Parameters:**

    ``sigma`` : :py:class:`int`
        Sigma used for determining derivatives.
        Default: 5.

    ``norm_p2p_dist_flag`` : :py:class:`bool`
        If ``True`` normalize the mean distance between the point pairs in the
        input binary image to the ``selected_mean_dist`` value.
        Default: ``False``.

    ``selected_mean_dist`` : :py:class:`float`
        Normalize the mean distance between the point pairs in the
        input binary image to this value.
        Default: 100.

    ``name`` : :py:class:`str`
        Name of predefinied extractor.

    ``median`` : :py:class:`bool`
        Flag to indicate, if Median filter is applied to output. Default -
        False.

    ``size`` : :py:class:`int`
        Size of median filter. Default - 5.
    """

    #==========================================================================
    def __init__(self, sigma = 5,
                 norm_p2p_dist_flag = False, selected_mean_dist = 100,
                 name = 'Adaptive_ski_25_3_50',
                 median = False,
                 size = 5):

        Extractor.__init__(self,
                           sigma = sigma,
                           norm_p2p_dist_flag = norm_p2p_dist_flag,
                           selected_mean_dist = selected_mean_dist,
                           name = name,
                           median = median,
                           size = size)

        self.sigma = sigma
        self.norm_p2p_dist_flag = norm_p2p_dist_flag
        self.selected_mean_dist = selected_mean_dist
        self.name = name
        self.median = median
        self.size = int(size)


    #==========================================================================
    def __call__(self, input_data):
        """
        Compute the binary pattern of the veins as a combination of the
        outputs of two vein extractors: hresholdExtractor and
        MaximumCurvatureScaleRotation.

        **Parameters:**

        ``input_data`` : tuple
            input_data[0] is an input image: 2D `numpy.ndarray`
            input_data[1] is the binary mask of the ROI: 2D `numpy.ndarray`

        **Returns:**

        ``image_norm`` : 2D :py:class:`numpy.ndarray`
            Binary image of the veins.
        """

        image = input_data[0]
        mask = input_data[1]

        extractor_mc = MaximumCurvatureScaleRotation(self.sigma,
                                                     norm_p2p_dist_flag = False, selected_mean_dist = 100,
                                                     sum_of_rotated_images_flag = False, angle_limit = 10, angle_step = 1)

        extractors_threshold = ThresholdExtractor(name = self.name, median = self.median, size = self.size,
                                                  thin_veins_flag = True)


        binary_image_mc = extractor_mc([image, mask])

        binary_image_threshold = extractors_threshold([image, mask])

        binary_image = np.bitwise_or(binary_image_mc.astype(np.uint8),
                                     binary_image_threshold.astype(np.uint8)).astype(np.float)

        if self.norm_p2p_dist_flag:

            scale = extractor_mc.find_scale(binary_image, self.selected_mean_dist)

            binary_image = extractor_mc.scale_binary_image(binary_image, scale)

        return binary_image










































