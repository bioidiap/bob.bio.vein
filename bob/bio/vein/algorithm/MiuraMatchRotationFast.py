#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:02:17 2017

@author: onikisins
"""

import numpy as np
import scipy.signal
from scipy import ndimage
from skimage import morphology
from skimage import transform as tf

from bob.bio.base.algorithm import Algorithm


#==============================================================================
class MiuraMatchRotationFast (Algorithm):
    """

    This method is an enhancement of Miura Matching algorithm introduced in:

    Based on N. Miura, A. Nagasaka, and T. Miyatake. Feature extraction of finger
    vein patterns based on repeated line tracking and its application to personal
    identification. Machine Vision and Applications, Vol. 15, Num. 4, pp.
    194--203, 2004

    The algorithm is designed to compensate both rotation and translation
    in a computationally efficient manner, prior to score computation.

    This is achieved computing the cross-correlation of enrolment and probe samples 
    twice. In the first pass the probe is cross-correlated with an image, which is
    the sum of pre-rotated enroll images. 
    This makes the cross-correlation robust to the rotation in the certain 
    range of angles, and with some additional steps helps to define the angle between 
    enrolment and probe samples. The angular range is defined by the ``angle_limit`` 
    parameter of the algorithm.

    Next, the enrolled image is rotated by the obtained angle, thus compensating the 
    angle between the enrolment and probe samples. After that, the ordinary Miura
    matching algorithm is applied. 

    The matching of both binary and gray-scale vein patterns is possible. 
    Set the ``gray_scale_input_flag`` to ``True`` if the input is gray-scale.

    The details of the this algorithm are introduced in the following paper:

    Olegs Nikisins, Andre Anjos, Teodors Eglitis, Sebastien Marcel.
    Fast cross-correlation based wrist vein recognition algorithm with 
    rotation and translation compensation. 


    **Parameters:**

    ``ch`` : :py:class:`int`
        Maximum search displacement in y-direction.
        Default value: 5.

    ``cw`` : :py:class:`int`
        Maximum search displacement in x-direction.
        Default value: 5.

    ``angle_limit`` : :py:class:`float`
        Rotate the probe in the range [-angle_limit, +angle_limit] degrees.
        Default value: 10.

    ``angle_step`` : :py:class:`float`
        Rotate the probe with this step in degrees.
        Default value: 1.

    ``perturbation_matching_flag`` : :py:class:`bool`
        Compute the score using perturbation_matching method of the class.
        Default: ``False``.

    ``kernel_radius`` : :py:class:`int`
        Radius of the circular kernel used in the morphological dilation of
        the enroll. Only valid when ``perturbation_matching_flag`` is ``True``.
        Default: 3.

    ``score_fusion_method`` : :py:class:`str`
        Score fusion method.
        Default value: 'mean'.
        Possible options: 'mean', 'max', 'median'.

    ``gray_scale_input_flag`` : :py:class:`bool`
        Set this flag to ``True`` if image is grayscale. Defaults: ``False``.
    """


    #==========================================================================
    def __init__(self, ch = 5, cw = 5,
                 angle_limit = 10, angle_step = 1,
                 perturbation_matching_flag = False,
                 kernel_radius = 3,
                 score_fusion_method = 'mean',
                 gray_scale_input_flag = False):

        # call base class constructor
        Algorithm.__init__(self,
                            ch = ch,
                            cw = cw,
                            angle_limit = angle_limit,
                            angle_step = angle_step,
                            perturbation_matching_flag = perturbation_matching_flag,
                            kernel_radius = kernel_radius,
                            score_fusion_method = score_fusion_method,
                            gray_scale_input_flag = gray_scale_input_flag,
                            multiple_model_scoring = None,
                            multiple_probe_scoring = None)

        self.ch = ch
        self.cw = cw
        self.angle_limit = angle_limit
        self.angle_step = angle_step
        self.perturbation_matching_flag = perturbation_matching_flag
        self.kernel_radius = kernel_radius
        self.score_fusion_method = score_fusion_method
        self.gray_scale_input_flag = gray_scale_input_flag


    #==========================================================================
    def enroll(self, enroll_features):
        """Enrolls the model by computing an average graph for each model"""

        # return the generated model
        return enroll_features


    #==========================================================================
    def perturbation_matching(self, enroll, probe, kernel_radius):
        """
        Compute the matching score as a normalized intersection of the enroll and
        probe allowing perturbation of the enroll in the computation
        of the intersection.

        **Parameters:**

        ``enroll`` : 2D :py:class:`numpy.ndarray`
            Binary image of the veins representing the enroll.

        ``probe`` : 2D :py:class:`numpy.ndarray`
            Binary image of the veins representing the probe.

        ``kernel_radius`` : :py:class:`int`
            Radius of the circular kernel used in the morphological dilation of
            the enroll.

        **Returns:**

        ``score`` : :py:class:`float`
            Natching score, larger value means a better match.
        """

        ellipse_kernel = morphology.disk(radius = kernel_radius)

        enroll_dilated = ndimage.morphology.binary_dilation(enroll, structure = ellipse_kernel).astype(np.float)

        probe_dilated = ndimage.morphology.binary_dilation(probe, structure = ellipse_kernel).astype(np.float)

        normalizer = np.sum(enroll_dilated) + np.sum(probe_dilated)

        score = np.sum( enroll_dilated * probe_dilated ) / normalizer

        return score


    #==========================================================================
    def miura_match(self, image_enroll, image_probe, ch, cw, compute_score_flag = True,
                    perturbation_matching_flag = False, kernel_radius = 3):
        """
        Match two binary vein images using Miura matching algorithm.

        **Parameters:**

        ``image_enroll`` : 2D :py:class:`numpy.ndarray`
            Binary image of the veins representing the model.

        ``image_probe`` : 2D :py:class:`numpy.ndarray`
            Probing binary image of the veins.

        ``ch`` : :py:class:`int`
            Cropping parameter in Y-direction.

        ``cw`` : :py:class:`int`
            Cropping parameter in X-direction.

        ``compute_score_flag`` : :py:class:`bool`
            Compute the score if True. Otherwise only the ``crop_image_probe``
            is returned. Default: ``True``.

        ``perturbation_matching_flag`` : :py:class:`bool`
            Compute the score using perturbation_matching method of the class.
            Only valid if ``compute_score_flag`` is set to ``True``.
            Default: ``False``.

        ``kernel_radius`` : :py:class:`int`
            Radius of the circular kernel used in the morphological dilation of
            the enroll.

        **Returns:**

        ``score`` : :py:class:`float`
            Natching score between 0 and 0.5, larger value means a better match.
            Only returned if ``compute_score_flag`` is set to ``True``.

        ``crop_image_probe`` : 2D :py:class:`numpy.ndarray`
            Cropped binary image of the probe.
        """

        if image_enroll.dtype != np.float64:
            image_enroll = image_enroll.astype(np.float64)

        if image_probe.dtype != np.float64:
            image_probe = image_probe.astype(np.float64)

        h, w = image_enroll.shape

        crop_image_enroll = image_enroll[ ch : h - ch, cw : w - cw ]

        Nm = scipy.signal.fftconvolve(image_probe, np.rot90(crop_image_enroll, k=2), 'valid')

        t0, s0 = np.unravel_index(Nm.argmax(), Nm.shape)

        Nmm = Nm[t0,s0]

        crop_image_probe = image_probe[ t0: t0 + h - 2 * ch, s0: s0 + w - 2 * cw ]

        return_data = crop_image_probe

        if compute_score_flag:

            if perturbation_matching_flag:

                score = self.perturbation_matching(crop_image_enroll, crop_image_probe, kernel_radius)

            else:

                score = Nmm / ( np.sum( crop_image_enroll ) + np.sum( crop_image_probe ) )

            return_data = ( score, crop_image_probe )

        return return_data


    #==========================================================================
    def sum_of_rotated_images(self, image, angle_limit, angle_step, gray_scale_input_flag):
        """
        Generate the output image, which is the sum of input images rotated
        in the specified range with the defined step.

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            Input image.

        ``angle_limit`` : :py:class:`float`
            Rotate the image in the range [-angle_limit, +angle_limit] degrees.

        ``angle_step`` : :py:class:`float`
            Rotate the image with this step in degrees.

        ``gray_scale_input_flag`` : :py:class:`bool`
            Set this flag to ``True`` if image is grayscale. Defaults: ``False``.

        **Returns:**

        ``output_image`` : 2D :py:class:`numpy.ndarray`
            Sum of rotated images.

        ``rotated_images`` : 3D :py:class:`numpy.ndarray`
            A stack of rotated images. Array size:
            (N_images, Height, Width)
        """

        offset = np.array(image.shape)/2

        h, w = image.shape

        image_coords = np.argwhere(image) - offset # centered coordinates of the vein (non-zero) pixels

        if gray_scale_input_flag:

            image_val = image[image>0]

        angles = np.arange(-angle_limit, angle_limit + 1, angle_step) / 180. * np.pi # angles in the radians

        rotated_images = np.zeros( (angles.shape[0], image.shape[0], image.shape[1]) )

        for idx, angle in enumerate(angles):

            rot_matrix = np.array([[np.cos(angle), - np.sin(angle)],
                                   [np.sin(angle),  np.cos(angle)]]) # rotation matrix

            rotated_coords = np.round( np.dot( image_coords, rot_matrix ) ).astype(np.int) + offset

            rotated_coords[rotated_coords < 0] = 0
            rotated_coords[:, 0][rotated_coords[:, 0] >= h] = h-1
            rotated_coords[:, 1][rotated_coords[:, 1] >= w] = w-1

            rotated_coords = rotated_coords.astype(np.int)

            if gray_scale_input_flag:

                rotated_images[idx, rotated_coords[:,0], rotated_coords[:,1]] = image_val

            else:

                rotated_images[idx, rotated_coords[:,0], rotated_coords[:,1]] = 1

        output_image = np.sum(rotated_images, axis = 0)

        return output_image, rotated_images


    #==========================================================================
    def score(self, model, probe):
        """Computes the score between the probe and the model.

        **Parameters:**

        ``model`` : 2D :py:class:`numpy.ndarray`
            Binary image of the veins representing the model.

        ``probe`` : 2D :py:class:`numpy.ndarray`
            Probing binary image of the veins.

        **Returns:**

        ``score_fused`` : :py:class:`float`
            Natching score between 0 and 0.5, larger value means a better match.
        """

        if probe.dtype != np.float64:
            probe = probe.astype(np.float64)

        scores = []

        angles = np.arange(-self.angle_limit, self.angle_limit + 1, self.angle_step)

        # iterate over all models for a given individual
        for enroll in model:

            if enroll.dtype != np.float64:
                enroll = enroll.astype(np.float64)

            sum_of_rotated_img_enroll, rotated_images_enroll = self.sum_of_rotated_images(enroll, self.angle_limit, self.angle_step,
                                                                                          self.gray_scale_input_flag)

            h, w = enroll.shape

            crop_rotated_images_enroll = rotated_images_enroll[:, self.ch: h - self.ch, self.cw: w - self.cw]

            crop_probe = self.miura_match(sum_of_rotated_img_enroll, probe, self.ch, self.cw, compute_score_flag = False)

            scores_internal = []

            for crop_binary_image_enroll in crop_rotated_images_enroll:

                scores_internal.append( np.sum(crop_binary_image_enroll * crop_probe) )

            idx_selected = np.argmax(scores_internal) # the index of the rotated enroll image having the best match

            if self.gray_scale_input_flag:

                angle = angles[idx_selected]

                enroll_rotated =tf.rotate(enroll, angle = -angle, preserve_range = True)

                score = self.miura_match(enroll_rotated, probe, self.ch, self.cw, compute_score_flag = True,
                                     perturbation_matching_flag = False,
                                     kernel_radius = self.kernel_radius)[0]

            else:

                score = self.miura_match(rotated_images_enroll[ idx_selected ], probe, self.ch, self.cw, compute_score_flag = True,
                                         perturbation_matching_flag = self.perturbation_matching_flag,
                                         kernel_radius = self.kernel_radius)[0]

            scores.append( score )

        score_fused = getattr( np, self.score_fusion_method )(scores)

        return score_fused


