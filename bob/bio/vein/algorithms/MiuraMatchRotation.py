#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import numpy as np
import scipy.signal

from bob.bio.base.algorithm import Algorithm

from skimage import transform as tf

class MiuraMatchRotation(Algorithm):
    """Finger vein matching: match ratio via cross-correlation

    The method is based on "cross-correlation" between a model and a probe image.
    It convolves the binary image(s) representing the model with the binary image
    representing the probe (rotated by 180 degrees), and evaluates how they
    cross-correlate. If the model and probe are very similar, the output of the
    correlation corresponds to a single scalar and approaches a maximum. The
    value is then normalized by the sum of the pixels lit in both binary images.
    Therefore, the output of this method is a floating-point number in the range
    :math:`[0, 0.5]`. The higher, the better match.

    In case model and probe represent images from the same vein structure, but
    are misaligned, the output is not guaranteed to be accurate. To mitigate this
    aspect, Miura et al. proposed to add a *small* cropping factor to the model
    image, assuming not much information is available on the borders (``ch``, for
    the vertical direction and ``cw``, for the horizontal direction). This allows
    the convolution to yield searches for different areas in the probe image. The
    maximum value is then taken from the resulting operation. The convolution
    result is normalized by the pixels lit in both the cropped model image and
    the matching pixels on the probe that yield the maximum on the resulting
    convolution.

    For this to work properly, input images are supposed to be binary in nature,
    with zeros and ones.

    Based on N. Miura, A. Nagasaka, and T. Miyatake. Feature extraction of finger
    vein patterns based on repeated line tracking and its application to personal
    identification. Machine Vision and Applications, Vol. 15, Num. 4, pp.
    194--203, 2004

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

    ``score_fusion_method`` : :py:class:`str`
        Score fusion method.
        Default value: 'mean'.
        Possible options: 'mean', 'max', 'median'.
    """


    #==========================================================================
    def __init__(self, ch = 5, cw = 5,
                 angle_limit = 10, angle_step = 1,
                 score_fusion_method = 'mean'):

        # call base class constructor
        Algorithm.__init__(self,
                            ch = ch,
                            cw = cw,
                            angle_limit = angle_limit,
                            angle_step = angle_step,
                            score_fusion_method = score_fusion_method,
                            multiple_model_scoring = None,
                            multiple_probe_scoring = None)

        self.ch = ch
        self.cw = cw
        self.angle_limit = angle_limit
        self.angle_step = angle_step
        self.score_fusion_method = score_fusion_method


    #==========================================================================
    def enroll(self, enroll_features):
        """Enrolls the model by computing an average graph for each model"""

        # return the generated model
        return enroll_features


    #==========================================================================
    def comp_score_different_angles(self, image_enroll, image_probe, ch, cw, angle_limit, angle_step):
        """
        Rotate the probe image in the specified range computing the Miura
        match score for each angle. The returned score is the maximum of the
        computed scores.

        **Parameters:**

        ``image_enroll`` : 2D :py:class:`numpy.ndarray`
            Input enroll image.

        ``image_probe`` : 2D :py:class:`numpy.ndarray`
            Input probe image.

        ``ch`` : :py:class:`int`
            Maximum search displacement in y-direction.

        ``cw`` : :py:class:`int`
            Maximum search displacement in x-direction.

        ``angle_limit`` : :py:class:`float`
            Rotate the probe in the range [-angle_limit, +angle_limit] degrees.

        ``angle_step`` : :py:class:`float`
            Rotate the probe with this step in degrees.

        **Returns:**

        ``score`` : :py:class:`float`
            Similarity score.

        ``scores`` : :py:class:`list`
            A list of all scores.
        """

        image_probe_max = np.max(image_probe)

        angles = np.arange(-angle_limit, angle_limit + 1, angle_step)

        scores = []

        for angle in angles:

            image_probe_rotated =tf.rotate(image_probe, angle = angle, preserve_range = True)

            image_probe_rotated[ image_probe_rotated > 0.5 ] = image_probe_max
            image_probe_rotated = image_probe_rotated.astype(np.uint)
            image_probe_rotated = image_probe_rotated.astype(np.float64)

            h, w = image_enroll.shape

            crop_image_enroll = image_enroll[ch: h - ch, cw: w - cw]

            Nm = scipy.signal.fftconvolve(image_probe_rotated, np.rot90(crop_image_enroll, k=2), 'valid')

            t0, s0 = np.unravel_index(Nm.argmax(), Nm.shape)

            Nmm = Nm[t0,s0]

            scores.append(Nmm/(sum(sum(crop_image_enroll)) + sum(sum(image_probe_rotated[t0:t0+h-2*ch, s0:s0+w-2*cw]))))

        score = np.max(scores)

        return score, scores


    #==========================================================================
    def score(self, model, probe):
        """Computes the score between the probe and the model.

        Parameters:

          ``model`` (numpy.ndarray): The model of the user to test the probe agains

          ``probe`` (numpy.ndarray): The probe to test


        Returns:

          ``score`` (float): Value between 0 and 0.5, larger value means a better match

        """

        image_probe = probe.astype(np.float64)

#        if not isinstance(model, list):
#
#            model = [model] # this is necessary for unit tests only

        scores = []

        # iterate over all models for a given individual
        for enroll in model:

            image_enroll = enroll.astype(np.float64)

            data = self.comp_score_different_angles(image_enroll, image_probe,
                                                           self.ch, self.cw,
                                                           self.angle_limit, self.angle_step)

            scores.append(data[0])

        score_fused = getattr( np, self.score_fusion_method )(scores)

        return score_fused






















