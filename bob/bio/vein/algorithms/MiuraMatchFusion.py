#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import numpy
import scipy.signal

from bob.bio.base.algorithm import Algorithm

import numpy as np


class MiuraMatchFusion (Algorithm):
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

    Parameters:

      ch (:py:class:`int`, optional): Maximum search displacement in y-direction.

      cw (:py:class:`int`, optional): Maximum search displacement in x-direction.

      score_fusion_method (:py:class:`str`, optional): The strategy to be used for score fusion.

    """

    def __init__(self,
          ch = 8,       # Maximum search displacement in y-direction
          cw = 5,       # Maximum search displacement in x-direction
          score_fusion_method = 'mean' # The strategy to be used for score fusion
          ):

        # call base class constructor
        Algorithm.__init__(
            self,

            ch = ch,
            cw = cw,
            score_fusion_method = score_fusion_method,

            multiple_model_scoring = None,
            multiple_probe_scoring = None
        )

        self.ch = ch
        self.cw = cw
        self.score_fusion_method = score_fusion_method
        self.available_score_fusion_methods = ['mean', 'max', 'median', 'adaptive_mean']


    def enroll(self, enroll_features):
        """Enrolls the model by computing an average graph for each model"""

        # return the generated model
        return numpy.array(enroll_features)


    def fuse_scores(self, scores, score_fusion_method):
        """
        Method to fuse the scores using different strategies.

        Parameters:

          scores (:py:class:`list`): A list with scores.

          score_fusion_method (:py:class:`str`): The strategy to be used for score fusion.


        Returns:

          score (:py:class:`float`): Resulting similarity score.
        """

        if score_fusion_method == 'mean':

            score = numpy.mean(scores)

        if score_fusion_method == 'max':

            score = numpy.max(scores)

        if score_fusion_method == 'median':

            score = numpy.median(scores)

        if score_fusion_method == 'adaptive_mean':

            scores = np.array(scores)

            scores = np.sort(scores)

            if len(scores) >= 2:

                diff = scores[1:] - scores[0:-1]

                selected_scores = scores[np.argmax(diff)+1:]

                score = np.mean(selected_scores)

            else:

                score = np.max(scores)

        return score


    def score(self, model, probe):
        """Computes the score between the probe and the model.

        Parameters:

          model (numpy.ndarray): The model of the user to test the probe agains

          probe (numpy.ndarray): The probe to test


        Returns:

          score (float): Value between 0 and 0.5, larger value means a better match

        """

        if not( self.score_fusion_method in self.available_score_fusion_methods ):
            raise Exception("Specified score fusion approach is not in the list of available_score_fusion methods")

        I=probe.astype(numpy.float64)

        if len(model.shape) == 2:
          model = numpy.array([model])

        scores = []

        # iterate over all models for a given individual
        for md in model:

          # erode model by (ch, cw)
          R = md.astype(numpy.float64)
          h, w = R.shape
          crop_R = R[self.ch:h-self.ch, self.cw:w-self.cw]

          # correlates using scipy - fastest option available iff the self.ch and
          # self.cw are height (>30). In this case, the number of components
          # returned by the convolution is high and using an FFT-based method
          # yields best results. Otherwise, you may try  the other options bellow
          # -> check our test_correlation() method on the test units for more
          # details and benchmarks.
          Nm = scipy.signal.fftconvolve(I, numpy.rot90(crop_R, k=2), 'valid')
          # 2nd best: use convolve2d or correlate2d directly;
          # Nm = scipy.signal.convolve2d(I, numpy.rot90(crop_R, k=2), 'valid')
          # 3rd best: use correlate2d
          # Nm = scipy.signal.correlate2d(I, crop_R, 'valid')

          # figures out where the maximum is on the resulting matrix
          t0, s0 = numpy.unravel_index(Nm.argmax(), Nm.shape)

          # this is our output
          Nmm = Nm[t0,s0]

          # normalizes the output by the number of pixels lit on the input
          # matrices, taking into consideration the surface that produced the
          # result (i.e., the eroded model and part of the probe)
          scores.append(Nmm/(sum(sum(crop_R)) + sum(sum(I[t0:t0+h-2*self.ch, s0:s0+w-2*self.cw]))))

        score = self.fuse_scores(scores, self.score_fusion_method)

        return score

