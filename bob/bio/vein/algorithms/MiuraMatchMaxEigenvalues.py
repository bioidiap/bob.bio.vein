#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import numpy as np
import scipy.signal

from bob.bio.base.algorithm import Algorithm

import bob.io.base


class MiuraMatchMaxEigenvalues (Algorithm):
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

    """


    #==========================================================================
    def __init__(self, ch = 5, cw = 5, score_fusion_method = 'mean'):

        # call base class constructor
        Algorithm.__init__(self,
                            ch = ch,
                            cw = cw,
                            score_fusion_method = score_fusion_method,
                            multiple_model_scoring = None,
                            multiple_probe_scoring = None)

        self.ch = ch
        self.cw = cw
        self.score_fusion_method = score_fusion_method


    #==========================================================================
    def enroll(self, enroll_features):
        """Enrolls the model by computing an average graph for each model"""

        # return the generated model
        return enroll_features


    #==========================================================================
    def mean_std_normalization(self, image, mask):
        """
        Perform mean-std normalization of the input image given weights in the mask
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

        image[np.isnan(image)] = 0 # set NaN elements to zero

        image_average = np.average(image, weights = mask)

        image_normalized = ( image - image_average ) * mask

        image_std = np.sqrt( np.average(image_normalized**2, weights = mask) )

        image_normalized = image_normalized / image_std

        return image_normalized


    #==========================================================================
    def score(self, model, probe):
        """Computes the score between the probe and the model.

        Parameters:

          ``model`` (numpy.ndarray): The model of the user to test the probe agains

          ``probe`` (numpy.ndarray): The probe to test


        Returns:

          ``score`` (float): Value between 0 and 0.5, larger value means a better match

        """

        image_probe = probe[0].astype(np.float64)

        mask_probe = probe[1]

#        image_probe = self.mean_std_normalization(image_probe, mask_probe)

        if not isinstance(model, list):

            model = [model] # this is necessary for unit tests only

        scores = []

        # iterate over all models for a given individual
        for enroll in model:

            image_enroll = enroll[0].astype(np.float64)

            mask_enroll = enroll[1]

            h, w = image_enroll.shape

            crop_image_enroll = image_enroll[self.ch:h-self.ch, self.cw:w-self.cw]

            crop_mask_enroll = mask_enroll[self.ch:h-self.ch, self.cw:w-self.cw]

#            crop_image_enroll = self.mean_std_normalization(crop_image_enroll, crop_mask_enroll)

            Nm = scipy.signal.fftconvolve(image_probe, np.rot90(crop_image_enroll, k=2), 'valid')

            t0, s0 = np.unravel_index(Nm.argmax(), Nm.shape)

            Nmm = Nm[t0,s0]

            scores.append( Nmm / ( np.sum( crop_image_enroll ) + np.sum( image_probe[t0:t0+h-2*self.ch, s0:s0+w-2*self.cw] ) ) )

        score_fused = getattr( np, self.score_fusion_method )(scores)

        return score_fused


    #==========================================================================
    def write_model(self, model, model_file):
        """
        Writes the enrolled model to the given file.
        In this base class implementation:

        - If the given model has a 'save' attribute, it calls ``model.save(bob.io.base.HDF5File(model_file), 'w')``.
          In this case, the given model_file might be either a file name or a :py:class:`bob.io.base.HDF5File`.
        - Otherwise, it uses :py:func:`bob.io.base.save` to do that.

        If you have a different format, please overwrite this function.

        **Parameters:**

        ``model`` : object
          A model as returned by the :py:meth:`enroll` function, which should be written.

        ``model_file`` : str or :py:class:`bob.io.base.HDF5File`
          The file open for writing, or the file name to write to.
        """

        # Save the model in the HDF5File
        f = bob.io.base.HDF5File(model_file, 'w')

        for enroll in model:

            f.append("images", enroll[0])
            f.append("masks", enroll[1])

        del f


    #==========================================================================
    def read_model(self, model_file):
        """
        read_model(model_file) -> model

        Loads the enrolled model from file.
        In this base class implementation, it uses :py:func:`bob.io.base.load` to do that.

        If you have a different format, please overwrite this function.

        **Parameters:**

        ``model_file`` : str or :py:class:`bob.io.base.HDF5File`
          The file open for reading, or the file name to read from.

        **Returns:**

        ``model`` : object
          The model that was read from file.
        """

        # Load the model from the HDF5File
        f = bob.io.base.HDF5File(model_file, 'r')

        images = f.read( 'images' )

        masks = f.read( 'masks' )

        model = []

        for image, mask in zip(images, masks):

            model.append( [image, mask] )

        del f

        return model


    #==========================================================================
    def read_probe( self, probe_file ):
        """
        Reads the probe feature from file.

        **Parameters:**

        ``probe_file`` : str or :py:class:`bob.io.base.HDF5File`
          The file open for reading, or the file name to read from.

        **Returns:**

        ``max_eigenvalues`` : 2D :py:class:`numpy.ndarray`
            Maximum eigenvalues of Hessian matrices.

        ``mask`` : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.
        """

        f = bob.io.base.HDF5File( probe_file, 'r' )
        max_eigenvalues = f.read( 'image' )
        mask = f.read( 'mask' )
        del f

        return max_eigenvalues, mask




















