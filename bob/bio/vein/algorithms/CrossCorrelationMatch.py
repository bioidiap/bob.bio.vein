#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 13:55:22 2016

@author: onikisins
"""

#==============================================================================
# Import what is needed here:
from bob.bio.base.algorithm import Algorithm

import numpy as np

from skimage import transform as tf

import bob.io.base

#==============================================================================
# Class implementation:
class CrossCorrelationMatch(Algorithm):
    """
    This class is designed to match the vein images using cross-correlation
    based method.
    The matching is composed of the following steps:

        1. First the ``enroll`` and ``probe`` images are mean normalized if
           if ``is_mean_normalize_flag`` is set to True.
        2. The probe data (image and binary mask of the ROI) can also be rotated,
           but this option is currently unused.
        3. The probe is next alligned to the enroll using cross-correlation.
        4. After the alignment the similarity score is computed as the normalized
           cross-correlation score. Only the joint ROI is considered in the
           normalization process.

    **Parameters:**

    ``is_mean_normalize_flag`` : :py:class:`bool`
        Set this flag to False if data to be aligned is already mean-normalized.
        Otherwise this flag must be True.

    ``score_fusion_method`` : :py:class:`str`
        The score fusion strategy. Possible values: mean, max, median.
    """


    #==========================================================================
    def __init__(self, is_mean_normalize_flag, score_fusion_method):

        Algorithm.__init__( self,
                           is_mean_normalize_flag = is_mean_normalize_flag,
                           score_fusion_method = score_fusion_method)

        self.is_mean_normalize_flag = is_mean_normalize_flag
        self.score_fusion_method = score_fusion_method


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

        image_average = np.average(image, weights = mask)

    #    image_std = np.sqrt( np.average((image - image_average)**2, weights = mask) )

        image_normalized = ( image - image_average ) * mask

        image_std = np.sqrt( np.average(image_normalized**2, weights = mask) )

        image_normalized = image_normalized / image_std

        return image_normalized


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
    def allign_hessian_eigenval_images(self, enroll, probe, enroll_mask, probe_mask, angle, is_mean_normalize_flag):
        """
        Align the enroll and probe using the cross-correlation. After alignment
        compute the similarity score of two images as a normalized cross-correlation
        score. Only the joint ROI is taken into account in score computation.

        **Parameters:**

        ``enroll`` : 2D :py:class:`numpy.ndarray`
            Image (max eigenvalues of hessian matrices)
            of the veins representing the enroll.

        ``probe`` : 2D :py:class:`numpy.ndarray`
            Image (max eigenvalues of hessian matrices)
            of the veins representing the probe.

        ``enroll_mask`` : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI of the enroll.

        ``probe_mask`` : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI of the probe.

        ``angle`` : :py:class:`float`
            Rotate probe by this angle counter clockwise before cross-correlation
            based matching. The angle is in degrees.

        ``is_mean_normalize_flag`` : :py:class:`bool`
            Set this flag to False if data to be aligned is already mean-normalized.
            Otherwise this flag must be True.

        **Returns:**

        ``score`` : :py:class:`float`
            The similarity score (maximum of the normalized cross-correlation).

        ``enroll_updated`` : 2D :py:class:`numpy.ndarray`
            Mean-std normalized image representing the enroll in the jount ROI.

        ``probe_updated`` : 2D :py:class:`numpy.ndarray`
            Mean-std normalized image representing the probe
            in the jount ROI after the alignment.

        ``cc_image`` : 2D :py:class:`numpy.ndarray`
            Image representing the result of cross-correlation image of enroll and probe.
        """

        if is_mean_normalize_flag:
            image_enroll = self.mean_normalization(enroll, enroll_mask)
            image_probe = self.mean_normalization(probe, probe_mask)
        else:
            image_enroll = enroll
            image_probe = probe

        probe_rotated = tf.rotate(image_probe, angle = angle,  preserve_range = True)

        probe_mask_rotated = tf.rotate(probe_mask, angle = angle, preserve_range = True)

        probe_mask_rotated[probe_mask_rotated>0.1] = 1
        probe_mask_rotated = probe_mask_rotated.astype(np.uint8)

        image_product = np.fft.fft2(image_enroll) * np.fft.fft2(probe_rotated).conj()
        cc_image = np.fft.fftshift(np.fft.ifft2(image_product)).real

        offset = np.array(cc_image.shape)/2 - np.unravel_index(cc_image.argmax(), cc_image.shape)
        tform = tf.SimilarityTransform(scale=1, rotation=0, translation=(offset[1], offset[0]))

        probe_rotated_translated = tf.warp(probe_rotated, tform, preserve_range = True)

        probe_mask_rotated_translated = tf.warp(probe_mask_rotated, tform, preserve_range = True)

        probe_mask_rotated_translated[probe_mask_rotated_translated>0.1] = 1
        probe_mask_rotated_translated = probe_mask_rotated_translated.astype(np.uint8)

        joint_roi = probe_mask_rotated_translated*enroll_mask

        joint_roi = joint_roi.astype(np.uint8)

        if np.sum(joint_roi) > 100:

            enroll_updated = self.mean_std_normalization(image_enroll, joint_roi)

            probe_updated = self.mean_std_normalization(probe_rotated_translated, joint_roi)

            score = np.average( enroll_updated * probe_updated, weights = joint_roi)

        else:

            enroll_updated = image_enroll

            probe_updated = probe_rotated_translated

            score = 0

        return score, enroll_updated, probe_updated, cc_image


    #==========================================================================
    def enroll( self, enroll_features ):
        """enroll(enroll_features) -> model

        This function will enroll and return the model from the given list of features.
        It must be overwritten by derived classes.

        **Parameters:**

        ``enroll_features`` : [object]
            A list of features used for the enrollment of one model.

        **Returns:**

        ``model`` : object
            The model enrolled from the ``enroll_features``.
            Must be writable with the :py:meth:`write_model` function and readable with the :py:meth:`read_model` function.
        """

        return enroll_features # Just leave as is


    #==========================================================================
    def score(self, model, probe):
        """
        score(model, probe) -> score

        This function computes the similarity score of the ``enroll`` and ``probe``
        using the normalized cross-correlation method.

        **Parameters:**

        ``model`` : :py:class:`list` or a 2D :py:class:`numpy.ndarray`
            The model enrolled by the :py:meth:`enroll` function.

        ``probe`` : 2D :py:class:`numpy.ndarray`
            The probe read by the :py:meth:`read_probe` function.

        **Returns:**

        ``score_fused`` : :py:class:`float`
            The resulting similarity score.
        """

        scores = []

        if not isinstance(model, list):

            model = [model] # this is necessary for unit tests only

        for enroll in model:

            data = self.allign_hessian_eigenval_images(enroll[0], probe[0], enroll[1], probe[1], 0, self.is_mean_normalize_flag)

            score = data[0]

            scores.append( score )

        # Options here are mean, median, max, min
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

        ``max_eigenvalues`` : 2D :py:class:`numpy.ndarray`
            Binary mask of the ROI.
        """

        f = bob.io.base.HDF5File( probe_file, 'r' )
        max_eigenvalues = f.read( 'max_eigenvalues' )
        mask = f.read( 'mask' )
        del f

        return max_eigenvalues, mask

























