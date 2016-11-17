#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:53:24 2016

@author: onikisins
"""

#==============================================================================
# Import what is needed here:

from bob.bio.base.algorithm import Algorithm

import cv2

import numpy as np

#==============================================================================
# Class implementation:

class KeypointsMatcher(Algorithm):
    """
    This class is designed to match two sets of keypoints (AKAZE features for example).

    **Parameters:**

    ``ratio_to_match`` : :py:class:`float`
        Parameter used in the selection of the stable key-points.
    """


    def __init__(self, ratio_to_match):

        Algorithm.__init__( self,
                           ratio_to_match = ratio_to_match)

        self.ratio_to_match = ratio_to_match


    #==========================================================================
    def match_keypoints(self, descriptors1, descriptors2, ratio_to_match):
        """
        Match two sets of keypoints. The matching score is ratio of selected/stable
        keypoints to the total number of keypoints in both sets.

        **Returns:**

        ``descriptors1`` : 2D :py:class:`numpy.ndarray`
            Array conatining the AKAZE features for the enroll. The dimensionality of the array:
            (N_features x Length_of_feature_vector).

        ``descriptors2`` : 2D :py:class:`numpy.ndarray`
            Array conatining the AKAZE features for the probe. The dimensionality of the array:
            (N_features x Length_of_feature_vector).

        **Returns:**

        ``score`` : :py:class:`float`
            The similarity score.
        """
        # Match the features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Select good matches
        n_matches = 0
        for m, n in matches:
            if m.distance < ratio_to_match * n.distance:
                n_matches += 1

        # Compute ratio between matching keys and total numbers
        n_keys1 = descriptors1.shape[0]
        n_keys2 = descriptors2.shape[0]

        score = 2.0 * n_matches / float(n_keys1 + n_keys2)

        return score


    #==========================================================================
    def enroll(self, enroll_features):
        """
        enroll(enroll_features) -> model

        This function will enroll and return the model from the given list of features.
        It must be overwritten by derived classes.

        **Parameters:**

        enroll_features : [object]
            A list of features used for the enrollment of one model.

        **Returns:**

        model : object
            The model enrolled from the ``enroll_features``.
            Must be writable with the :py:meth:`write_model` function and readable with the :py:meth:`read_model` function.
        """

        # Find the height of the largest array of features:
        height_max = np.max([item.shape[0] for item in enroll_features])

        # Pad the array of features with zeros in the bootom to make them of the same shape:
        enroll_features_padded = [np.vstack( [ item, np.zeros( (height_max-item.shape[0], item.shape[1]) ) ] ) for item in enroll_features]

        if isinstance(enroll_features_padded, list):
            enroll_features_padded = np.array(enroll_features_padded)

        return enroll_features_padded # Do nothing in our case


    #==========================================================================
    def score(self, model, probe):
        """
        score(model, probe) -> score

        Computes the score of the probe and the model using ``match_keypoints()``
        method of this class

        **Parameters:**

        model : 2D/3D :py:class:`numpy.ndarray`
            The model enrolled by the :py:meth:`enroll` function.

        probe : 2D :py:class:`numpy.ndarray`
            The probe read by the :py:meth:`read_probe` function.

        **Returns:**

        score_mean : float
            The resulting similarity score.
        """

        scores = []

        if isinstance(model, np.ndarray):

            if len( model.shape ) == 2:

                model = [ model ] # this is necessary for unit tests only

            else:

                num_models = model.shape[0] # number of enroll samples

                model = np.split( model, num_models, 0 ) # split 3D array into a list of 2D arrays of dimensions: (1,H,W)

        model = [ np.squeeze( item ) for item in model ] # remove single-dimensional entries from the shape of an array

        # Remove vectors of zeros, which were substituted in the enroll() function:
        model = [ item[0: item.shape[0]-np.argwhere(np.sum(item, 1)==0).shape[0], :] for item in model ]

        model = [ np.uint8(item) for item in model ] # Convert to the proper data type

        for enroll in model:

            score = self.match_keypoints(enroll, probe, self.ratio_to_match)

            scores.append(score)

        score_mean = np.mean( scores )

        return score_mean


