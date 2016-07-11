#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from ...extractors import LocalBinaryPatterns


# Parameters
SIGMA_DERIVATES = 5 #Sigma used for determining derivatives

GPU_ACCELERATION = False

#Define feature extractor
feature_extractor = bob.fingervein.features.MaximumCurvature(
	sigma = SIGMA_DERIVATES,
	gpu = GPU_ACCELERATION

)

