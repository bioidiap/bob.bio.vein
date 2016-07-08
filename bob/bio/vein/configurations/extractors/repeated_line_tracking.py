#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Pedro Tome <Pedro.Tome@idiap.ch>

import bob.fingervein

# Parameters
NUMBER_ITERATIONS = 3000 	# Maximum number of iterations
DISTANCE_R = 1 			# Distance between tracking point and cross section of the profile
PROFILE_WIDTH = 21 		# Width of profile


#Define feature extractor
feature_extractor = bob.fingervein.features.RepeatedLineTracking(
	iterations = NUMBER_ITERATIONS, 
	r = DISTANCE_R,             
      	profile_w = PROFILE_WIDTH            
)

