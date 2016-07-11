#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import bob.fingervein

# Parameters
RADIUS_NEIGHBOURHOOD_REGION = 5 # Radius of the circular neighbourhood region
NEIGHBOURHOOD_THRESHOLD = 1
SUM_NEIGHBOURHOOD = 41 #Sum of neigbourhood threshold
RESCALE = True

#Define feature extractor
feature_extractor = bob.fingervein.features.WideLineDetector(
  radius = RADIUS_NEIGHBOURHOOD_REGION,
      threshold = NEIGHBOURHOOD_THRESHOLD,
  g = SUM_NEIGHBOURHOOD,
  rescale = RESCALE
)

